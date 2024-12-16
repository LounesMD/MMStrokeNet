import sys
import os
import shutil
import re
import tempfile
import logging
import configparser as ConfParser
import subprocess
from pathlib import Path
from collections import OrderedDict
import json
import yaml
from dotenv import load_dotenv
import SimpleITK as sitk
import numpy as np
Path.ls = lambda x: sorted(list(x.iterdir()))

anima_config_file_path = Path().home() / '.anima' / 'config.txt'

# Open the configuration parser and exit if anima configuration cannot be loaded
if not anima_config_file_path.exists():
	sys.exit('The anima configuration file ' + str(anima_config_file_path) + ' does not exists. Please follow the anima script installation instructions (https://anima.readthedocs.io/en/latest/install_anima_scripts.html).')

config_parser = ConfParser.RawConfigParser()
config_parser.read(anima_config_file_path)

# Initialize anima directories
anima = Path(config_parser.get("anima-scripts", 'anima'))
anima_scripts = Path(config_parser.get("anima-scripts", 'anima-scripts-public-root'))
anima_extra_data = Path(config_parser.get("anima-scripts", 'extra-data-root'))
atlas_path = anima_extra_data / 'uspio-atlas' / 'scalar-space'
atlas_paths = { 'flair': atlas_path / 'FLAIR' / 'FLAIR_1.nrrd', 't1': atlas_path / 'T1' / 'T1_1.nrrd', 't2': atlas_path / 'T2' / 'T2_1.nrrd' }

load_dotenv()

if 'nnUNet_raw_data_base' not in os.environ or 'nnUNet_preprocessed' not in os.environ or 'RESULTS_FOLDER' not in os.environ:
	sys.exit('nnUNet environment variables are undefined, please create a .env file to initialize the nnUNet_raw_data_base, nnUNet_preprocessed and RESULTS_FOLDER variables (see .env.example file).')

def init_config(configuration_file):
	global configuration, prediction_folder, intermediate_folder

	config_file_path = Path(configuration_file)

	if not config_file_path.exists():
		sys.exit('The configuration file ' + str(config_file_path) + ' does not exists. Please follow the installation instructions in README.md.')

	configuration = None

	with open(config_file_path, "r") as f:
		try:
			configuration = yaml.safe_load(f)
		except yaml.YAMLError as exc:
			sys.exit(exc)

	try:
		prediction_folder = Path(configuration['prediction_folder'].replace('{nnunet_base}', os.environ['nnUNet_raw_data_base']))
		intermediate_folder = Path(configuration['intermediate_folder']) if 'intermediate_folder' in configuration else Path(tempfile.mkdtemp())
	except Exception as e:
		sys.exit('Error while reading the configuration file: ' + str(e))

	if 'patient_structure' not in configuration:
		sys.exit('patient_structure is missing in the configuration file.')

	return

nnunet_folder = Path(os.environ['nnUNet_raw_data_base'])
nnunet_preprocessed_folder = Path(os.environ['nnUNet_preprocessed'])
nnunet_model_folder = Path(os.environ['RESULTS_FOLDER'])

environment_variables = os.environ.copy()

valid_preprocess_steps = ['prepare', 'prepare_without_brain_extraction', 'reorient', 'remove_bias', 'normalize', 'adjust', 'check', 'install', 'plan', 'none']

def get_attribute(object, *attributes):
	for attribute in attributes:
		if attribute not in object: return None
		object = object[attribute]
	return object

def get_configuration(*attributes):
	return get_attribute(configuration, *attributes)

# Returns the stem of the path
# path.stem only works with single extension like .zip, whereas this function works with extensions like .nii.gz
def stem(path):
	return path.name[:-len(''.join(path.suffixes))]

def replace_string_suffix(string, old_suffix, new_suffix):
	return string[:-len(old_suffix)] + new_suffix

def replace_path_suffix(path, old_suffix, new_suffix):
	return path.parent / replace_string_suffix(path.name, old_suffix, new_suffix)

# Calls a command, if there are errors: outputs them and exit
def call(command, env=os.environ.copy(), stdout=None):
	command = [str(arg) for arg in command]
	status = subprocess.call(command, env=env, stdout=stdout)
	if status != 0:
		print(' '.join(command) + '\n')
		sys.exit('Command exited with status: ' + str(status))
	return status

def copy_file(input_file, output_file, symlink=False, skip_if_exists=False):
	if skip_if_exists and output_file.exists():
		return output_file
	output_file.parent.mkdir(exist_ok=True, parents=True)
	if symlink:
		os.symlink(input_file.resolve(), output_file)
	else:
		shutil.copyfile(input_file, output_file)
	return output_file

def copy_file_if_exists(input_file, output_file, symlink=False, skip_if_exists=False):
	if not input_file.exists():
		return None
	return copy_file(input_file, output_file, symlink, skip_if_exists)

def copy_file_relative(input_file, input_folder, output_folder, symlink=False):
	output_file = output_folder / input_file.relative_to(input_folder).parent / input_file.name
	copy_file(input_file, output_file, symlink)
	return output_file

def get_patient_structure_from_files(modalities_time01, modalities_time02, segmentation_time01, segmentation_time02, reference):
	patient_structure = {
		'times': [
			{
				'modalities': modalities_time01,
				'segmentation': segmentation_time01
			},
			{
				'modalities': modalities_time02,
				'segmentation': segmentation_time02,
			}
		],
		'reference': reference
	}
	return patient_structure

def create_file_path(time, type, modality, extension, file_pattern='{time}{type}{modality}{extension}'):
	return file_pattern.replace('{time}', time).replace('{type}', type).replace('{modality}', modality).replace('{extension}', extension)

def get_patient_structure(cross_sectional=False):
	return configuration['patient_structure']['cross_sectional' if cross_sectional else 'longitudinal']

def get_patient_structure_from_description(time_names=['time01', 'time02'], modality_folder_name='anatomy-brain', segmentation_folder_name='segmentations-brain', segmentation_name='groundTruth-new', modality_names=['flair', 't1', 't2'], extension='.nii.gz', file_pattern='{time}/{type}/{modality}{extension}', reference='time01/anatomy-brain/flair.nii.gz'):
	patient_structure = {'times': []}
	for i in range(2):
		patient_structure['times'].append({'modalities': {}, 'segmentation': create_file_path(time_names[i], segmentation_folder_name, segmentation_name, extension, file_pattern) })
		for modality_name in modality_names:
			patient_structure['times'][i]['modalities'][modality_name] = create_file_path(time_names[i], modality_folder_name, modality_name, extension, file_pattern)
	patient_structure['reference'] = reference
	return patient_structure

def get_file_format(file_type, default_format):
	is_in_config = 'file_formats' in configuration and file_type in configuration['file_formats']
	return configuration['file_formats'][file_type] if is_in_config else default_format

def create_patient_folder(modalities_time01, modalities_time02, reference, patient_structure, patient_name=None, patient_folder=None):
	if patient_folder is None:
		patient_folder = intermediate_folder / (patient_name or 'patient')
	patient_folder.mkdir(exist_ok=True, parents=True)
	copy_file(reference, patient_folder / patient_structure['reference'])
	for i, m in enumerate(modalities_time01):
		copy_file(m, patient_folder / get_modality(0, i, patient_structure))
	for i, m in enumerate(modalities_time02):
		copy_file(m, patient_folder / get_modality(1, i, patient_structure))
	return patient_folder

def get_modality(time, index, patient_structure):
	modalities = patient_structure['times'][time]['modalities']
	modality_name = [*modalities.keys()][index]
	return modalities[modality_name]

def get_modalities(cross_sectional):
	patient_structure = get_patient_structure(cross_sectional)
	return [*patient_structure['times'][0]['modalities'].keys()]

def get_task_id_from_name(task_name):
	match = re.search(r'Task(\d+)', task_name)
	return match.group(1) if match else None

def get_last_task_id(task_folder=nnunet_folder / 'nnUNet_raw_data', default_task=''):
	task_ids = [get_task_id_from_name(p.name) for p in task_folder.ls()]
	tasks_sorted_by_prefix = sorted([m for m in task_ids if m])
	last_prefix = tasks_sorted_by_prefix[-1] if len(tasks_sorted_by_prefix) > 0 else default_task
	return last_prefix.replace('Task', '')

def get_last_task_id(task_folder=nnunet_folder / 'nnUNet_raw_data', default_task=''):
	task_matches = [re.search(r'Task(\d+)', p.name) for p in task_folder.ls()]
	tasks_sorted_by_prefix = sorted([match.group(1) for match in task_matches if match])
	last_prefix = tasks_sorted_by_prefix[-1] if len(tasks_sorted_by_prefix) > 0 else default_task
	return last_prefix.replace('Task', '')

def get_last_task_name(display_task_name=False):
	nnunet_raw_data_folder = nnunet_folder / 'nnUNet_raw_data'
	task_path = list(nnunet_raw_data_folder.glob('Task' + str(get_last_task_id(nnunet_raw_data_folder, '500')) + '*'))[0]
	if display_task_name:
		print('The task ' + task_path.name + ' will be used (full path is ' + str(task_path) + ').')
	return task_path.name

def get_last_model_name(model_architecture, display_task_name=False):
	nnunet_model_task_folder = nnunet_model_folder / 'nnUNet' / model_architecture
	last_model_id = str(get_last_task_id(nnunet_model_task_folder, ''))
	if last_model_id == '':
		sys.exit('No model found in ' + str(nnunet_model_folder) + '. Please install one or set the RESULTS_FOLDER environment variable to a folder containing one or more models.')
	task_path = list(nnunet_model_task_folder.glob('Task' + last_model_id + '*'))[0]
	if display_task_name:
		print('The model ' + task_path.name + ' will be used (full path is ' + str(task_path) + ').')
	return task_path.name

def get_next_task_name(suffix='', display_new_task_name=False):
	task_number = int(get_last_task_id()) + 1
	task_name = f'Task{task_number:03}' + ('_' + suffix if suffix else '')
	if display_new_task_name:
		print('A new task ', task_name, ' will be created in "', environment_variables['nnUNet_raw_data_base'] + '/nnUNet_raw_data/' + task_name + '"')
	return task_name

def create_task_descriptor(modalities, names, path, task_name='', task_description='', cross_sectional=False):

	json_dict = OrderedDict()
	json_dict['name'] = task_name
	json_dict['description'] = task_description
	json_dict['tensorImageSize'] = "4D"
	json_dict['reference'] = ""
	json_dict['licence'] = ""
	json_dict['release'] = "0.0"
	json_dict['modality'] = {}

	if not cross_sectional:
		for i, modality in enumerate(modalities):
			json_dict['modality'][str(i*2)] = modality + '_time01'
			json_dict['modality'][str(i*2+1)] = modality + '_time02'
	else:
		for i, modality in enumerate(modalities):
			json_dict['modality'][str(i)] = modality

	json_dict['labels'] = {
		"0": "background",
		"1": "lesion"
	}
	json_dict['numTraining'] = len(names['training'])
	json_dict['numTest'] = len(names['testing'])
	json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in names['training']]
	json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in names['testing']]

	with open(os.path.join(path, "dataset.json"), 'w') as f:
		json.dump(json_dict, f, indent=4, sort_keys=True)

def get_or_create_identity_transform_serie():
	id_xml = Path(anima_extra_data / 'id.xml')
	if id_xml.exists():
		return id_xml
	try:
		call([anima / 'animaTransformSerieXmlGenerator', '-i', anima_extra_data / 'id.txt', '-o', id_xml])
	except Exception as inst:
		sys.exit('The following exception occured ' + str(inst) + ' while creating the identity serie file in ' + str(id_xml) + '.\n' +
		'You can generate the file yourself with the following command: animaTransformSerieXmlGenerator -i ' + str(anima_extra_data) + '/id.txt -o ' + str(anima_extra_data) + '/id.xml')

	return id_xml

def check_images_exist(patient, modalities, check_segmentation=False, cross_sectional=False, silent=False):

	patient_structure = get_patient_structure(cross_sectional)
	reference_path = patient / patient_structure['reference']

	if not reference_path.exists():
		if not silent:
			print(str(reference_path) + ' does not exist.')
		return False
	
	existing_modalities = []

	times = patient_structure['times'] if not cross_sectional else [patient_structure['times'][0]]

	for modality_type in (modalities or get_modalities(cross_sectional)):
		all_modalities_exist = all([(patient / time['modalities'][modality_type]).exists() for time in times])
		if all_modalities_exist:
			existing_modalities.append(modality_type)
		elif modalities is not None:
			if not silent:
				print([patient / time['modalities'][modality_type] for time in times])
				print(f'A modality is missing on patient {patient.name} does not exist.')
			return None

	if check_segmentation:
		for i, time in enumerate(times):
			segmentation_path = patient / time['segmentation']
			if not segmentation_path.exists() and i == len(times) - 1:
				if not silent:
					print(str(segmentation_path) + ' does not exist.')
				None

	return existing_modalities

def get_modalities_to_use(patients, modalities, check_segmentation=False, cross_sectional=False, silent=False):
	existing_modalities_patients = []
	for patient in patients.ls():
		if not patient.is_dir(): continue
		existing_modalities = check_images_exist(patient, modalities, check_segmentation, cross_sectional, silent)
		if not existing_modalities:
			return None
		existing_modalities_patients.append(existing_modalities)
	modalities_to_use = []
	modality_types = get_modalities(cross_sectional)
	for m in modality_types:
		modality_in_every_patient = True
		for existing_modalities_patient in existing_modalities_patients:
			if m not in existing_modalities_patient:
				modality_in_every_patient = False
				break
		if modality_in_every_patient:
			modalities_to_use.append(m)
	return modalities_to_use if len(modalities_to_use) > 0 else None

default_tolerance = 6.4999843e-07

def vectors_are_equal(vector1, vector2, tolerance=default_tolerance):
	return all( abs(v2-v1)<tolerance for v1, v2 in zip(vector1, vector2))

def check_images_match(image1_path, image2_path, image1=None, image2=None, message_prefix=None, use_logging=False, silent=False, tolerance=default_tolerance):
	
	image1 = image1 or sitk.ReadImage(str(image1_path))
	image2 = image2 or sitk.ReadImage(str(image2_path))
	
	if message_prefix is None:
		message_prefix = str(image1_path) + ' and ' + str(image2_path)
	
	if not use_logging and not message_prefix.startswith('error'):
		message_prefix = 'error: ' + message_prefix

	messages = []

	match = True
	if not vectors_are_equal(image1.GetSize(), image2.GetSize(), tolerance):
		messages.append(f'{message_prefix} sizes do not match: {image1.GetSize()} - {image2.GetSize()}')
		match = False
	if not vectors_are_equal(image1.GetOrigin(), image2.GetOrigin(), tolerance):
		messages.append(f'{message_prefix} origins do not match: {image1.GetOrigin()} - {image2.GetOrigin()}')
		match = False
	if not vectors_are_equal(image1.GetDirection(), image2.GetDirection(), tolerance):
		messages.append(f'{message_prefix} directions do not match: {image1.GetDirection()} - {image2.GetDirection()}')
		match = False
	if not vectors_are_equal(image1.GetSpacing(), image2.GetSpacing(), tolerance):
		messages.append(f'{message_prefix} spacings do not match: {image1.GetSpacing()} - {image2.GetSpacing()}')
		match = False
	
	if not silent:
		for message in messages:
			if use_logging:
				logging.error(message)
			else:
				print(message)

	return match, image1, image2

def check_images_are_equal(image1_path, image2_path, image1=None, image2=None, message_prefix=None, use_logging=False, silent=False, tolerance=default_tolerance):
	match, image1, image2 = check_images_match(image1_path, image2_path, image1, image2, message_prefix, use_logging, silent, tolerance)
	if not match:
		return False, image1, image2
	image1_data = sitk.GetArrayFromImage(image1)
	image2_data = sitk.GetArrayFromImage(image2)
	return np.all(image1_data == image2_data), image1, image2

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def init_logging(log_file):
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		datefmt='%Y-%m-%d %H:%M:%S',
		# filemode='a',
		handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler(sys.stdout)
		]
	)

def open_yml(path, default_yml={}, fail_silently=True):
	try:
		with open(path, 'r') as yaml_file:
			return yaml.safe_load(yaml_file)
	except Exception as e:
		# print(e)
		if not fail_silently:
			raise
	return default_yml


def write_yml(path, yml):
	path.parent.mkdir(exist_ok=True, parents=True)
	with open(path, 'w') as yaml_file:
		yaml.dump(yml, yaml_file, default_flow_style=False)

def flatten(list_of_list):
	return [x for l in list_of_list for x in l]
