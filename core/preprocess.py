from asyncio import subprocess
from pathlib import Path
import os
import shutil
import SimpleITK as sitk
import numpy as np
import logging
import yaml
import core.check
import core.histogram_contrast_matching as hcm
import core.crop as crop
import core.utils as utils
Path.ls = lambda x: sorted(list(x.iterdir()))

doif = sitk.DICOMOrientImageFilter()

def convert_image(image, image_type=sitk.sitkFloat64):
	# Necessary since animaCropImage (and maybe other anima tools?) does not support short and long types
	# Or replace animaCropImage with sitk.LabelStatisticsImageFilter()?  
	return sitk.Cast(image, image_type)

	# sitk.RescaleIntensity(image, 0, 255) # or:
	# img_data = sitk.GetArrayFromImage(image)
	# if np.max(img_data) > 32767 or np.min(img_data) < -32767:
	# 	max_value = np.max(img_data)
	# 	if abs(np.min(img_data)) > max_value:
	# 		max_value = abs(np.min(img_data))
	# 	img_data = img_data/max_value * 32767
	# new_img = sitk.GetImageFromArray(img_data)
	# new_img.CopyInformation(image)
	# new_img = sitk.Cast(new_img, sitk.sitkInt16)
	# return new_img

def reorient_and_convert_image(input_path, orientation, image_type=sitk.sitkFloat64, output_path=None):
	output_path = output_path or input_path
	image = sitk.ReadImage(str(input_path))
	if orientation:
		doif.SetDesiredCoordinateOrientation(orientation)
		image = doif.Execute(image)
	image = convert_image(image, image_type)
	sitk.WriteImage(image, str(output_path))
	return output_path

def mask_path_from_modality_path(modality_path):
	return utils.replace_path_suffix(modality_path, '.nii.gz', '_mask.nii.gz')

def extract_brain(input_path, output_path=None):

	mask_path = mask_path_from_modality_path(input_path)
	if output_path is None: output_path = input_path

	keep_brain_extraction_args = ['--keep-intermediate-folder'] if 'keep_brain_extraction_folder' in utils.configuration and utils.configuration['keep_brain_extraction_folder'] else []
	utils.call(['python', utils.anima_scripts / 'brain_extraction' / 'animaAtlasBasedBrainExtraction.py', '-i', input_path, '--mask', mask_path, '--brain', output_path] + keep_brain_extraction_args, stdout=subprocess.DEVNULL)
	
	'''
	# FG: make it larger coz we have lesions very close to skull
	utils.call([utils.anima / 'animaMorphologicalOperations', '-i', mask_path, '-o', '/home/francy/Desktop/FG_AVC_2023/test.nii.gz'] ,"-a", "dil", "-r", "5"])
    call(command)
    '''
    
   
	return output_path, mask_path

def rigid_registration(moving_path, reference_path, output_path=None, transform_path=None):
	transform_args = ['-O', transform_path] if transform_path is not None else []
	utils.call([utils.anima / 'animaPyramidalBMRegistration', '-m', moving_path, '-r', reference_path, '-o', output_path or moving_path] + transform_args, stdout=subprocess.DEVNULL)
	return output_path or moving_path, transform_path

def prepare_modality(modality, orientation, skip_brain_extraction=False):

	reorient_and_convert_image(modality, orientation)
	if not skip_brain_extraction:
		extract_brain(modality)

	return modality

def prepare_segmentation(segmentation, union_mask, reference_cropped, orientation, skip_crop=False, force_same_space=True):

	reorient_and_convert_image(segmentation, orientation, sitk.sitkUInt8)
	
	if not skip_crop:
		crop.crop(segmentation, union_mask)
	
	if force_same_space:
		put_image_in_target_space(segmentation, reference_cropped)

	return segmentation

def put_image_in_target_space(input_path, target_path, output_path=None):
	output_path = output_path or input_path
	image = sitk.ReadImage(str(input_path))
	target = sitk.ReadImage(str(target_path))

	image.SetSpacing(target.GetSpacing())
	image.SetDirection(target.GetDirection())
	image.SetOrigin(target.GetOrigin())
	sitk.WriteImage(image, str(output_path))
	return output_path

def create_union_mask(output_folder, modality_to_image_and_transform):
	union_mask_path = output_folder / 'union_mask.nii.gz'
	union_mask = None
	for modality_name, image_and_transform in modality_to_image_and_transform.items():
		registered_modality_path = image_and_transform['modality']
		registered_modality_mask = abs(sitk.ReadImage(str(registered_modality_path)))>1e-6
		union_mask = registered_modality_mask if union_mask is None else sitk.Or(registered_modality_mask, union_mask)
	sitk.WriteImage(union_mask, str(union_mask_path))
	return union_mask_path

def apply_anima_transform(moving_path, fixed_path, transform_txt_path, output_path):
	if transform_txt_path is not None:
		transform_xml_path = utils.replace_path_suffix(transform_txt_path, '.txt', '.xml')
		utils.call([utils.anima / 'animaTransformSerieXmlGenerator', '-i', transform_txt_path, '-o', transform_xml_path])
	else:
		transform_xml_path = utils.get_or_create_identity_transform_serie()
	utils.call([utils.anima / 'animaApplyTransformSerie', '-i', moving_path, '-g', fixed_path, '-t', transform_xml_path, '-o', output_path])
	transform_xml_path.unlink()
	return output_path

def prepare_patient(input_folder, output_folder, overwrite=False, modalities=None, orientation='RAS', cross_sectional=False, skip_brain_extraction=False, skip_crop=False, skip_registration=False, force_same_space=True):
	
	if output_folder.exists() and not overwrite:
		return output_folder
	
	patient_structure = utils.get_patient_structure(cross_sectional)

	if not modalities:
		modalities = utils.get_modalities(cross_sectional)

	output_folder.mkdir(exist_ok=True, parents=True)

	reference = patient_structure['reference']
	modality_path = output_folder / reference
	utils.copy_file(input_folder / reference, modality_path)
	prepare_modality(modality_path, orientation, skip_brain_extraction)
	
	modality_to_image_and_transform = dict()
	modality_to_image_and_transform[reference] = dict(modality=modality_path, transform=None)

	output_reference_path = output_folder / 'reference.nii.gz'
	output_reference_cropped_path = output_folder / 'reference_cropped.nii.gz'
	utils.copy_file(output_folder / reference, output_reference_path)

	for time in  patient_structure['times']:
		for modality_type in modalities:
			modality = time['modalities'][modality_type]
			modality_path = input_folder / modality

			if modality != reference and modality_path.exists():

				modality_path = utils.copy_file(modality_path, output_folder / modality)
				transform_path = output_folder / utils.replace_string_suffix(modality, '.nii.gz', '_transform.txt') if not skip_registration else None
				modality_to_image_and_transform[modality] = dict(modality=modality_path, transform=transform_path)
				prepare_modality(modality_path, orientation, skip_brain_extraction)
				if not skip_registration:
					rigid_registration(modality_path, output_reference_path, transform_path=transform_path)
	
	union_mask_path = create_union_mask(output_folder, modality_to_image_and_transform) if not skip_brain_extraction else output_reference_path

	if not skip_crop:
		crop.crop(output_reference_path, union_mask_path, output_reference_cropped_path)
	
	for time in  patient_structure['times']:
		for modality_type in modalities:
			modality = time['modalities'][modality_type]
			modality_path = input_folder / modality

			if modality_path.exists():
				
				if not skip_brain_extraction:
					transform_path = modality_to_image_and_transform[modality]['transform']
					modality_path = apply_anima_transform(modality_path, output_reference_path, transform_path, output_folder / modality)
					utils.call([utils.anima / 'animaMaskImage', '-i', modality_path, '-o', modality_path, '-m', union_mask_path])
				else:
					modality_path = output_folder / modality
				
				if not skip_crop:
					crop.crop(modality_path, union_mask_path)
				if force_same_space:
					put_image_in_target_space(modality_path, output_reference_cropped_path)
		
		segmentation = time['segmentation']
		segmentation_path = input_folder / segmentation
		if segmentation_path.exists():
			segmentation_path = utils.copy_file(segmentation_path, output_folder / segmentation)

			prepare_segmentation(segmentation_path, union_mask_path, output_reference_cropped_path, orientation, skip_crop, force_same_space)

	output_reference_path.unlink()
	output_reference_cropped_path.unlink()

	return output_folder

def remove_bias(input_path, output_path=None):
	utils.call([utils.anima / 'animaN4BiasCorrection', '-i', input_path, '-o', output_path or input_path])

def nyul_normalization(input_path, atlas_path, output_path=None):
	output_path = output_path or input_path
	utils.call([utils.anima / 'animaNyulStandardization', '-m', input_path,  '-r', atlas_path, '-o', output_path])
	return output_path

def compute_image_mask(image):
	data = sitk.GetArrayFromImage(image)

	zero = sitk.GetImageFromArray((data == 0).astype(np.uint8))
	
	ccifilter = sitk.ConnectedComponentImageFilter()
	ccifilter.FullyConnectedOn()
	labeled = ccifilter.Execute(zero)
	labeled_data = sitk.GetArrayFromImage(labeled)
	biggest_zero_region = np.argmax(np.bincount(labeled_data.flatten())[1:]) + 1
	mask = labeled_data != biggest_zero_region

	return data, mask

def mean_std_normalization(input_path, output_path=None):
	output_path = output_path or input_path
	
	# image = sitk.Normalize(image) 					# SimpleITK does not take into account the mask

	image = sitk.ReadImage(str(input_path))
	data, mask = compute_image_mask(image)

	data[mask] = ( data[mask] - data[mask].mean() ) / (data[mask].std() + 1e-8) 
	data[mask == 0] = 0

	noramlized_image = sitk.GetImageFromArray(data)
	noramlized_image.CopyInformation(image)

	sitk.WriteImage(noramlized_image, str(output_path))
	return output_path

def normalize_times(input_folder, output_folder, times, modality_type, cross_sectional):
	
	atlas_path = utils.atlas_paths[modality_type]
	
	for normalization_step in utils.configuration['normalization_schemes'][modality_type]:

		for i, time in enumerate(times):
			modality_path = output_folder / time['modalities'][modality_type]

			if normalization_step == 'nyul':
				nyul_normalization(modality_path, atlas_path)
			if normalization_step == 'meanstd':
				mean_std_normalization(modality_path)
			if normalization_step == 'hist' and not cross_sectional and i == 1:
				hcm.rectify_histogram(modality_path, output_folder / times[0]['modalities'][modality_type], modality_path)

	return

def custom_adjust_patient(input_folder, output_folder, overwrite=False, modalities=None, cross_sectional=False, adjust_modality=lambda **kwargs: remove_bias(kwargs['modality_path']), normalize=None):

	if output_folder.exists() and not overwrite:
		return output_folder
	
	patient_structure = utils.get_patient_structure(cross_sectional)

	if not modalities:
		modalities = utils.get_modalities(cross_sectional)
	
	output_folder.mkdir(exist_ok=True, parents=True)

	times = patient_structure['times'] if not cross_sectional else [patient_structure['times'][0]]

	for modality_type in modalities:

		for i, time in enumerate(times):
			modality = time['modalities'][modality_type]

			modality_path = utils.copy_file(input_folder / modality, output_folder / modality)
			
			adjust_modality(modality_path=modality_path, modality_type=modality_type)
		
		if normalize:
			normalize(input_folder=input_folder, output_folder=output_folder, times=times, modality_type=modality_type, cross_sectional=cross_sectional)

	for time in patient_structure['times']:
		segmentation = time['segmentation']
		utils.copy_file_if_exists(input_folder / segmentation, output_folder / segmentation)

	return output_folder

def remove_bias_patient(input_folder, output_folder, overwrite=False, modalities=None, cross_sectional=False):
	adjust_modality = lambda modality_path, modality_type, **kwargs: remove_bias(modality_path) if 'pmap' not in modality_type else None
	return custom_adjust_patient(input_folder, output_folder, overwrite=overwrite, modalities=modalities, cross_sectional=cross_sectional, adjust_modality=adjust_modality)

def normalize_patient(input_folder, output_folder, overwrite=False, modalities=None, cross_sectional=False):
	adjust_modality = lambda modality_path, modality_type, **kwargs: mean_std_normalization(modality_path) if 'pmap' not in modality_type else None
	return custom_adjust_patient(input_folder, output_folder, overwrite=overwrite, modalities=modalities, cross_sectional=cross_sectional, adjust_modality=adjust_modality)

def adjust_patient(input_folder, output_folder, overwrite=False, modalities=None, cross_sectional=False):
	adjust_modality = lambda modality_path, modality_type, **kwargs: remove_bias(modality_path) if 'pmap' not in modality_type else None
	return custom_adjust_patient(input_folder, output_folder, overwrite=overwrite, modalities=modalities, cross_sectional=cross_sectional, adjust_modality=adjust_modality, normalize=normalize_times)

def install_nnunet_image(image, nnunet_path, install, overwrite=False):

	if nnunet_path.exists() and not overwrite:
		return
	
	nnunet_path.parent.mkdir(exist_ok=True, parents=True)

	if install == 'copy':
		if image.name.endswith('.nii.gz'):
			shutil.copyfile(image, nnunet_path)
		else:
			m = sitk.ReadImage(str(image))
			sitk.WriteImage(m, str(nnunet_path))
	elif install == 'move':
		try:
			shutil.move(image, nnunet_path)
		except:
			shutil.copyfile(image, nnunet_path)
	else:
		os.symlink(image.resolve(), nnunet_path.resolve())
	
	return

def format_patient_to_nnunet(input_folder, nnunet_task_folder, overwrite=False, modalities=None, training=False, install='symlink', patient_name=None, cross_sectional=False):

	patient_name = patient_name or input_folder.name

	patient_structure = utils.get_patient_structure(cross_sectional)

	if not modalities:
		modalities = utils.get_modalities(cross_sectional)
	
	nnunet_task_folder.mkdir(exist_ok=True, parents=True)
	
	type_suffix = 'r' if training else 's'

	n_modalities = 0

	times = patient_structure['times'] if not cross_sectional else [patient_structure['times'][0]]

	for it, time in enumerate(times):
		im = 0
		for modality_type in modalities:
			modality = time['modalities'][modality_type]
			modality_path = input_folder / modality
			if modality_path.exists():			
				nnunet_name = f'{patient_name}_{it * n_modalities + im:04}.nii.gz' if not cross_sectional else f'{patient_name}_{im:04}.nii.gz'
				nnunet_path = nnunet_task_folder / ('imagesT' + type_suffix) / nnunet_name
				install_nnunet_image(modality_path, nnunet_path, install, overwrite)
				im += 1
		n_modalities = im
		
		segmentation_path = input_folder / time['segmentation']
		if segmentation_path.exists():
			nnunet_path = nnunet_task_folder / ('labelsT' + type_suffix) / (patient_name + '.nii.gz')
			install_nnunet_image(segmentation_path, nnunet_path, install, overwrite)

	return patient_name

def process_patient(patient_folder, preprocess_steps, folders, overwrite, modalities, cross_sectional, training, install='symlink', force_same_space=True, patient_name=None, print_prefix='   '):
	
	if 'prepare' in preprocess_steps:
		print(print_prefix + 'prepare...')
		patient_folder = prepare_patient(patient_folder, folders['prepare'], overwrite, modalities, cross_sectional=cross_sectional, force_same_space=force_same_space)
	
	if 'prepare_without_brain_extraction' in preprocess_steps:
		print(print_prefix + 'prepare without brain extraction...')
		patient_folder = prepare_patient(patient_folder, folders['prepare'], overwrite, modalities, cross_sectional=cross_sectional, skip_brain_extraction=True, force_same_space=force_same_space)
	
	if 'reorient' in preprocess_steps:
		print(print_prefix + 'reorient...')
		patient_folder = prepare_patient(patient_folder, folders['reorient'], overwrite, modalities, cross_sectional=cross_sectional, skip_brain_extraction=True, skip_crop=True, skip_registration=True, force_same_space=force_same_space)

	if 'remove_bias' in preprocess_steps:
		print(print_prefix + 'remove bias...')
		patient_folder = remove_bias_patient(patient_folder, folders['remove_bias'], overwrite, modalities, cross_sectional=cross_sectional)

	if 'normalize' in preprocess_steps:
		print(print_prefix + 'normalize...')
		patient_folder = normalize_patient(patient_folder, folders['normalize'], overwrite, modalities, cross_sectional=cross_sectional)

	if 'adjust' in preprocess_steps:
		print(print_prefix + 'adjust...')
		patient_folder = adjust_patient(patient_folder, folders['adjust'], overwrite, modalities, cross_sectional=cross_sectional)

	if 'check' in preprocess_steps:
		print(print_prefix + 'check...')
		core.check.check_patient(patient_folder, modalities, cross_sectional, True)
	
	if 'install' in preprocess_steps:
		print(print_prefix + 'install...')
		format_patient_to_nnunet(patient_folder, folders['install'], overwrite, modalities, training=training, install=install, patient_name=patient_name, cross_sectional=cross_sectional)
	
	return

def update_dataset_description_file(args, dataset_folder, patients, modalities, preprocess_steps, nnunet_task_folder, training=False):

	dataset_description_file = dataset_folder / 'description.yml'
	
	dataset_description = utils.open_yml(dataset_description_file, {
			'dataset_name': dataset_folder.name,
			'task_description': args.task_description if hasattr(args, 'task_description') else None,
			'preprocessings': []
		})
	
	new_preprocessing = {
		'input': str(patients),
		'preprocess_steps': preprocess_steps,
		'modalities': modalities,
		'cross_sectionnal': args.cross_sectional,
		'reference': utils.get_patient_structure(args.cross_sectional)['reference'],
		'normalizations': utils.configuration['normalization_schemes'],
		'configuration': args.configuration
	}

	preprocessings = []

	for preprocessing in dataset_description['preprocessings']:
		if 'input' in preprocessing and preprocessing['input'] == str(patients) and all([ pp in preprocess_steps for pp in preprocessing['preprocess_steps'] ]):
			logging.warning(f'The same preprocessing steps {preprocess_steps} were applied on this input dataset {patients}. The corresponding preprocessing description will be overwritten.')
			logging.warning(f'The current preprocessing description in {dataset_description_file} is:')
			logging.warning(yaml.dump(preprocessing))
			logging.warning(f'It will be replaced by:')
			logging.warning(yaml.dump(new_preprocessing))
		else:
			preprocessings.append(preprocessing)
	preprocessings.append(new_preprocessing)
	dataset_description['preprocessings'] = preprocessings
	
	utils.write_yml(dataset_description_file, dataset_description)

	nnunet_task_folder.mkdir(parents=True, exist_ok=True)
	utils.copy_file(dataset_description_file, nnunet_task_folder / f'description_{"training" if training else "testing"}.yml')

def copy_dataset_description_files(task_name):
	nnunet_task_folder = utils.nnunet_folder / 'nnUNet_raw_data' / task_name
	nnunet_task_folder_cropped = utils.nnunet_folder / 'nnUNet_cropped_data' / task_name
	nnunet_task_folder_preprocessed = utils.nnunet_preprocessed_folder / task_name
	utils.copy_file_if_exists(nnunet_task_folder / 'description_training.yml', nnunet_task_folder_cropped / 'description_training.yml')
	utils.copy_file_if_exists(nnunet_task_folder / 'description_testing.yml', nnunet_task_folder_cropped / 'description_testing.yml')
	utils.copy_file_if_exists(nnunet_task_folder / 'description_training.yml', nnunet_task_folder_preprocessed / 'description_training.yml')
	utils.copy_file_if_exists(nnunet_task_folder / 'description_testing.yml', nnunet_task_folder_preprocessed / 'description_testing.yml')
	return

def open_dataset_description_file(task_name):
	nnunet_task_folder = utils.nnunet_folder / 'nnUNet_raw_data' / task_name
	nnunet_task_folder_cropped = utils.nnunet_folder / 'nnUNet_cropped_data' / task_name
	nnunet_task_folder_preprocessed = utils.nnunet_preprocessed_folder / task_name
	dataset_description = {}
	for task_folder in [nnunet_task_folder_cropped, nnunet_task_folder_preprocessed, nnunet_task_folder]:
		dataset_description_file = task_folder / 'description_training.yml'
		if not dataset_description_file.exists(): continue
		dataset_description = utils.open_yml(dataset_description_file)
	return dataset_description
