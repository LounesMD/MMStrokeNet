from pathlib import Path
import sys
import logging
import argparse
import core.args_parser
import core.preprocess
import core.utils as utils
from batchgenerators.utilities.file_and_folder_operations import save_pickle
Path.ls = lambda x: sorted(list(x.iterdir()))

parser = argparse.ArgumentParser(description="""Preprocess one or more patients.""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-p', '--patients', '--training', type=str, help='Path to the patients folder. The patients will be considered as training data for the install step.')
parser.add_argument('-ts', '--testing', type=str, help='Path to the testing patients folder.')
parser.add_argument('-v', '--validation', type=str, help='Path to the validation patients folder.')
core.args_parser.add_task_name_arg(parser, 'Name of the task to create. Default is TaskNNN where NNN is the last task ID + 1 (0-padded with 3 digits).')
parser.add_argument('-td', '--task_description', type=str, help='Description of the task (will be stored in the task description file dataset.json).')
core.args_parser.add_modalities_arg(parser)
core.args_parser.add_preprocess_steps_arg(parser, """\n "plan" will execute the nnUNet_plan_and_preprocess step.""", nargs='+', default=['prepare', 'adjust', 'check', 'install', 'plan'])
core.args_parser.add_preprocess_output_arg(parser)
core.args_parser.add_overwrite_arg(parser)
core.args_parser.add_cross_sectional_arg(parser)
core.args_parser.add_log_file_arg(parser)
core.args_parser.add_configuration_arg(parser)

args = parser.parse_args()
utils.init_config(args.configuration)

modalities = core.args_parser.get_modalities(args)

# Make sure that the preprocess_steps argument is properly formatted and exit if a step is missing
preprocess_steps = core.args_parser.check_preprocess_steps(args.preprocess_steps)

utils.init_logging(args.log_file)

task_name = args.task_name or utils.get_next_task_name(display_new_task_name='install' in preprocess_steps or 'plan' in preprocess_steps)

if not task_name.startswith('Task'): task_name = 'Task' + task_name

patients_types = []

if args.testing:
	patients_types.append( (False, Path(args.testing)) )

if args.patients:
	patients_types.append( (True, Path(args.patients)) )

if args.validation:
	patients_types.append( (True, Path(args.validation)) )

nnunet_task_folder = utils.nnunet_folder / 'nnUNet_raw_data' / task_name

names = { 'training': [], 'testing': [] }

# get the modalities to use from the given argument or the existing modalities (check the file structure) 
for training, patients in patients_types:

	if len(preprocess_steps) > 1 or len(preprocess_steps) > 0 and preprocess_steps[0] != 'plan':
		modalities = utils.get_modalities_to_use(patients, modalities, check_segmentation=training, cross_sectional=args.cross_sectional)
		if not modalities:
			sys.exit('One or more image is missing.')

# preprocess all patients
for training, patients in patients_types:

	dataset_folder = patients.parent

	core.preprocess.update_dataset_description_file(args, dataset_folder, patients, modalities, preprocess_steps, nnunet_task_folder, training)

	# if preprocess steps must be saved in preprocess_output: dataset_folder will be the preprocess_output / the dataset name (e.g. preprocess_output/ofsepTesting0/)
	if args.preprocess_output:
		dataset_folder = Path(args.preprocess_output) / dataset_folder.name
	
	for patient_folder in patients.ls():
		if not patient_folder.is_dir(): continue

		print(patient_folder.name)

		names['training' if training else 'testing'].append(patient_folder.name)

		folders = {
			'prepare': dataset_folder / 'prepared' / patient_folder.name,
			'reorient': dataset_folder / 'reoriented' / patient_folder.name,
			'remove_bias': dataset_folder / 'unbiased' / patient_folder.name,
			'normalize': dataset_folder / 'normalized' / patient_folder.name,
			'adjust': dataset_folder / 'adjusted' / patient_folder.name,
			'install': nnunet_task_folder
		}
		
		try:
			core.preprocess.process_patient(patient_folder, preprocess_steps, folders, args.overwrite, modalities, args.cross_sectional, training)
		except RuntimeError as e:
			if len(e.args) > 0 and 'Filter does not support casting from casting vector of' in e.args[0]:
				logging.error(f'An image of patient {patient_folder.name} cannot be converted since it has multiple channels: {e}.')
			else:
				print(e)

if 'install' in preprocess_steps:
	for folder_name in ['imagesTs', 'imagesTr', 'labelsTs', 'labelsTr']:
		(nnunet_task_folder / folder_name).mkdir(exist_ok=True, parents=True)
	utils.create_task_descriptor(modalities, names, nnunet_task_folder, task_name=task_name, task_description=args.task_description, cross_sectional=args.cross_sectional)
	
if 'plan' in preprocess_steps:
	print('plan and preprocess task ' + task_name + '...')
	task_id = utils.get_task_id_from_name(task_name)
	command = ['nnUNet_plan_and_preprocess', '-t', task_id, '--verify_dataset_integrity']
	if 'custom_nnunet_parameters' in utils.configuration and 'experiment_planner' in utils.configuration['custom_nnunet_parameters']:
		experiment_planner = utils.configuration['custom_nnunet_parameters']['experiment_planner']
		if isinstance(experiment_planner, str):
			command += [ '-pl2d', 'None' ]
			command += [ '-pl3d', experiment_planner ]
		elif '2d' in experiment_planner or '3d' in experiment_planner:
			command += [ '-pl2d', experiment_planner['2d'] if '2d' in experiment_planner else 'None' ]
			if '3d' in experiment_planner: command += [ '-pl3d', experiment_planner['3d'] ]

	utils.call(command, env=utils.environment_variables)

	if args.validation:
		split = [{'train': [p.name for p in Path(args.patients).iterdir()], 'val': [p.name for p in Path(args.validation).iterdir()]}]
		save_pickle(split, utils.nnunet_preprocessed_folder / task_name / 'splits_final.pkl')

	# Copy description files to nnUNet_cropped_data and preprocessed
	core.preprocess.copy_dataset_description_files(task_name)
