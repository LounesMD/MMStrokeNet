from pathlib import Path
import sys
import core.utils as utils

def add_modalities_arg(parser):
	parser.add_argument('-m', '--modalities', type=str, help='Modalities to use for this task. Use all modalities by default.', nargs='*')

def add_model_architecture_arg(parser):
	parser.add_argument('-ma', '--model_architecture', type=str, help='Model architecture (2d or 3d_fullres).', default='3d_fullres')

def add_task_name_arg(parser, help='Name of the task.', required=False):
	parser.add_argument('-tn', '--task_name', type=str, help=help, required=required)

def add_overwrite_arg(parser):
	parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite preprocessing step folders if they exist (the existing preprocessing step folders will be used if overwrite is false).', default=False)

def add_checkpoint_arg(parser):
	parser.add_argument('-chk', '--checkpoint', type=str, help='Checkpoint name (nnUNet will use model_final_checkpoint as default).', default=None)

def add_cross_sectional_arg(parser):
	parser.add_argument('-cs', '--cross_sectional', action='store_true', help='Enable cross sectional mode to detect all lesions on the first time point.')

def add_log_file_arg(parser):
	parser.add_argument('-lf', '--log_file', type=str, help="Path to the log file", default='check.log')
	return

def add_configuration_arg(parser):
	parser.add_argument('-cfg', '--configuration', help='Path to configuration file.', type=str, default='config.yml')
	return

def add_preprocess_steps_arg(parser, additional_help, nargs, default):
	parser.add_argument('-prep', '--preprocess_steps', type=str, help="""The preprocessing to apply: 
	\n "prepare" will reorient images, extract the brain, register all modalities on the reference image, and crop all images (for each patient), voxel intensities are untouched during this step,
	\n "prepare_without_brain_extraction" does the same without the brain extraction step,
	\n "reorient" will just reorient images,
	\n "normalize" will apply the mean std normalization on the modalities,
	\n "remove_bias" will remove the bias on the modalities,
	\n "adjust" will adjust the voxel intensities, remove the bias (not on the pmaps) and apply the normalization steps for each modality,
	\n "check" will check that all modalities and segmentations match with the reference image,
	\n "install" will install the data to the nnunet format""" + additional_help, nargs=nargs, default=default)
	return

def add_preprocess_output_arg(parser):
	parser.add_argument('-po', '--preprocess_output', type=str, help='Path to the preprocess output where preprocess folders will be written. Useful to keep the input dataset structure untouched.', default=None)
	return

def add_postprocess_steps_arg(parser):
	parser.add_argument('-postp', '--postprocess_steps', type=str, help="""The postprocessing to apply:
	\n "remove_external_lesions" will remove the lesions which are outside the images intersection (the intersection mask from all timesteps), only used in longitudinal mode,
	\n "threshold" will threshold the pmaps with the min_pmap_threshold parameter,
	\n "remove_small" will remove lesions smaller than min_volume,
	\n "remove_low_keep_largest" remove voxels below max_pmap_threshold but keep n largest voxels of lesions (n being the minimum number of voxels in a lesion),
	\n "remove_low" will remove lesions with a max value lower than max_pmap_threshold""", nargs='*', default=['threshold', 'remove_small', 'remove_low_keep_largest', 'remove_external_lesions'])
	return

def add_postprocessing_args(parser):
	parser.add_argument('-pmin', '--min_pmap_threshold', type=float, help='Low pmap value threshold.', default=0.2)
	parser.add_argument('-pmax', '--max_pmap_threshold', type=float, help='High pmap value threshold.', default=0.2)
	parser.add_argument('-vmin', '--min_volume', type=float, help='Lesion volume threshold.', default=4)
	return

def get_modalities(args):
	all_modalities = utils.get_modalities(args.cross_sectional)
	if args.modalities:
		for modality in args.modalities:
			if modality not in all_modalities:
				sys.exit('The modality ' + modality + ' does not exist in the modalities of the patient structure: ' + str(all_modalities))
	return args.modalities

def check_preprocess_steps(preprocess_steps):
	for preprocess_step in preprocess_steps:
		if preprocess_step not in utils.valid_preprocess_steps:
			sys.exit(preprocess_step + ' is not a valid preprocessing (must be in ' + str(utils.valid_preprocess_steps) + ').')
	return preprocess_steps

def initialize_prediction(parser, batch=False):
	add_modalities_arg(parser)
	parser.add_argument('-mn', '--model_names', type=str, help='The names of the models to use (the task names used to train the models).', nargs='+')
	add_model_architecture_arg(parser)
	
	parser.add_argument('-f', '--folds', nargs='+', default=['None'], help="Folds to use for prediction. Default is None which means that folds will be detected automatically in the model output folder.")
	
	add_task_name_arg(parser, 'Name of the task to create. Default is TaskNNN where NNN is the last task ID + 1 (0-padded with 3 digits).')

	parser.add_argument('-o', '--output', type=str, help='Path to the output folder which will contain the segmentations. Default is nnUNet_raw_data_base / predictions / --task_name.')
	
	add_preprocess_steps_arg(parser, """\n "none" will not apply any preprocessing.
	\n If "prepare" is given, "adjust" and "install" will be added.
	\n If "adjust" is given, "install" will be added.""", nargs='*', default=['prepare', 'adjust', 'check', 'install'])

	add_preprocess_output_arg(parser)
	add_overwrite_arg(parser)
	add_checkpoint_arg(parser)
	add_postprocess_steps_arg(parser)
	add_postprocessing_args(parser)
	add_cross_sectional_arg(parser)
	add_log_file_arg(parser)
	add_configuration_arg(parser)

	args = parser.parse_args()

	utils.init_config(args.configuration)

	modalities = get_modalities(args)

	id_xml = utils.get_or_create_identity_transform_serie()

	preprocess_steps = args.preprocess_steps
	
	task_to_process_name = None

	# If process mulitple patients, and patients is null: use given nnunet task and ignore preprocess steps
	# or if 'none' in preprocess_steps: ignore preprocess steps
	if batch and args.patients is None or 'none' in preprocess_steps:
		preprocess_steps = []
		if args.patients:
			task_to_process_name = Path(args.patients).name

	if 'prepare' in preprocess_steps and 'adjust' not in preprocess_steps:
		preprocess_steps.append('adjust')

	if 'adjust' in preprocess_steps and 'install' not in preprocess_steps:
		preprocess_steps.append('install')

	check_preprocess_steps(preprocess_steps)

	utils.init_logging(args.log_file)
	
	model_task_names = args.model_names or (utils.configuration['ensembling_weights'].keys() if 'ensembling_weights' in utils.configuration else [utils.get_last_model_name(args.model_architecture, display_task_name=True)])

	model_task_names = ['Task' + model_task_name if not model_task_name.startswith('Task') else model_task_name for model_task_name in model_task_names ]

	if not task_to_process_name:
		task_to_process_name = args.task_name or utils.get_next_task_name(display_new_task_name='install' in preprocess_steps)

	if not task_to_process_name.startswith('Task'): task_to_process_name = 'Task' + task_to_process_name

	nnunet_task_folder = utils.nnunet_folder / 'nnUNet_raw_data' / task_to_process_name

	return args, nnunet_task_folder, model_task_names, modalities, preprocess_steps, id_xml
