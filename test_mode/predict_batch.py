from pathlib import Path
import shutil
import argparse
import sys
import yaml
import core.args_parser
import core.preprocess
import core.predict
import core.postprocess
import core.utils as utils
from core.ensemble import ensemble

Path.ls = lambda x: sorted(list(x.iterdir()))

parser = argparse.ArgumentParser(description="""Detect new MS lesions from two time points.""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-p', '--patients', type=str, help='Path to the folder of patients (datasets/raw).')

parser.add_argument('-kid', '--keep_intermediate_data', action='store_true', help='Keep the generated intermediate data.', default=None	)

args, nnunet_task_folder, model_task_names, modalities, preprocess_steps, id_xml = core.args_parser.initialize_prediction(parser, True)

patients = Path(args.patients) if args.patients else None

if patients is None and not nnunet_task_folder.exists():
	sys.exit(f'The --patients argument is undefined and the nnunet task folder "{nnunet_task_folder}" does not exist.')

dataset_folder = Path(args.preprocess_output) / patients.parent.name if args.preprocess_output else (patients.parent if patients else None)

if patients is not None and len(preprocess_steps) > 0:
	modalities = utils.get_modalities_to_use(patients, modalities, check_segmentation=False, cross_sectional=args.cross_sectional)
	if not modalities:
		sys.exit('One or more image is missing.')
	core.predict.check_models_accepts_modalities(args.model_architecture, model_task_names, modalities)

	core.preprocess.update_dataset_description_file(args, dataset_folder, patients, modalities, preprocess_steps, nnunet_task_folder)

	for patient_folder in patients.ls():
		
		if not patient_folder.is_dir(): continue

		print(patient_folder.name)

		folders = {
			'prepare': dataset_folder / 'prepared' / patient_folder.name,
			'reorient': dataset_folder / 'reoriented' / patient_folder.name,
			'remove_bias': dataset_folder / 'unbiased' / patient_folder.name,
			'normalize': dataset_folder / 'normalized' / patient_folder.name,
			'adjust': dataset_folder / 'adjusted' / patient_folder.name,
			'install': nnunet_task_folder
		}
		core.preprocess.process_patient(patient_folder, preprocess_steps, folders, args.overwrite, modalities, args.cross_sectional, False)
else:
	core.predict.check_models_accepts_modalities(args.model_architecture, model_task_names, modalities)
		
parent_output_folder = Path(args.output) if args.output else utils.prediction_folder / nnunet_task_folder.name

for model_task_name in model_task_names:
	output_folder = parent_output_folder / model_task_name
	output_folder.mkdir(exist_ok=True, parents=True)

	core.predict.predict(args, nnunet_task_folder, output_folder, model_task_name)

	core.postprocess.create_prediction_description_file(args, nnunet_task_folder, model_task_name, output_folder)
	core.postprocess.postprocess_patients(patients if 'install' in preprocess_steps else None, output_folder, args, id_xml, modalities)

if len(model_task_names)>1:
	output_folder = ensemble(patients, parent_output_folder, args.cross_sectional)
	core.postprocess.create_prediction_description_file(args, nnunet_task_folder, model_task_name, output_folder)
	core.postprocess.postprocess_patients(patients if 'install' in preprocess_steps else None, output_folder, args, id_xml, modalities)

keep_intermediate_data = args.keep_intermediate_data

if keep_intermediate_data is None and any([ps in preprocess_steps for ps in ['prepare', 'prepare_without_brain_extraction', 'reorient', 'remove_bias', 'normalize', 'adjust']]):
	keep_intermediate_data_string = input('Do you want to keep the intermediate data (folders containing the prepared and adjusted data)? [y/N] ')
	keep_intermediate_data = keep_intermediate_data_string.lower()[0] == 'y'

if not keep_intermediate_data:
	if ('prepare' in preprocess_steps or 'prepare_without_brain_extraction' in preprocess_steps) and (dataset_folder / 'prepared').exists():
		shutil.rmtree(dataset_folder / 'prepared')
	if 'reorient' in preprocess_steps and (dataset_folder / 'reoriented').exists():
		shutil.rmtree(dataset_folder / 'reoriented')
	if 'remove_bias' in preprocess_steps and (dataset_folder / 'unbiased').exists():
		shutil.rmtree(dataset_folder / 'unbiased')
	if 'normalize' in preprocess_steps and (dataset_folder / 'normalized').exists():
		shutil.rmtree(dataset_folder / 'normalized')
	if 'adjust' in preprocess_steps and (dataset_folder / 'adjusted').exists():
		shutil.rmtree(dataset_folder / 'adjusted')
	if 'install' in preprocess_steps and nnunet_task_folder.exists():
		shutil.rmtree(nnunet_task_folder)
