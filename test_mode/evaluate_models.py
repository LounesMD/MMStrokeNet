from pathlib import Path
import os
import sys
import logging
import argparse
import json
import yaml
import base64
import pandas
import requests
import subprocess
from xml.dom import minidom
import core.utils as utils
import core.preprocess
import core.evaluation
import core.args_parser
import plotly.express as px
pandas.options.plotting.backend = "plotly"

Path.ls = lambda x: sorted(list(x.iterdir()))

parser = argparse.ArgumentParser(description="""Evaluate different models.""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-mo', '--model_outputs', '--model_output', type=str, help='Path to the folder of model outputs (folder containing one folder of segmentations per model). If the folder does not contain any folder but contains some nifti files, it will be considered as a single model output and evaluated as such. Lesions in the segmentations must be 1s (more precisely any value between 0.5 and 1.5), other values will be ignored.', required=True)
parser.add_argument('-gt', '--ground_truth', type=str, help='Path to the ground truth dataset. In the ground truth, new lesions must be 1, growing lesion must be 2 and will be ignored, and unclassifiable lesions must be 3 and will be counted as true positives when detected and true negatives otherwise.', required=True)
parser.add_argument('-o', '--output', type=str, help='Path to the output folder where the evaluation statistics will be saved.', required=True)
parser.add_argument('-en', '--evaluation_name', type=str, help='Name of the evaluation. Default is the name of the output folder.', default=False)
parser.add_argument('-lf', '--log_file', type=str, help="Path to the log file (default is output/evaluation.log)", default=None)
parser.add_argument('-f', '--formats', type=str, help='List of formats to output. Possible formats are "csv" or "tsv".', nargs='+', default=['csv'])

parser.add_argument('-psigts', '--put_segmentations_in_ground_truth_space', action='store_true', help='Put the segmentation in the space of the ground truth images. Useful when the segmentations were applied on the preprocessed (cropped) dataset, not on the raw one (this will put the segmentations back to the space of origin). The new segmentations will be suffixed with _on_ground_truth_space. If such suffixed segmentation exists, it is used and will not be recomputed.', default=False)
parser.add_argument('-lr', '--localize_lesions', action='store_true', help='Localize lesions.', default=False)
parser.add_argument('-ra', '--remove_atlases', action='store_true', help='Remove registered atlases after evaluation.', default=False)
parser.add_argument('-fr', '--force_recompute', action='store_true', help='Recompute the evaluation (do not read statistics files if they exists; overwrite them instead).', default=False)

parser.add_argument('-spa', '--seg_perf_analyzer', action='store_true', help='Compute true positives, false negatives and false positives as in animaSegPerfAnalyzer.')
parser.add_argument('-v', '--MinLesionVolume', type=float, help='(Used by animaSegPerfAnalyzer) Min volume of lesion for "Lesions detection metrics" in mm^3 (default 0 mm^3).', default=0.0)
parser.add_argument('-x', '--MinOverlapRatio', type=float, help='(Used by animaSegPerfAnalyzer) Minimum overlap ratio to say if a lesion of the GT is detected (detection threshold alpha).', default=0.1)
parser.add_argument('-y', '--MaxFalsePositiveRatio', type=float, help='(Used by animaSegPerfAnalyzer) Maximum of false positive ratio to limit the detection of a lesion in GT if a lesion in the image is too big (detection threshold beta).', default=0.7)
parser.add_argument('-z', '--MaxFalsePositiveRatioModerator', type=float, help='(Used by animaSegPerfAnalyzer) Percentage of the regions overlapping the tested lesion not too much outside of this lesion (detection threshold gamma).', default=0.65)

parser.add_argument('-rd', '--round', type=int, help='Round results to the given number of decimals.', default=3)
parser.add_argument('-sgg', '--skip_generate_graphs', action='store_true', help='By default, graphs are generated from the evaluation. Use this option to skip the graph generation.', default=False)
parser.add_argument('-sgp', '--skip_generate_previews', action='store_true', help='By default, lesion previews are generated from the evaluation. Use this option to skip the previews generation.', default=False)

core.args_parser.add_cross_sectional_arg(parser)
core.args_parser.add_configuration_arg(parser)
args = parser.parse_args()
utils.init_config(args.configuration)

model_outputs = Path(args.model_outputs).ls()

# if none of model_outputs are folder: then assume that the given folder is the output of only one model which we must evaluate
if not any([mo.is_dir() for mo in model_outputs]) and any([mo.name.endswith('.nii.gz') for mo in model_outputs]):
	model_outputs = [Path(args.model_outputs)]

ground_truth_dataset = Path(args.ground_truth)
output_path = Path(args.output)
output_path.mkdir(exist_ok=True, parents=True)
evaluation_name = args.evaluation_name or output_path.name

logging.basicConfig(
	level=logging.INFO,
	format="[%(levelname)s] %(message)s",
	handlers=[
		logging.FileHandler(args.log_file or str(output_path / 'evaluation.log')),
		logging.StreamHandler(sys.stdout)
	]
)

formats = []

# Validate given formats, use 'csv' by default
for format in args.formats:
	format = format[:-1] if format[-1] == '.' else format
	if format not in ['csv', 'tsv']:
		print('Format', format, 'is invalid.')
		continue
	formats.append(format)

if len(formats) == 0:
	print('No valid format was found. Using default format csv.')
	formats.append(['csv'])

format = 'csv' if 'csv' in formats else formats[0]
sep = '\t' if format == 'tsv' else ','

patient_structure = utils.get_patient_structure(args.cross_sectional)

global_stats_records = []
regions = ['deep white matter', 'brainstem', 'cerebellum', 'cortex', 'ventricles']
global_locations = pandas.DataFrame(data={'Region': regions, 'TP': [0] * 5, 'FN': [0] * 5, 'FP': [0] * 5})

id_xml = utils.get_or_create_identity_transform_serie()

lesions_stats_records = []

lesions_stats_path = output_path / f'lesions.{format}'

if not args.force_recompute and lesions_stats_path.exists():
	lesions_stats = pandas.read_csv(lesions_stats_path, sep=sep, dtype={'Patient': str, 'Index': int})
	lesions_stats_records = lesions_stats.to_dict('records')

all_models_stats = pandas.DataFrame()

task_fields_names = ['type', 'volume', 'p_max', 'p_mean', 'p_median', 'locations']
task_fields = [{ 'field': tfn, 'sortable': True, 'resizable': True, 'filter': True, 'editable': False } for tfn in task_fields_names]
task = { 'lesions': [], 'fields': task_fields }
task_path = output_path / 'lesions.json'
if task_path.exists():
	with open(str(task_path), 'r') as f:
		task = json.load(f)

lesion_previews_path = output_path / 'lesion_previews'
lesion_previews_path.mkdir(exist_ok=True)

for model_output in model_outputs:
	
	if not model_output.is_dir(): continue

	model_name = model_output.name

	print(model_name)

	suffix = f'mlv{args.MinLesionVolume}_mor{args.MinOverlapRatio}_mfpr{args.MaxFalsePositiveRatio}_mfprm{args.MaxFalsePositiveRatioModerator}'
	model_stats_path = output_path / f'{model_name}_{suffix}.{format}'
	model_locations_path = output_path / f'{model_name}_{suffix}_locations.{format}'
	
	model_locations = pandas.DataFrame(data={'Region': regions, 'TP': [0] * 5, 'FN': [0] * 5, 'FP': [0] * 5})

	prediction_description = utils.open_yml(model_output / 'description.yml')
	try:
		modalities = prediction_description['model']['preprocessings'][-1]['modalities']
	except Exception as e:
		print('Error while reading model description:', e)
		modalities = patient_structure['times'][0]['modalities'].keys()
	
	if not args.force_recompute and model_stats_path.exists() and (model_locations_path.exists() or not args.localize_lesions):
		model_stats = pandas.read_csv(model_stats_path, sep=sep, dtype={'Patient': str})
		if model_locations_path.exists():
			model_locations = pandas.read_csv(model_locations_path, sep=sep)
	else:
		model_stats_records = []

		for patient in ground_truth_dataset.ls():
			
			if not patient.is_dir(): continue

			print('   ', patient.name)

			ground_truth = patient / patient_structure['times'][-1]['segmentation']
			reference = patient / patient_structure['reference']
			segmentation = model_output / utils.get_file_format('final_segmentation', '{patient_name}_final_segmentation.nii.gz').format(patient_name=patient.name)
			pmap = model_output / f'{patient.name}pmap-1.nii.gz'
			patient_atlas = model_output / f'{patient.name}_atlas.nii.gz'

			performance = output_path / 'performance'

			if not segmentation.exists():
				segmentation = model_output / f'{patient.name}_final_segmentation_on_ground_truth_space.nii.gz'
				if not segmentation.exists():
					logging.error(f'Segmentation {segmentation} does not exist.')
					continue

			if args.put_segmentations_in_ground_truth_space:
				maps = [
					dict(on_current_space=segmentation, on_ground_truth_space=model_output / f'{patient.name}_final_segmentation_on_ground_truth_space.nii.gz'),
					dict(on_current_space=pmap, on_ground_truth_space=model_output / f'{patient.name}_pmap_on_ground_truth_space.nii.gz')
				]
				
				for m in maps:
					if m['on_current_space'].exists() and not m['on_ground_truth_space'].exists():
						match, _, _ = utils.check_images_match(ground_truth, m['on_current_space'], silent=True)
						if not match:
							utils.call([utils.anima / 'animaApplyTransformSerie', '-i', m['on_current_space'], '-g', ground_truth, '-o', m['on_ground_truth_space'], '-t', id_xml, '-n', 'nearest'], stdout=subprocess.DEVNULL)
						else:
							print('The segmentation', m['on_current_space'],'is already in the ground truth space.')
							m['on_ground_truth_space'] = m['on_current_space']

				segmentation = maps[0]['on_ground_truth_space']
				pmap = maps[1]['on_ground_truth_space']

			image_paths = {}
			image_descriptions = []

			utils.copy_file_if_exists(ground_truth, output_path / 'predictions' / f'{model_name}_{patient.name}_ground_truth.nii.gz', symlink=True, skip_if_exists=True)
			for time_index, time in enumerate(patient_structure['times']):
				for modality_index, modality_name in enumerate(modalities):
					modality_path = patient / time['modalities'][modality_name]
					if modality_path.exists():
						image_name = f'{modality_name}_time{time_index+1:02}'
						image_paths[image_name] = modality_path
						link = output_path / 'predictions' / f'{model_name}_{patient.name}_{image_name}.nii.gz'
						utils.copy_file(modality_path, link, symlink=True, skip_if_exists=True)
						image_descriptions.append({
							'name': image_name,
							'file': link.name,
							'parameters': {'minPercent': 0, 'maxPercent': 1, 'lut': 'Grayscale'},
							'display': modality_index==0
						})
			
			utils.copy_file(segmentation, output_path / 'predictions' / f'{model_name}_{segmentation.name}', symlink=True, skip_if_exists=True)
			image_descriptions.append({ 'name': 'segmentation', 'file': f'{model_name}_{segmentation.name}', 'parameters': {'min': 0, 'max': 2, 'lut': 'Blue Overlay'}, 'display': True })
			if pmap.exists():
				utils.copy_file(pmap, output_path / 'predictions' / f'{model_name}_{pmap.name}', symlink=True, skip_if_exists=True)
				image_descriptions.append({ 'name': 'pmap', 'file': f'{model_name}_{pmap.name}', 'parameters': {'minPercent': 0, 'maxPercent': 1, 'lut': 'Fire'}, 'display': False })
			utils.copy_file(ground_truth, output_path / 'predictions' / f'{model_name}_{patient.name}_{ground_truth.name}', symlink=True, skip_if_exists=True)
			image_descriptions.append({ 'name': 'ground_truth', 'file': f'{model_name}_{patient.name}_{ground_truth.name}', 'parameters': {'min': 0, 'max': 2, 'lut': 'Green Overlay'}, 'display': True })

			match, _, _ = utils.check_images_match(ground_truth, segmentation)
			if not match:
				logging.error(f'Segmentation {segmentation} is not in same space as ground truth {ground_truth}.')
				continue
			
			utils.call([utils.anima / 'animaSegPerfAnalyzer', "-i", segmentation, "-r", ground_truth, "--MaxFalsePositiveRatioModerator", min(1, args.MaxFalsePositiveRatioModerator), "--MaxFalsePositiveRatio", min(1, args.MaxFalsePositiveRatio), "--MinOverlapRatio", max(1e-6, args.MinOverlapRatio), "--MinLesionVolume", args.MinLesionVolume, "-dlsTX", "-o", performance])

			xml_file = output_path / 'performance_global.xml'
			txt_file = output_path / 'performance_global.txt'
			xmldoc = minidom.parse(str(xml_file))

			xml_values = {}
			for e in xmldoc.getElementsByTagName('measure'):
				xml_values[e.getAttribute('name')] = float(e.firstChild.nodeValue)

			xml_file.unlink()
			txt_file.unlink()

			if args.localize_lesions and not patient_atlas.exists():
				core.preprocess.rigid_registration(utils.atlas_paths['flair'], reference, patient_atlas)

			image_paths['ground_truth'] = ground_truth
			image_paths['segmentation'] = segmentation
			image_paths['pmap'] = pmap
			image_paths['atlas'] = patient_atlas
			previews_path = lesion_previews_path / patient.name if not args.skip_generate_previews else None
			
			#Jaccard;	Dice;	Sensitivity;	Specificity;	PPV;	NPV;	RelativeVolumeError;	HausdorffDistance;	ContourMeanDistance;	SurfaceDistance;	PPVL;	SensL;	F1_score;	NbTestedLesions;	VolTestedLesions;
			if args.seg_perf_analyzer:
				f1_score, ppvl, sensl, true_positive_stats, false_negative_stats, false_positive_stats, n_lesions_ground_truth = core.evaluation.evaluate_segmentation_seg_perf_analyzer(image_paths, args.MinLesionVolume, args.MinOverlapRatio, args.MaxFalsePositiveRatio, args.MaxFalsePositiveRatioModerator, full_connectivity=True, previews_path=previews_path)
			else:
				f1_score = xml_values['F1_score'] if 'F1_score' in xml_values else None
				true_positive_stats, false_negative_stats, false_positive_stats, n_lesions_ground_truth = core.evaluation.evaluate_segmentation(image_paths, args.MinLesionVolume, full_connectivity=True, previews_path=previews_path)

			if args.remove_atlases and patient_atlas.exists():
				patient_atlas.unlink()

			patient_data = {
				'Patient': patient.name, 
				'F1': f1_score,
				'Dice': xml_values['Dice'] if 'Dice' in xml_values else None,
				
				'Jaccard': xml_values['Jaccard'] if 'Jaccard' in xml_values else None,
				'Sensitivity': xml_values['Sensitivity'] if 'Sensitivity' in xml_values else None,
				'Specificity': xml_values['Specificity'] if 'Specificity' in xml_values else None,	
				'PPV': xml_values['PPV'] if 'PPV' in xml_values else None,
				
				'HausdorffDistance': xml_values['HausdorffDistance'] if 'HausdorffDistance' in xml_values else None,
                                'ContourMeanDistance': xml_values['ContourMeanDistance'] if 'ContourMeanDistance' in xml_values else None,							     					'SurfaceDistance': xml_values['SurfaceDistance'] if 'SurfaceDistance' in xml_values else None,							      			        
				'NLesionsWhenEmptyGT': xml_values['NbTestedLesions'] if 'NbTestedLesions' in xml_values else 0,
				'VolumeWhenEmptyGT': xml_values['VolTestedLesions'] if 'VolTestedLesions' in xml_values else 0,
				'NLesions': n_lesions_ground_truth,
				'TP': len(true_positive_stats),
				'FN': len(false_negative_stats),
				'FP': len(false_positive_stats)
			}

			for stats_type, stats_list in [('TP', true_positive_stats), ('FN', false_negative_stats), ('FP', false_positive_stats)]:
				for var in ['PMax', 'PMean', 'PMedian']:
					patient_data[f'{stats_type}-{var}'] = 0
				patient_data[stats_type]
				for stats in stats_list:
					lesion_index = stats['index']
					lesions_stats_record = {
						'Model': model_name,
						'Patient': patient.name,
						'Index': lesion_index,
						'Type': stats_type,
						'Center': stats['center'],
						'Location': stats['locations'][0]['location'] if len(stats['locations']) > 0 else '',
						'Volume': stats['volume'],
						'Locations': str(stats['locations']),
						'PMax': stats['p_max'],
						'PMean': stats['p_mean'],
						'PMedian': stats['p_median'],
						'PStd': stats['p_std'],
					}
					for var in ['PMax', 'PMean', 'PMedian']:
						patient_data[f'{stats_type}-{var}'] += stats[var.replace('PM', 'p_m')] / len(stats_list)
					# exclude lesions_stats_records which have the same Model, Patient and Index
					lesions_stats_records = [lsr for lsr in lesions_stats_records if not (lsr['Model'] == model_name and lsr['Patient'] == patient.name and lsr['Index'] == lesion_index and lsr['Type'] == stats_type)]
					lesions_stats_records.append(lesions_stats_record)

					lesion_data = {
						'name': f'{model_name}_{patient.name}_{lesion_index}_{stats_type}', 		# Each lesion name must be unique, required to be able to retrieve the lesion for later processes
						'patient': patient.name,
						'index': lesion_index,
						'model': model_name,
						'location_voxel': str(stats['center']),				            # The center of the lesion in voxel space
						'bounding_box': str(list(stats['bounding_box'])),
						'description': f'Model: {model_name}, patient: {patient.name}, lesion: {lesion_index}, type: {stats_type}', 	# The description of the lesion
						'type': stats_type,                                         # The type of the lesion
						'images': image_descriptions,								# The list of images for the lesion
					}
					for var in ['volume', 'p_min', 'p_max', 'p_mean', 'p_median', 'locations']:
						lesion_data[var] = str(stats[var])
					task['lesions'].append(lesion_data)
			
			model_stats_records.append(patient_data)

			if args.localize_lesions:
				for lesion_stats, lesion_type in [(false_negative_stats, 'FN'), (false_positive_stats, 'FP'), (true_positive_stats, 'TP')]:
					for s in lesion_stats:
						if len(s['locations']) == 0: continue
						region = s['locations'][0]['location']
						model_locations.loc[region, lesion_type] += 1
						global_locations.loc[region, lesion_type] += 1

		model_stats = pandas.DataFrame.from_records(model_stats_records)

		if len(model_stats) == 0: continue
		
		model_stats = model_stats.round(args.round)
		model_stats.sort_values(by=['NLesions', 'Patient'], ascending=(False, True), inplace=True)
		
		for format in formats:
			sep = '\t' if format == 'tsv' else ','
			model_stats.to_csv(model_stats_path, sep=sep, index=False)
			if args.localize_lesions:
				model_locations.to_csv(model_locations_path, sep=sep)
		
	model_stats['Model'] = model_name
	cols = model_stats.columns.tolist()
	model_stats = model_stats[cols[-1:] + cols[:-1]]

	all_models_stats = pandas.concat([all_models_stats, model_stats])
		
	model_stats_with_lesions = model_stats[model_stats['NLesions'] > 0]
	model_stats_without_lesions = model_stats[model_stats['NLesions'] == 0]

	training_dataset_yml = {}
	testing_dataset_yml = {}
	postprocessing_yml = {}
	model_description = ''
	custom_nnunet_parameters = ''
	disable_tta = ''

	try:
		training_dataset_yml['name'] = prediction_description['training_dataset_name']
		testing_dataset_yml['name'] = prediction_description['testing_dataset_name']
		training_dataset_yml['step'] = prediction_description['training_dataset_description']['preprocessings'][-1]['preprocess_steps'][-1]
		testing_dataset_yml['step'] = prediction_description['testing_dataset_description']['preprocessings'][-1]['preprocess_steps'][-1]
		training_dataset_yml['modalities'] = prediction_description['training_dataset_description']['preprocessings'][-1]['modalities']
		testing_dataset_yml['modalities'] = prediction_description['testing_dataset_description']['preprocessings'][-1]['modalities']
		postprocessing_yml['steps'] = prediction_description['postprocessing_steps']
		postprocessing_yml['minpmap'] = prediction_description["postprocessing_parameters"]["min_pmap_threshold"]
		postprocessing_yml['maxpmap'] = prediction_description["postprocessing_parameters"]["max_pmap_threshold"]
		postprocessing_yml['minvol'] = prediction_description["postprocessing_parameters"]["min_volume"]
		model_description = prediction_description['model']['model_description']
		custom_nnunet_parameters = prediction_description['model']['custom_nnunet_parameters']
		disable_tta = prediction_description['disable_tta']
	except Exception as e:
		print('Error while reading prediction description:', e)
	
	model_data = {
		'Model': model_name,

		'F1':  model_stats_with_lesions['F1'].mean(),
		'Dice':  model_stats_with_lesions['Dice'].mean(),

		'Jaccard':  model_stats_with_lesions['Jaccard'].mean(),
		'Sensitivity':  model_stats_with_lesions['Sensitivity'].mean(),
		'Specificity':  model_stats_with_lesions['Specificity'].mean(),
		'PPV':  model_stats_with_lesions['PPV'].mean(),
		'HausdorffDistance':  model_stats_with_lesions['HausdorffDistance'].mean(),
		'ContourMeanDistance':  model_stats_with_lesions['ContourMeanDistance'].mean(),	
		'SurfaceDistance':  model_stats_with_lesions['SurfaceDistance'].mean(),	
					
		'NLesionsWhenEmptyGT':  model_stats_without_lesions['NLesionsWhenEmptyGT'].mean(),
		'VolumeWhenEmptyGT':  model_stats_without_lesions['VolumeWhenEmptyGT'].mean(),
		'NLesions': model_stats['NLesions'].sum(),
		'TP':  model_stats['TP'].sum(),
		'FN':  model_stats['FN'].sum(),
		'FP':  model_stats['FP'].sum(),
		'NPatients': len(model_stats),

		'ModelDescription': model_description,
		'TrainingDataset': json.dumps(training_dataset_yml) if len(training_dataset_yml) > 0 else '',
		'TestingDataset': json.dumps(testing_dataset_yml) if len(testing_dataset_yml) > 0 else '',
		'PostProcessing': json.dumps(postprocessing_yml) if len(postprocessing_yml) > 0 else '',
		'CustomNNUnetParameters': custom_nnunet_parameters,
		'DisableTTA': disable_tta,
		
		'EvaluationParameters': json.dumps({
			'MinLesionVolume': args.MinLesionVolume,
			'MinOverlapRatio': args.MinOverlapRatio,
			'MaxFalsePositiveRatio': args.MaxFalsePositiveRatio,
			'MaxFalsePositiveRatioModerator': args.MaxFalsePositiveRatioModerator
		})
	}
	global_stats_records.append(model_data)

if len(task['lesions'])>0:
	with open(str(task_path), 'w') as f:
		json.dump(task, f, indent=4)

lesions_stats = None
if len(lesions_stats_records)>0:
	lesions_stats = pandas.DataFrame.from_records(lesions_stats_records)
	lesions_stats = lesions_stats.round(args.round)
	lesions_stats.sort_values(by=['Model', 'Patient', 'Type'], ascending=(True, True, True), inplace=True)

global_stats = pandas.DataFrame.from_records(global_stats_records)

global_stats = global_stats.round(args.round)
global_stats.sort_values(by=['F1', 'Model'], ascending=(False, True), inplace=True)

if 'csv' not in formats: formats.append('csv') # Add csv to use it in web page

all_models_stats.sort_values(by=['Model', 'Patient'], ascending=(True, True), inplace=True)

for format in formats:
	sep = '\t' if format == 'tsv' else ','
	global_stats.to_csv(output_path / f'methods.{format}', sep=sep, index=False)
	all_models_stats.to_csv(output_path / f'methods_by_patients.{format}', sep=sep, index=False)
	if lesions_stats is not None:
		lesions_stats.to_csv(lesions_stats_path, sep=sep, index=False)
	global_locations_path = output_path / f'locations.{format}'
	if args.localize_lesions and (not global_locations_path.exists() or args.force_recompute):
		global_locations.to_csv(global_locations_path, sep=sep)

def publish_file(file_path, file_data):
	methods_csv_url = f'https://api.github.com/repos/{os.environ["github_owner"]}/{os.environ["github_repository"]}/contents/{file_path}'

	headers = { 'Accept': 'application/vnd.github.v3+json', 'Authorization': f'token {os.environ["github_personal_access_token"]}' }
	response = requests.get(methods_csv_url, headers=headers)

	if response.status_code != 200:
		print(f'Get {file_path} returned {response.status_code}:')
		print(response.content)
		return response

	data = {
		'message': 'update longiseg4ms evaluation',
		'content': base64.b64encode(file_data.encode()).decode(),
		'sha': response.json()['sha'] if response.status_code == 200 else None
	}

	response = requests.put(methods_csv_url, headers=headers, data=json.dumps(data))

	if response.status_code != 200:
		print(f'Put {file_path} returned {response.status_code}:')
		print(response.content)
	return response

# Publish methods.csv
methods_csv_path = f'{evaluation_name}.csv'
methods_csv_data = None
with open(output_path / 'methods.csv', 'r') as f:
	methods_csv_data = f.read()
response = publish_file(methods_csv_path, methods_csv_data)

# Publish methods.html
methods_html_path = f'{evaluation_name}.html'
html_template_path = Path('templates/template.html')
html_template = None
if html_template_path.exists():
	with open(html_template_path, 'r') as f:

		html_template = f.read()
methods_html_data = html_template.replace('methods.csv', methods_csv_path)
response = publish_file(methods_html_path, methods_html_data)

if response.status_code == 200:
	print(f'The page was successfully published to https://{os.environ["github_owner"]}.github.io/{os.environ["github_repository"]}/{methods_html_path}')

if not args.skip_generate_graphs:

	colors = ['rgb(15, 69, 168)', 'rgb(116, 27, 71)', 'rgb(204, 0, 0)']
	cols = ['Model', 'Patient', 'TP', 'FN', 'FP']
	y_names = ['TP', 'FN', 'FP']

	all_models_stats.sort_values(by=['NLesions', 'Patient'], ascending=False)
	all_models_stats = all_models_stats[cols]
	all_models_stats.reset_index()

	patients_path = output_path / 'patients'
	patients_path.mkdir(exist_ok=True)
	methods_path = output_path / 'models'
	methods_path.mkdir(exist_ok=True)

	for patient, df in all_models_stats.groupby('Patient'):
		print(patient)
		df = df[['Model', 'TP', 'FN', 'FP']]
		fig = df.plot.bar(x='Model', y=y_names, barmode='group', text='value', labels=dict(index='', value='', variable='', Model='Experts and Models'), width=1200, height=800, color_discrete_sequence=colors)
		fig.write_image(patients_path / f'{patient}.png')

	for model, df in all_models_stats.groupby('Model'):
		print(model)
		df = df[['Patient', 'TP', 'FN', 'FP']]
		fig = df.plot.bar(x='Patient', y=y_names, barmode='group', labels=dict(index='', value='', variable='', Patient='Patients'), width=1200, height=800, color_discrete_sequence=colors)
		fig.write_image(methods_path / f'{model}.png')

	global_stats = global_stats[['Model', 'TP', 'FN', 'FP']]
	fig = global_stats.plot.bar(x='Model', y=y_names, barmode='group', text='value', labels=dict(index='', value='', variable='', Model='Experts and models'), width=1200, height=800, color_discrete_sequence=colors)
	fig.write_image(output_path / 'models.png')
