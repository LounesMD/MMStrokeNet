import math
import logging
import  yaml
import SimpleITK as sitk
import numpy as np
import core.utils as utils

valid_steps = ['threshold', 'remove_small', 'remove_low_keep_largest', 'remove_low', 'remove_external_lesions']
default_steps = ['threshold', 'remove_small', 'remove_low_keep_largest', 'remove_external_lesions']

def create_prediction_description_file(args, nnunet_task_folder, model_task_name, output_folder):

	testing_dataset_description = utils.open_yml(nnunet_task_folder / 'description_testing.yml')
	training_dataset_description = utils.open_yml(nnunet_task_folder / 'description_training.yml')
	model_description = utils.open_yml(utils.nnunet_model_folder / 'nnUNet' / args.model_architecture / model_task_name / 'description.yml')
	
	description = {
		'task_name': nnunet_task_folder.name,
		'model_name': model_task_name,
		'model': model_description,
		'disable_tta': utils.get_configuration('custom_nnunet_parameters', 'disable_tta'),
		'testing_dataset_name': testing_dataset_description['dataset_name'] if 'dataset_name' in testing_dataset_description else '',
		'testing_dataset_description': testing_dataset_description,
		'training_dataset_name': training_dataset_description['dataset_name'] if 'dataset_name' in training_dataset_description else '',
		'training_dataset_description': training_dataset_description,
		'postprocessing_parameters': {
			'min_pmap_threshold': args.min_pmap_threshold,
			'max_pmap_threshold': args.max_pmap_threshold,
			'min_volume': args.min_volume,
		},
		'postprocessing_steps': args.postprocess_steps,
	}

	utils.write_yml(output_folder / 'description.yml', description)

	return

def update_postprocessing_decription_file(model_output, postprocessed_predictions_path, postprocess_steps, min_pmap_threshold, max_pmap_threshold, min_volume):

	prediction_description_path = model_output / 'description.yml'
	try:
		prediction_description = utils.open_yml(prediction_description_path, fail_silently=False)
	except Exception as e:
		print(f'Error while creating postprocessing decription file: unable to read original description file {prediction_description_path}, {e}.')
		return

	if 'postprocessing_parameters' not in prediction_description or 'min_pmap_threshold' not in prediction_description['postprocessing_parameters']:
		print('Error while creating postprocessing decription file: original file was incomplete')
		return

	prediction_description['postprocessing_parameters']['min_pmap_threshold'] = min_pmap_threshold
	prediction_description['postprocessing_parameters']['max_pmap_threshold'] = max_pmap_threshold
	prediction_description['postprocessing_parameters']['min_volume'] = min_volume
	prediction_description['postprocessing_steps'] = postprocess_steps
	
	utils.write_yml(postprocessed_predictions_path / 'description.yml', prediction_description)

	return

def postprocess_patient(segmentation, pmap, reference, cross_sectional, steps, min_pmap_threshold, max_pmap_threshold, min_volume, id_xml=None, output=None, patient_folder=None, modalities=None):

	for step in steps:
		if step not in valid_steps:
			print(f'Warning, {step} is not a valid postprocessing step! It will be ignored.')

	output = output or segmentation
	pmap_image = sitk.ReadImage(str(pmap))
	pmap_data = sitk.GetArrayFromImage(pmap_image)

	if 'threshold' in steps:
		btif = sitk.BinaryThresholdImageFilter()
		btif.SetLowerThreshold(min_pmap_threshold)
		thresholded_pmap_image = btif.Execute(pmap_image)

		thresholded_pmap_data = sitk.GetArrayFromImage(thresholded_pmap_image)
	else:
		thresholded_pmap_data = sitk.ReadImage(str(segmentation))

	if 'remove_external_lesions' in steps and not cross_sectional:
			
		patient_structure = utils.get_patient_structure(cross_sectional)

		if not modalities:
			modalities = utils.get_modalities(cross_sectional)
		
		time_steps_intersections = None
		for time in  patient_structure['times']:
			modality_union = None
			for modality_type in modalities:
				modality = time['modalities'][modality_type]
				modality_path = patient_folder / modality
				if not modality_path.exists(): continue
				modality_image_mask = abs(sitk.ReadImage(str(modality_path)))>1e-6
				modality_union = modality_image_mask if modality_union is None else sitk.Or(modality_union, modality_image_mask)
			time_steps_intersections = modality_union if time_steps_intersections is None else sitk.And(time_steps_intersections, modality_union)
		
		time_steps_intersections_data = sitk.GetArrayFromImage(time_steps_intersections) if time_steps_intersections is not None else None

	# Extract lesions

	ccifilter = sitk.ConnectedComponentImageFilter()
	ccifilter.FullyConnectedOn()
	labeled = ccifilter.Execute(thresholded_pmap_image)
	labeledData = sitk.GetArrayFromImage(labeled)
	lsifilter = sitk.LabelStatisticsImageFilter()
	lsifilter.Execute(labeled, labeled)
	n_components = ccifilter.GetObjectCount()

	spacing = pmap_image.GetSpacing()
	voxel_volume = spacing[0] * spacing[1] * spacing[2]

	# For each lesion, remove if too small
	# or max pmap value is too low
	# and remove voxels below max_pmap_threshold but keep n largest voxels of lesions (n being the minimum number of voxels in a lesion)

	for i in range(1, n_components+1):
		n_voxels = lsifilter.GetCount(i)
		lesion_volume = n_voxels * voxel_volume
		min_volume_in_voxels = math.ceil(min_volume / voxel_volume)

		if 'remove_external_lesions' in steps and not cross_sectional and time_steps_intersections_data is not None: # remove lesions which are outside timesteps intersection
			
			values_under_intersections = np.extract(labeledData == i, time_steps_intersections_data)

			n_nonzeros = np.count_nonzero(values_under_intersections)
			n_zeros = len(values_under_intersections) - n_nonzeros
			if n_zeros > 0:

				remove_inside_voxels = 'remove_small' in steps and n_nonzeros * voxel_volume < min_volume
				if remove_inside_voxels:
					thresholded_pmap_data[np.logical_and(labeledData == i, time_steps_intersections_data > 0)] = 0
				
				logging.info(f'The lesion {i} is outside the timesteps intersection: {n_nonzeros} voxels inside, {n_zeros} outside. ' + ('Inside voxels were removed since too small.' if remove_inside_voxels else 'Inside voxels were NOT removed!') )
			
			thresholded_pmap_data[np.logical_and(labeledData == i, time_steps_intersections_data == 0)] = 0

		if 'remove_small' in steps and lesion_volume < min_volume:      # if lesion volume is too small: remove lesion 
			thresholded_pmap_data[labeledData == i] = 0
			continue
		
		if 'remove_low' in steps and np.max(np.extract(labeledData == i, pmap_data)) < max_pmap_threshold:      # if max voxel of lesion is too small: remove lesion 
			thresholded_pmap_data[labeledData == i] = 0
			continue

		if 'remove_low_keep_largest' in steps:
			lesion_voxels = sorted(np.extract(labeledData == i, pmap_data), reverse=True)
			nth_voxel_value = lesion_voxels[min(min_volume_in_voxels, len(lesion_voxels)-1)]
			thresholded_pmap_data[np.logical_and(labeledData == i, pmap_data < min(max_pmap_threshold, nth_voxel_value))] = 0
			continue
		

	segmentation_image = sitk.GetImageFromArray(thresholded_pmap_data)
	segmentation_image.CopyInformation(pmap_image)
	sitk.WriteImage(segmentation_image, str(segmentation))

	# Put output segmentation in original space:
	if reference is None or not reference.exists():
		if output != segmentation:
			utils.copy_file(segmentation, output)
	else:
		utils.call([utils.anima / 'animaApplyTransformSerie', '-i', segmentation, '-g', reference, '-o', output, '-t', id_xml, '-n', 'nearest'])

	return output

def postprocess_patients(raw_patients_folder, output_folder, args, id_xml=None, modalities=None):

	id_xml = id_xml or utils.get_or_create_identity_transform_serie()

	for pmap in sorted(list(output_folder.glob('*pmap-1.nii.gz'))):

		segmentation = utils.replace_path_suffix(pmap, '.nii.gz', '_thresholded.nii.gz')
		patient_name = utils.replace_string_suffix(pmap.name, 'pmap-1.nii.gz', '')
		output = pmap.parent / utils.get_file_format('final_segmentation', '{patient_name}_final_segmentation.nii.gz').format(patient_name=patient_name)

		# If cross sectional: remove last suffix which refers to the time point
		if args.cross_sectional:
			patient_name = '_'.join(patient_name.split('_')[:-1])

		print('   postprocess ' + patient_name + '...')
		
		reference = raw_patients_folder / patient_name / utils.get_patient_structure(args.cross_sectional)['reference'] if raw_patients_folder is not None else None

		postprocess_patient(segmentation, pmap, reference, args.cross_sectional, args.postprocess_steps, args.min_pmap_threshold, args.max_pmap_threshold, args.min_volume, id_xml, output, raw_patients_folder / patient_name, modalities)
