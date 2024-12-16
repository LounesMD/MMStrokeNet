import os
import sys
import json
from pathlib import Path
import tempfile
import SimpleITK as sitk
import core.utils as utils

def ensemble(patients, model_outputs_path, cross_sectional):

	id_xml = utils.get_or_create_identity_transform_serie()

	model_outputs_path = Path(model_outputs_path)
	model_outputs = sorted(list(model_outputs_path.iterdir()))
	model_outputs = [mo for mo in model_outputs if 'ensemble' not in mo.name]
	
	weights = utils.configuration['ensembling_weights'] if 'ensembling_weights' in utils.configuration else None
	if weights is not None:
		model_outputs = [mo for mo in model_outputs if mo.name in weights.keys()]

	# suffix = '_'.join([m.name+':'+weights[m.name] for m in model_outputs])
	suffix = '-'.join([str(w) for m, w in weights.items()])
	output_path = Path(model_outputs_path / f'ensemble_{suffix}')
	output_path.mkdir(exist_ok=True, parents=True)

	for pmap_path in model_outputs[0].glob('*pmap-1.nii.gz'):
		
		print(pmap_path)
		patient_name = utils.replace_string_suffix(pmap_path.name, 'pmap-1.nii.gz', '')

		reference = patients / patient_name / utils.get_patient_structure(cross_sectional)['reference'] if patients is not None else None

		pmap = None
		for model_output in model_outputs:
			
			if not model_output.is_dir(): continue
			if weights[model_output.name] < 1e-6: continue

			with tempfile.TemporaryDirectory() as tmp_dir:
				if reference is not None:
					output = os.path.join(tmp_dir, 'pmap.nii.gz')
					utils.call([utils.anima / 'animaApplyTransformSerie', '-i', model_output / pmap_path.name, '-g', reference, '-o', output, '-t', id_xml, '-n', 'linear'])
				else:
					output = model_output / pmap_path.name
				image = sitk.ReadImage(str(output))
				image *= weights[model_output.name] if weights is not None else 1/len(model_outputs)
				pmap = image if pmap is None else pmap + image

		sitk.WriteImage(pmap, str(output_path / pmap_path.name))
	
	return output_path