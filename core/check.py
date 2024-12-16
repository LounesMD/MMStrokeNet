from pathlib import Path
import logging
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import core.utils as utils
Path.ls = lambda x: sorted(list(x.iterdir()))

def check_orientation(image_path, orientations):
    image = nib.load(str(image_path))
    orientation = str(nib.aff2axcodes(image.affine))
    if orientation not in orientations:
        orientations[orientation] = []
    orientations[orientation].append(str(image_path))
    return

def check_patient(patient, modalities, cross_sectional, check_match, orientations={}, full=False):
        patient_structure = utils.get_patient_structure(cross_sectional)

        logging.info('   ' + patient.name)
        reference_path = patient / patient_structure['reference']

        if not reference_path.exists():
            logging.error(str(reference_path) + ' does not exist.')
            return
        
        try:
            reference = sitk.ReadImage(str(reference_path))
        except Exception as e:
            logging.error(str(reference_path) + ' could not be read.')
            logging.error(e)
            return
        
        for i, time in enumerate(patient_structure['times']):
            logging.info('     time ' + str(i))

            for modality_type in modalities:

                modality = time['modalities'][modality_type]

                modality_path = patient / modality

                if not modality_path.exists():
                    logging.error(str(modality_path) + ' does not exist.')
                    continue

                if modality_path != reference_path and check_match:

                    match, image1, image2 = utils.check_images_match(modality_path, reference_path, image2=reference, use_logging=True)
                    if full:
                        image1_data = sitk.GetArrayFromImage(image1)
                        logging.info('        ' + modality_path.name + ' has type ' + str(image1_data.dtype) + ', min: ' + str(np.min(image1_data)) + ', max: ' + str(np.max(image1_data)) )

                check_orientation(modality_path, orientations)
            
            segmentation_path = patient / time['segmentation']

            if not segmentation_path.exists():
                if i == 0:
                    logging.info('        ' + str(segmentation_path) + ' does not exist.')
                else:
                    logging.error(str(segmentation_path) + ' does not exist.')
                continue

            match, image1, image2 = utils.check_images_match(segmentation_path, reference_path, image2=reference, use_logging=True)

            if full:
                image1_data = sitk.GetArrayFromImage(image1)

                logging.info('        ' + segmentation_path.name + ' has type ' + str(image1_data.dtype) + ', min: ' + str(np.min(image1_data)) + ', max: ' + str(np.max(image1_data)) )

                unique, counts = np.unique(image1_data, return_counts=True)

                logging.info('        ' + segmentation_path.name + ' has unique: counts:' + str(dict(zip(unique, counts))))

                if len(unique) > 2:
                    logging.warning(str(segmentation_path) + ' has more than 2 values.')
                
            check_orientation(segmentation_path, orientations)