import logging
import math
import SimpleITK as sitk
import numpy as np
import core.preview as preview

# This code is a clone of animaSegPerfAnalyzer with the addition of lesion localisation and TP, FP and FN counts

# See section 11 of the article "Objective Evaluation of Multiple Sclerosis Lesion Segmentation using a Data Management and Processing Infrastructure" https://www.nature.com/articles/s41598-018-31911-7#Sec11

# A ground truth lesion is detected if:
# - the ground truth lesion is overlapped at least at a rate of α% by the segmentation lesions
# - segmentation lesions that contribute the most to the detection of the ground truth lesion (summing up to γ% of the total overlap) do not go outside of the ground truth lesion by more than β%.

# detection_threshold_alpha = α% = minimum sensibility for a lesion ( sensibility = n_detected_voxels / float(n_detected_voxels + n_missed_voxels) )
# detection_threshold_beta = β% => n missed voxels of test lesion / n voxels of test lesion > detection_threshold_beta = invalid_ratio_max)
# detection_threshold_gamma = γ% => stop looking for the largest segmentation lesion once we have γ% of the ground truth lesion voxels

def get_components(image, min_size, full_connectivity=False):
    ccifilter = sitk.ConnectedComponentImageFilter()
    ccifilter.SetFullyConnected(full_connectivity)
    labeled = ccifilter.Execute(image)
    rcif = sitk.RelabelComponentImageFilter()
    rcif.SetMinimumObjectSize(min_size)
    labeled = rcif.Execute(labeled)
    labeled_data = sitk.GetArrayFromImage(labeled)
    ncomponents = rcif.GetNumberOfObjects()
    if ncomponents != ccifilter.GetObjectCount():
        logging.warning('Number of components in the connected component image filter and the relabel component image filter do not match!')

    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(labeled, labeled)
    return labeled_data, ncomponents, lsif

# named falsePositiveRatioTester in animaSegPerfAnalyzer
def is_true_positive(i, n_lesions_ground_truth, n_lesions_segmentation, lesion_statistics, n_detected_voxels_of_ground_truth_lesion, n_voxels_of_segmentation_lesions, detection_threshold_beta, detection_threshold_gamma):
    ratio_outside_inside = 0
    segmentation_lesions_of_ground_truth_lesion_i = sorted([(j, lesion_statistics[i][j]) for j in range(1, n_lesions_segmentation)], key=lambda x: x[1], reverse=True)
    
    # Test in intersection size decreasing order that the regions overlapping the tested lesion are not too much outside of this lesion
    total_intersection_ratio = 0
    true_positive = True

    for segmentation_lesion_j, n_voxels_in_lesion_i_and_lesion_j in segmentation_lesions_of_ground_truth_lesion_i:

        n_false_positive_voxels_of_lesion_j = lesion_statistics[0][segmentation_lesion_j]
        ratio_outside_inside = n_false_positive_voxels_of_lesion_j / n_voxels_of_segmentation_lesions[segmentation_lesion_j]
        
        # if n missed voxels of test lesion / n voxels of test lesion > beta = maxFalsePositiveRatio
        # = if test lesion is too much outside reference lesion
        if ratio_outside_inside > detection_threshold_beta:
            true_positive = False
            break

        total_intersection_ratio += n_voxels_in_lesion_i_and_lesion_j / n_detected_voxels_of_ground_truth_lesion

        if total_intersection_ratio > detection_threshold_gamma:
            break

    return true_positive

def get_bounding_box_center(bounding_box):
    return [ (bounding_box[2*i] + bounding_box[2*i+1]) // 2 for i in range(3) ]

# named getTruePositiveLesions in animaSegPerfAnalyzer
def get_lesions(n_lesions_ground_truth, n_lesions_segmentation, lesion_statistics, labeled_data, lsif, atlas_data, pmap_data, detection_threshold_alpha, detection_threshold_beta, detection_threshold_gamma, voxel_volume, resampled_images, preview_path, is_ground_truth):

    n_detected_voxels_of_ground_truth_lesions = [0 for x in range(n_lesions_ground_truth)] # n voxels of ground truth lesion which are correctly detected = nTruePositiveVoxels
    n_voxels_of_segmentation_lesions = [0 for x in range(n_lesions_segmentation)] # n voxels of segmentation lesion

    for i in range(n_lesions_ground_truth):
        for j in range(n_lesions_segmentation):
            n_voxels_of_segmentation_lesions[j] += lesion_statistics[i][j]
            if j>0:
                n_detected_voxels_of_ground_truth_lesions[i] += lesion_statistics[i][j]

    positive_stats = []
    negative_stats = []
    
    for i in range(1, n_lesions_ground_truth):
        
        n_detected_voxels_of_ground_truth_lesion = n_detected_voxels_of_ground_truth_lesions[i]
        n_missed_voxels_of_ground_truth_lesion = lesion_statistics[i][0]
        
        # Compute sensibility for one element of the ground-truth
        sensibility = n_detected_voxels_of_ground_truth_lesion / float(n_detected_voxels_of_ground_truth_lesion + n_missed_voxels_of_ground_truth_lesion)

        is_positive = sensibility > detection_threshold_alpha and is_true_positive(i, n_lesions_ground_truth, n_lesions_segmentation, lesion_statistics, n_detected_voxels_of_ground_truth_lesion, n_voxels_of_segmentation_lesions, detection_threshold_beta, detection_threshold_gamma)
        lesion_type = is_positive and not is_ground_truth
        stats = get_lesion_stats(atlas_data, pmap_data, labeled_data == i, i, lsif, voxel_volume, resampled_images, preview_path / lesion_type if preview_path else None, is_positive and not is_ground_truth)

        if is_positive:
            positive_stats.append(stats)
        else:
            negative_stats.append(stats)

    return positive_stats, negative_stats

# named getDetectionMarks is animaSegPerfAnalyzer
def evaluate_segmentation_seg_perf_analyzer(image_paths, min_lesion_volume, detection_threshold_alpha, detection_threshold_beta, detection_threshold_gamma, full_connectivity=False, previews_path=None):
    
    ground_truth_image = sitk.ReadImage(str(image_paths['ground_truth']), sitk.sitkFloat64)
    segmentation_image = sitk.ReadImage(str(image_paths['segmentation']), sitk.sitkFloat64)
    ground_truth_image = sitk.Cast(sitk.BinaryThreshold(ground_truth_image, 0.5, 1.5), sitk.sitkInt8)
    segmentation_image = sitk.Cast(sitk.BinaryThreshold(segmentation_image, 0.5, 1.5), sitk.sitkInt8)

    pmap_image = sitk.ReadImage(str(image_paths['pmap'])) if image_paths['pmap'].exists() else None
    pmap_data = sitk.GetArrayFromImage(pmap_image) if pmap_image else None

    spacing = ground_truth_image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    min_size_in_voxel = math.floor(min_lesion_volume / voxel_volume) + 1    # see https://github.com/Inria-Empenn/Anima-Public/blob/master/Anima/segmentation/validation_tools/segmentation_performance_analyzer/animaSegPerfCAnalyzer.cxx#L330

    atlas_image = sitk.ReadImage(str(image_paths['atlas']), sitk.sitkInt8) if image_paths['atlas'].exists() else None
    atlas_data = sitk.GetArrayFromImage(atlas_image) if atlas_image else None

    ground_truth_labeled_data, n_lesions_ground_truth, lsif_ground_truth = get_components(ground_truth_image, min_size_in_voxel, full_connectivity)
    segmentation_labeled_data, n_lesions_segmentation, lsif_segmentation = get_components(segmentation_image, min_size_in_voxel, full_connectivity)

    n_lesions_ground_truth += 1
    n_lesions_segmentation += 1

    ground_truth_statistics = np.zeros((n_lesions_ground_truth, n_lesions_segmentation), np.uint64)
    segmentation_statistics = np.zeros((n_lesions_segmentation, n_lesions_ground_truth), np.uint64)

    for i in range(0, n_lesions_ground_truth):
        
        segmentation_under_lesion = np.extract(ground_truth_labeled_data == i, segmentation_labeled_data)
        
        uc = zip(*np.unique(segmentation_under_lesion, return_counts=True))
        for unique, count in uc:
            ground_truth_statistics[i][unique] = count
            segmentation_statistics[unique][i] = count

    positive_stats_ground_truth = []
    false_negative_stats = []
    positive_stats_segmentations = []
    false_positive_stats = []
    
    resampled_images = preview.get_resampled_images(image_paths) if previews_path is not None else []

    positive_stats_ground_truth, false_negative_stats = get_lesions(n_lesions_ground_truth, n_lesions_segmentation, ground_truth_statistics, ground_truth_labeled_data, lsif_ground_truth, atlas_data, pmap_data, detection_threshold_alpha, detection_threshold_beta, detection_threshold_gamma, voxel_volume, resampled_images, previews_path, True)
    positive_stats_segmentations, false_positive_stats  = get_lesions(n_lesions_segmentation, n_lesions_ground_truth, segmentation_statistics, segmentation_labeled_data, lsif_segmentation, atlas_data, pmap_data,detection_threshold_alpha, detection_threshold_beta, detection_threshold_gamma, voxel_volume, resampled_images, previews_path, False)
    
    if not (n_lesions_ground_truth > 1 and n_lesions_segmentation > 1):
        positive_stats_ground_truth = []
        positive_stats_segmentations = []

    n_true_positives_ground_truth = len(positive_stats_ground_truth)
    n_true_positives_segmentation = len(positive_stats_segmentations)

    positive_predictive_value = n_true_positives_segmentation / (n_lesions_segmentation - 1) if n_lesions_segmentation > 1 else 0

    sensitivity = n_true_positives_ground_truth / (n_lesions_ground_truth - 1) if n_lesions_ground_truth > 1 else 0

    f1_score = 2 * positive_predictive_value * sensitivity / ( positive_predictive_value + sensitivity ) if positive_predictive_value + sensitivity > 0 else 0

    n_lesions_ground_truth -= 1

    return f1_score, positive_predictive_value, sensitivity, positive_stats_ground_truth, false_negative_stats, false_positive_stats, n_lesions_ground_truth

def get_lesion_stats(atlas_data, pmap_data, lesion, i, lsif, voxel_volume, resampled_images, preview_path, ignore_preview=False):
    bounding_box = lsif.GetBoundingBox(i)
    center = get_bounding_box_center(bounding_box)
    if not ignore_preview and len(resampled_images) > 0 and preview_path is not None:
        preview.create_lesion_previews(resampled_images, bounding_box, preview_path / f'lesion_{i}')
    stats = dict(locations=[], p_mean=0, p_std=0, p_median=0, p_min=0, p_max=0, volume=len(lesion) * voxel_volume, index=i, center=center, bounding_box=bounding_box)
    if pmap_data is not None:
        pmap_under_lesion = np.extract(lesion, pmap_data)
        if len(pmap_under_lesion) > 0:
            stats['p_mean'] = np.mean(pmap_under_lesion)
            stats['p_std'] = np.std(pmap_under_lesion)
            stats['p_median'] = np.median(pmap_under_lesion)
            stats['p_min'] = np.min(pmap_under_lesion)
            stats['p_max'] = np.max(pmap_under_lesion)
    if atlas_data is None:
        return stats
    regions_under_lesion = np.extract(lesion, atlas_data)
    # get the proportion of voxels for each region
    regions_under_lesion_count = np.unique(regions_under_lesion, return_counts=True)
    # sort the regions by the number of voxels
    regions_under_lesion_count = sorted(zip(*regions_under_lesion_count), key=lambda x: x[1], reverse=True)
    # get the volume of each location
    stats['locations'] = [dict(location=x[0], volume=voxel_volume * x[1]) for x in regions_under_lesion_count]
    return stats

def get_positive_and_negative_stats(n_lesions, labeled_data, test_data, atlas_data, pmap_data, lsif, voxel_volume, resampled_images, preview_path, is_ground_truth, valid_values = [1, 3]):
    positive_stats = []
    negative_stats = []
    for i in range(1, n_lesions+1):
        lesion = labeled_data == i
        values_under_lesion_i = np.extract(lesion, test_data)
        is_positive = len(np.intersect1d(values_under_lesion_i, valid_values)) > 0
        lesion_type = 'TP' if is_positive else 'FN' if is_ground_truth else 'FP'
        stats = get_lesion_stats(atlas_data, pmap_data, lesion, i, lsif, voxel_volume, resampled_images, preview_path / lesion_type if preview_path else None, is_positive and not is_ground_truth)
        # if values_under_lesion_i contains valid values, then the lesion is detected
        if is_positive:
            positive_stats.append(stats)
        else:
            negative_stats.append(stats)
    return positive_stats, negative_stats

def evaluate_segmentation(image_paths, min_lesion_volume, full_connectivity=False, previews_path=None):

    ground_truth_image = sitk.ReadImage(str(image_paths['ground_truth']), sitk.sitkFloat64)
    segmentation_image = sitk.ReadImage(str(image_paths['segmentation']), sitk.sitkFloat64)
    ground_truth_image = sitk.Cast(sitk.BinaryThreshold(ground_truth_image, 0.5, 1.5), sitk.sitkInt8)
    segmentation_image = sitk.Cast(sitk.BinaryThreshold(segmentation_image, 0.5, 1.5), sitk.sitkInt8)

    ground_truth_data = sitk.GetArrayFromImage(ground_truth_image)
    segmentation_data = sitk.GetArrayFromImage(segmentation_image)

    spacing = ground_truth_image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    min_size_in_voxel = round(min_lesion_volume / voxel_volume)

    atlas_image = sitk.ReadImage(str(image_paths['atlas']), sitk.sitkInt8) if 'atlas' in image_paths and image_paths['atlas'].exists() else None
    atlas_data = sitk.GetArrayFromImage(atlas_image) if atlas_image else None
    pmap_image = sitk.ReadImage(str(image_paths['pmap'])) if image_paths['pmap'].exists() else None
    pmap_data = sitk.GetArrayFromImage(pmap_image) if pmap_image else None

    resampled_images = preview.get_resampled_images(image_paths) if previews_path is not None else []
    
    ground_truth_labeled_data, n_lesions_ground_truth, lsif_ground_truth = get_components(ground_truth_image, min_size_in_voxel, full_connectivity)
    segmentation_labeled_data, n_lesions_segmentation, lsif_segmentation = get_components(segmentation_image, min_size_in_voxel, full_connectivity)
    
    positive_stats_ground_truth, false_negative_stats = get_positive_and_negative_stats(n_lesions_ground_truth, ground_truth_labeled_data, segmentation_data, atlas_data, pmap_data, lsif_ground_truth, voxel_volume, resampled_images, previews_path, True)
    positive_stats_segmentation, false_positive_stats = get_positive_and_negative_stats(n_lesions_segmentation, segmentation_labeled_data, ground_truth_data, atlas_data, pmap_data, lsif_segmentation, voxel_volume, resampled_images, previews_path, False)

    if len(false_negative_stats) + len(positive_stats_ground_truth) != n_lesions_ground_truth:
        logging.warning('Number of detected and missed lesions does not match number of ground truth lesions!')
        
    return positive_stats_ground_truth, false_negative_stats, false_positive_stats, n_lesions_ground_truth
