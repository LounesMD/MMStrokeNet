from pathlib import Path
import SimpleITK as sitk
import numpy as np
import core.utils as utils
Path.ls = lambda x: sorted(list(x.iterdir()))

def get_image_end(image):
	return [c-1 for c in image.GetSize()]

def get_bounding_box_origin(bb):
	return [bb[0], bb[2], bb[4]]

def get_bounding_box_end(bb):
	return [bb[1], bb[3], bb[5]]

def get_expanded_bounding_box(max_size, bounding_box, amount):
	return utils.flatten([ ( max(0, bounding_box[2*i]-amount), min(max_size[i]-1, bounding_box[2*i+1]+amount) ) for i in range(3) ])

def get_bounding_box_from_origin_end(origin, end):
	return utils.flatten([ ( origin[i], end[i]) for i in range(3) ])

def get_index_in_image2(index, image1, image2):
	world = image1.TransformIndexToPhysicalPoint(ivec(index))
	return image2.TransformPhysicalPointToIndex(world)

def get_bounding_box_in_image2(bounding_box, image1, image2):
	origin = get_index_in_image2(get_bounding_box_origin(bounding_box), image1, image2)
	end = get_index_in_image2(get_bounding_box_end(bounding_box), image1, image2)
	return get_bounding_box_from_origin_end(origin, end)

def get_bounding_box_size_from_ends(origin, end):
	return [end[0] - origin[0] + 1, end[1] - origin[1] + 1, end[2] - origin[2] + 1]

def get_intersection(origin1, end1, origin2, end2):
	return np.maximum(origin1, origin2), np.minimum(end1, end2)

def ivec(vector):
	return [int(c) for c in vector]

def crop_from_ends(image, origin, end):
	size = get_bounding_box_size_from_ends(origin, end)
	filter = sitk.PasteImageFilter()

	image0 = sitk.Image(ivec(size), image.GetPixelID())
	image0.SetSpacing(image.GetSpacing())
	image0.SetDirection(image.GetDirection())
	image0.SetOrigin(image.TransformIndexToPhysicalPoint(ivec(origin)))

	intersection_origin, intersection_end = get_intersection(origin, end, [0, 0, 0], get_image_end(image))
	
	intersection_size = get_bounding_box_size_from_ends(intersection_origin, intersection_end)

	filter.SetSourceIndex(ivec(intersection_origin))
	filter.SetSourceSize(ivec(intersection_size))
	filter.SetDestinationIndex(ivec(np.subtract(intersection_origin, origin)))
	cropped = filter.Execute(image0, image)
	return cropped

def crop_from_ends_no_padding(image, origin, end):
	size = get_bounding_box_size_from_ends(origin, end)
	eif = sitk.RegionOfInterestImageFilter()
	eif.SetIndex(ivec(origin))
	eif.SetSize(ivec(size))
	return eif.Execute(image)

def crop_image_from_shape(image, mask=None, background_value=0):
	lsif = sitk.LabelShapeStatisticsImageFilter()
	lsif.Execute(mask if mask is not None else image != background_value)
	bounds = lsif.GetBoundingBox(1)
	size = image.GetSize()
	return sitk.Crop(image, [[bounds[i], size[i]-bounds[i]-bounds[i+3]] for i in range(3)])

def crop_image(image, mask=None, background_value=0):
	lsif = sitk.LabelStatisticsImageFilter()
	lsif.Execute(image, mask if mask is not None else image != background_value)
	bounds = lsif.GetBoundingBox(1)
	return sitk.Crop(image, [[bounds[2*i], bounds[2*i+1]] for i in range(3)])

def crop_from_ends_write(image_path, image, origin, end):
	cropped_mask = crop_from_ends(image, origin, end)
	image_cropped_path = utils.replace_path_suffix(image_path, '.nii.gz', '_cropped.nii.gz')
	sitk.WriteImage(cropped_mask, str(image_cropped_path))
	return image_cropped_path

def crop_anima(input_path, mask_path, output_path=None):
	print('Warning: problem if mask is greater than image')
	image = sitk.ReadImage(str(input_path))
	mask = sitk.ReadImage(str(mask_path))
	match, _, _ = utils.check_images_match(input_path, mask_path, image, mask)
	if not match:
		image_origin_mask = mask.TransformPhysicalPointToIndex(image.GetOrigin())
		image_end_mask = get_index_in_image2(get_image_end(image), image, mask)
		intersection_origin, intersection_end = get_intersection(image_origin_mask, image_end_mask, [0, 0, 0], get_image_end(mask))
		mask_cropped_path = crop_from_ends_write(mask_path, mask, intersection_origin, intersection_end)
		intersection_origin_image = get_index_in_image2(intersection_origin, mask, image)
		intersection_end_image = get_index_in_image2(intersection_end, mask, image)
		cropped_image_path = crop_from_ends_write(input_path, image, intersection_origin_image, intersection_end_image)
		utils.call([utils.anima / 'animaCropImage', '-i', cropped_image_path, '-m', mask_cropped_path, '-o', output_path or input_path])
		cropped_image_path.unlink()
		mask_cropped_path.unlink()
	else:
		utils.call([utils.anima / 'animaCropImage', '-i', input_path, '-m', mask_path, '-o', output_path or input_path])
	return

def crop_sitk(image, mask):
	ccifilter = sitk.ConnectedComponentImageFilter()
	ccifilter.FullyConnectedOn()
	lsifilter = sitk.LabelStatisticsImageFilter()
	mask_data = sitk.GetArrayFromImage(mask)
	mask_data = (np.abs(mask_data) > 1e-6).astype(np.uint8)
	mask_binary = sitk.GetImageFromArray(mask_data)
	mask_binary.CopyInformation(mask)
	labeled = ccifilter.Execute(mask_binary)
	lsifilter.Execute(labeled, labeled)
	n_components = ccifilter.GetObjectCount()

	global_bounding_box = None
	for i in range(1, n_components + 1):
		bounding_box = list(lsifilter.GetBoundingBox(i))
		if global_bounding_box is None:
			global_bounding_box = bounding_box
		for j in range(3):
			if bounding_box[2*j+0] < global_bounding_box[2*j+0]:
				global_bounding_box[2*j+0] = bounding_box[2*j+0]
			if bounding_box[2*j+1] > global_bounding_box[2*j+1]:
				global_bounding_box[2*j+1] = bounding_box[2*j+1]
	
	origin = get_index_in_image2(get_bounding_box_origin(global_bounding_box), mask, image)
	end = get_index_in_image2(get_bounding_box_end(global_bounding_box), mask, image)
	
	return crop_from_ends(image, origin, end)

def crop_sitk_from_path(input_path, mask_path, output_path=None):
	image = sitk.ReadImage(str(input_path))
	mask = sitk.ReadImage(str(mask_path))
	output = crop_sitk(image, mask)
	output_path = output_path or input_path
	sitk.WriteImage(output, str(output_path))
	return output_path

def crop(input_path, mask_path, output_path=None):
	return crop_sitk_from_path(input_path, mask_path, output_path or input_path)
	# output_path = output_path or input_path
	# utils.call([utils.anima / 'animaCropImage', '-i', input_path, '-m', mask_path, '-o', output_path])
	# return output_path
