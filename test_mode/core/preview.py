import SimpleITK as sitk
import numpy as np
import core.crop
import itertools

plane_names = ['axial', 'coronal', 'sagittal']

def make_isotropic(
	image,
	interpolator=sitk.sitkLinear,
	spacing=None,
	default_value=0,
	standardize_axes=False,
):
	"""
	Many file formats (e.g. jpg, png,...) expect the pixels to be isotropic, same
	spacing for all axes. Saving non-isotropic data in these formats will result in
	distorted images. This function makes an image isotropic via resampling, if needed.
	Args:
		image (SimpleITK.Image): Input image.
		interpolator: By default the function uses a linear interpolator. For
					  label images one should use the sitkNearestNeighbor interpolator
					  so as not to introduce non-existant labels.
		spacing (float): Desired spacing. If none given then use the smallest spacing from
						 the original image.
		default_value (image.GetPixelID): Desired pixel value for resampled points that fall
										  outside the original image (e.g. HU value for air, -1000,
										  when image is CT).
		standardize_axes (bool): If the original image axes were not the standard ones, i.e. non
								 identity cosine matrix, we may want to resample it to have standard
								 axes. To do that, set this paramter to True.
	Returns:
		SimpleITK.Image with isotropic spacing which occupies the same region in space as
		the input image.
	"""
	original_spacing = image.GetSpacing()
	# Image is already isotropic, just return a copy.
	if all(spc == original_spacing[0] for spc in original_spacing):
		return sitk.Image(image)
	# Make image isotropic via resampling.
	original_size = image.GetSize()
	if spacing is None:
		spacing = min(original_spacing)
	new_spacing = [spacing] * image.GetDimension()
	new_size = [
		int(round(osz * ospc / spacing))
		for osz, ospc in zip(original_size, original_spacing)
	]
	new_direction = image.GetDirection()
	new_origin = image.GetOrigin()
	# Only need to standardize axes if user requested and the original
	# axes were not standard.
	if standardize_axes and not np.array_equal(
		np.array(new_direction), np.identity(image.GetDimension()).ravel()
	):
		new_direction = np.identity(image.GetDimension()).ravel()
		# Compute bounding box for the original, non standard axes image.
		boundary_points = []
		for boundary_index in list(itertools.product(*zip([0, 0, 0], image.GetSize()))):
			boundary_points.append(image.TransformIndexToPhysicalPoint(boundary_index))
		max_coords = np.max(boundary_points, axis=0)
		min_coords = np.min(boundary_points, axis=0)
		new_origin = min_coords
		new_size = (((max_coords - min_coords) / spacing).round().astype(int)).tolist()
	return sitk.Resample(
		image,
		new_size,
		sitk.Transform(),
		interpolator,
		new_origin,
		new_spacing,
		new_direction,
		default_value,
		image.GetPixelID(),
	)

def get_bounding_box_in_resampled(image, resampled_image, bounding_box):
	return [resampled_image.TransformPhysicalPointToIndex(image.TransformIndexToPhysicalPoint(bbpt)) for bbpt in bounding_box]

def convert_image_to_uint8(image):
	mmif = sitk.MinimumMaximumImageFilter()
	mmif.Execute(image)
	return sitk.Cast(sitk.IntensityWindowing(image, windowMinimum=mmif.GetMinimum(), windowMaximum=mmif.GetMaximum(), outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)

red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]

def get_slice(image, bounding_box, center, i, is_segmentation, scale=1):
	slices = [center[i] if i==si else slice(bounding_box[2*si], bounding_box[2*si+1]) for si in range(3)]
	slice_image = image[slices[0], slices[1], slices[2]]
	return slice_image if scale == 1 else sitk.Resample(slice_image, [s*scale for s in slice_image.GetSize()], sitk.Transform(), sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear, slice_image.GetOrigin(), [s/scale for s in slice_image.GetSpacing()], slice_image.GetDirection(), 0.0, slice_image.GetPixelID())

def is_segmentation(image_name):
	return 'segmentation' in image_name or 'ground_truth' in image_name

def make_transparent_background(rgb):
	r = sitk.VectorIndexSelectionCast(rgb, 0)
	g = sitk.VectorIndexSelectionCast(rgb, 1)
	b = sitk.VectorIndexSelectionCast(rgb, 2)
	return sitk.Compose(r, g, b, 255*(r+g+b>0))

def save_preview(image, bounding_box, center, path, is_segmentation, segmentation_contour=None):
	paths = []
	for i in range(3):
		image_path = path / f'{plane_names[i]}.png'
		paths.append(str(image_path))

		image_slice = get_slice(image, bounding_box, center, i, is_segmentation, 2)
		if is_segmentation and segmentation_contour: # if is_segmentation and segmentation_contour: create a transparent image with the segmentation contour only
			image_slice[:,:] = 0
		image_slice = make_transparent_background(sitk.LabelToRGB(sitk.Cast(image_slice, sitk.sitkInt32))) if is_segmentation and not segmentation_contour else convert_image_to_uint8(image_slice)

		if segmentation_contour:
			segmentation_slice = get_slice(segmentation_contour, bounding_box, center, i, True, 2)
			segmentation_slice = sitk.Cast(segmentation_slice, sitk.sitkLabelUInt8)
			image_slice.CopyInformation(segmentation_slice)
			image_slice = sitk.LabelMapContourOverlay(segmentation_slice, image_slice, opacity=1, contourThickness=[1, 1], dilationRadius=[1, 1], colormap=red + green + blue)
			if is_segmentation:
				image_slice = make_transparent_background(image_slice)
		image_path.parent.mkdir(exist_ok=True, parents=True)
		sitk.WriteImage(image_slice[:, ::-1], str(image_path))
	return paths

def create_lesion_preview(image, resampled_image, lesion_bounding_box, path, is_segmentation, margin=25, resampled_segmentation=None):
	bounding_box_in_resampled = core.crop.get_bounding_box_in_image2(lesion_bounding_box, image, resampled_image)
	center = [(bounding_box_in_resampled[2*i] + bounding_box_in_resampled[2*i+1]) // 2 for i in range(3)]
	expanded_bounding_box = core.crop.get_expanded_bounding_box(resampled_image.GetSize(), bounding_box_in_resampled, margin)
	return save_preview(resampled_image, expanded_bounding_box, center, path, is_segmentation, resampled_segmentation)

def get_resampled_images(image_paths):
	resampled_images = {}
	for image_name, image_path in image_paths.items():
		if not image_path.exists(): continue
		is_seg = is_segmentation(image_name)
		image = sitk.ReadImage(str(image_path), sitk.sitkUInt8 if is_seg else sitk.sitkFloat64)
		resampled_images[image_name] = {
			'image': image, 
			'resampled': make_isotropic(image, interpolator=sitk.sitkNearestNeighbor if is_seg else sitk.sitkLinear)
		}
	return resampled_images

def create_lesion_previews(resampled_images, bounding_box, path, margin=25, contours=False):
	images = []
	ground_truth = resampled_images['ground_truth']['resampled'] if 'ground_truth' in resampled_images else None
	segmentation = resampled_images['segmentation']['resampled'] if 'segmentation' in resampled_images else None
	for image_name, image_object in resampled_images.items():
		is_seg = is_segmentation(image_name)
		slices = create_lesion_preview(image_object['image'], image_object['resampled'], bounding_box, path / image_name, is_seg, margin=margin, resampled_segmentation=segmentation or ground_truth if contours and is_seg else None)
		images.append({'name': image_name, 'slices': slices})
	return images

def render_html(env, patients):
	template = env.get_template('lesion_previews_template.html')
	return template.render(patients=patients)