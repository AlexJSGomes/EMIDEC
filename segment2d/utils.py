import numpy as np
from skimage.morphology import remove_small_objects

def min_max_normalize(image) -> np.ndarray:
    """
    Normalizes the image using min-max scaling.

    This pre-processing function removes outlier voxels by first computing the 
    minimum and maximum values of the image and then scales the image so that its 
    values fall in the range [0, 1]. The result is returned as a np.float32 array.

    Parameters:
        image (np.ndarray): Input image (e.g., a 3D volume).

    Returns:
        np.ndarray: The normalized image with values between 0 and 1, as type np.float32.
    """
    low, high = image.min(), image.max()
    image = (image - low) / (high - low)
    return image.astype(np.float32)


def pad_background(image, dim2pad=(128, 128)) -> tuple[np.ndarray, tuple, tuple]:
    """
    Pads the background of the image by centering the region of interest.

    The function performs the following steps:
      1. Finds the indices of all non-zero elements in the image.
      2. Determines the minimum and maximum indices along each axis to define a bounding box.
      3. Crops the image to only include the non-zero values.
      4. If any dimension of the cropped image is larger than the desired size,
         performs a further centered crop.
      5. Calculates the required padding to center the cropped image within a 
         zero-filled array of the desired dimensions.
      6. Inserts the cropped image into the padded array at the appropriate indices.

    Parameters:
        image (np.ndarray): Input image.
        dim2pad (tuple, optional): Desired dimensions for the output volume.
            If the tuple has fewer than 3 elements, the last dimension is assumed to be image.shape[2].
            Default is (128, 128).

    Returns:
        tuple: A tuple containing:
            - padded_image (np.ndarray): The image after applying padding.
            - crop_index_new (tuple): Slices used to crop the original image.
            - padded_index (tuple): Slices indicating where the cropped image was inserted.
    """
    dim2pad = dim2pad if len(dim2pad) == 3 else (dim2pad[0], dim2pad[1], image.shape[2])
    nz = np.nonzero(image)
    min_indices = np.min(nz, axis=1)
    max_indices = np.max(nz, axis=1)
    crop_index = tuple(slice(imin, imax + 1) for imin, imax in zip(min_indices, max_indices))
    cropped_img = image[crop_index]
    padded_image = np.zeros(dim2pad)
    crop_index_new = crop_index
    if cropped_img.shape[0] > dim2pad[0]:
        cx, cx_pad = cropped_img.shape[0] // 2, dim2pad[0] // 2
        cropped_img = cropped_img[cx - cx_pad : cx + cx_pad, :, :]
        crop_index_new = (
            slice(crop_index[0].start + cx - cx_pad, crop_index[0].start + cx + cx_pad),
            crop_index[1],
            crop_index[2],
        )
    if cropped_img.shape[1] > dim2pad[1]:
        cy, cy_pad = cropped_img.shape[1] // 2, dim2pad[1] // 2
        cropped_img = cropped_img[:, cy - cy_pad : cy + cy_pad, :]
        crop_index_new = (
            crop_index_new[0],
            slice(crop_index[1].start + cy - cy_pad, crop_index[1].start + cy + cy_pad),
            crop_index_new[2],
        )
    if cropped_img.shape[2] > dim2pad[2]:
        cz, cz_pad = cropped_img.shape[2] // 2, dim2pad[2] // 2
        cropped_img = cropped_img[:, :, cz - cz_pad : cz + cz_pad]
        crop_index_new = (
            crop_index_new[0],
            crop_index_new[1],
            slice(crop_index[2].start + cz - cz_pad, crop_index[2].start + cz + cz_pad),
        )
    pad_widths = [(padded_image.shape[i] - cropped_img.shape[i]) // 2 for i in range(3)]
    padded_index = tuple(slice(pad_widths[i], pad_widths[i] + cropped_img.shape[i]) for i in range(3))
    padded_image[padded_index] = cropped_img
    return padded_image, crop_index_new, padded_index


def pad_background_with_index(image, crop_index_new, padded_index, dim2pad=(128, 128)) -> np.ndarray:
    """
    Applies padding to the image using pre-calculated indices.

    This function assumes that the cropping indices (crop_index_new) and the 
    padded insertion indices (padded_index) have been determined. It extracts the 
    region of interest from the input image using crop_index_new and places it 
    into a zero-filled array of size dim2pad at the location specified by padded_index.

    Parameters:
        image (np.ndarray): Input image.
        crop_index_new (tuple): Slices defining the region to extract from the image.
        padded_index (tuple): Slices where the extracted region should be placed in the padded volume.
        dim2pad (tuple, optional): Dimensions of the output volume. If not 3 elements,
                                   the last dimension is taken from image.shape[2].
                                   Default is (128, 128).

    Returns:
        np.ndarray: The padded image with the extracted region inserted.
    """
    dim2pad = dim2pad if len(dim2pad) == 3 else (dim2pad[0], dim2pad[1], image.shape[2])
    padded_image = np.zeros(dim2pad)
    crop_image = image[crop_index_new]
    padded_image[padded_index] = crop_image
    return padded_image


def invert_padding(original_shape, padded_image, crop_index, padded_index) -> np.ndarray:
    """
    Reverses the padding operation to recover the original image region.

    This function extracts the padded region from the padded image using padded_index,
    and then inserts it into a zero-filled array of the original shape at the location
    specified by crop_index, effectively reversing the padding process.

    Parameters:
        original_shape (tuple): The original shape of the image before padding.
        padded_image (np.ndarray): The image after padding.
        crop_index (tuple): Slices indicating the region in the original image that was padded.
        padded_index (tuple): Slices where the padded region was inserted in padded_image.

    Returns:
        np.ndarray: The reconstructed image with the padded region restored to its original location.
    """
    cropped_img = padded_image[padded_index]
    inverted_image = np.zeros(original_shape)
    inverted_image[crop_index] = cropped_img
    return inverted_image


def remove_small_elements(segmentation_mask, min_size_remove=3) -> np.ndarray:
    """
    Removes small objects from a segmentation mask.

    The function first converts the segmentation mask into a binary mask (values > 0 become True),
    then uses skimage's remove_small_objects function to eliminate connected components
    smaller than the specified minimum size. Finally, the cleaned binary mask is applied 
    to the original segmentation mask to preserve the original intensity values.

    Parameters:
        segmentation_mask (np.ndarray): The segmentation mask with numerical values.
        min_size_remove (int, optional): Minimum size (in pixels/voxels) of connected components to keep.
                                         Connected components smaller than this size are removed.
                                         Default is 3.

    Returns:
        np.ndarray: The cleaned segmentation mask with small objects removed.
    """
    pred_mask = segmentation_mask > 0
    mask = remove_small_objects(pred_mask, min_size=min_size_remove)
    clean_segmentation = segmentation_mask * mask
    return clean_segmentation
