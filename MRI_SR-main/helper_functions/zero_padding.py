import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def load_img(image_dir):
    """Load and normalize the MR image"""
    image = nib.load(image_dir)
    image = np.array(image.dataobj).astype(np.float32)
    image /= np.max(image)
    return image


def fft_zip_ifft(image, target_resolution):
    """
    Apply FFT to the numpy MR image, use ZIP to increase its resolution and
    apply inverse FFT to transform back to image space
    :param image: original image in numpy format
    :param target_resolution: target resolution needed
    :return: Image of target resolution
    """
    # get some stats for padding
    depth, height, width = image.shape
    padded_depth, padded_height, padded_width = target_resolution
    depth_pad = (padded_depth - depth) // 2
    height_pad = (padded_height - height) // 2
    width_pad = (padded_width - width) // 2
    # transform from image space to k-space
    image_fft = np.fft.fftshift(np.fft.fftn(image))

    # plt.imshow(abs(image_fft[image_fft.shape[0] // 2, :, :]), cmap='gray')
    # plt.savefig('original_image_k_space.jpg')

    # Zero-pad the original image to target resolution
    padded_image_fft = np.pad(image_fft, ((depth_pad, depth_pad), (height_pad, height_pad), (width_pad, width_pad)),
                              'constant', constant_values=[(0, 0), (0, 0), (0, 0)])
    assert padded_image_fft.shape == target_resolution

    # plt.imshow(abs(padded_image_fft[padded_image_fft.shape[0] // 2, :, :]), cmap='gray')
    # plt.savefig('padded_image_k_space.jpg')

    # transform back from k-space to image space and normalize
    image = abs(np.fft.ifftn(image_fft))
    image /= image.max()
    return image


def get_cerebellum(image_dir):
    # we want to load image3_ex_resize.nii
    image = load_img(image_dir)
    image = image[:, :, 40: 280]
    plt.imshow(image[128, :, :], cmap='gray')
    plt.savefig('0509test.jpg')
    ce_image = image[40:208, 50: 150, 50: 150]
    plt.imshow(ce_image[:, 75, :], cmap='gray')
    plt.savefig('0509test1.jpg')
    new_image_ce = nib.Nifti1Image(ce_image, affine=np.eye(4))
    nib.save(new_image_ce, 'image3_ex_ce_resize.nii')


def zero_interpolation_padding(image_dir, target_resolution=(256, 320, 320),
                               target_name='image_ZIP_resized.nii'):
    """
    Given an MR image of a specific resolution, use ZIP method to increase its resolution
    :param image_dir: directory of the nii image
    :param target_resolution: target resolution
    :param target_name: target file name
    :return: Resized nii image in target name
    """
    image = load_img(image_dir)

    # plt.imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    # plt.savefig('original_image_before_ZIP.jpg')

    image = fft_zip_ifft(image=image, target_resolution=target_resolution)

    # plt.imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    # plt.savefig('resized_image_after_ZIP.jpg')

    # save numpy array to nii format
    nii_image = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(nii_image, target_name)


if __name__ == '__main__':
    zero_interpolation_padding(image_dir='../../../Image/image3_ex.nii', target_resolution=(256, 320, 320),
                               target_name='ZIP_image.nii')
