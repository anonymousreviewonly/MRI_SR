import os, logging, gzip, zipfile, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


"""Extract and pre-process MR images locally"""


# -------------- HCP unzip and brain extraction
def extract_images_hcp(image_dir, temp_dir, target_dir):
    """
    extract HCP MR images from a specific directory
    HCP: https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release
    image resolution: (256, 320, 320)
    """
    # create directories for output images
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    file_type = ".zip"

    for file_name in os.listdir(path=image_dir):
        # unzip images to temp dir
        if file_name.endswith(file_type):
            subject_id = file_name[:-4].split('_')[0]
            with zipfile.ZipFile(image_dir + file_name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # extract MPR image to target dir
            try:
                with gzip.open(temp_dir + f'{subject_id}/' + 'unprocessed/3T/T1w_MPR1/' +
                               f'{subject_id}_3T_T1w_MPR1.nii.gz', 'rb') as f_in:
                    with open(target_dir + f'{subject_id}_3T_T1w_MPR1.nii', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                shutil.rmtree(temp_dir + f'{subject_id}/')
            except ValueError:
                logger.info(file_name)


def extract_brains_hcp(image_dir, target_dir):
    """Unzip brain-extracted images to the final directory"""
    # create directories for output images
    os.makedirs(target_dir, exist_ok=True)

    for file_name in os.listdir(path=image_dir):
        subject_id = file_name[:-4].split('_')[0]
        try:
            with gzip.open(image_dir + file_name, 'rb') as f_in:
                with open(target_dir + f'{subject_id}_3T_T1w_MPR1_ex.nii', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except ValueError:
            logger.info(file_name)


def split_dataset_hcp(image_dir):
    """split HCP dataset into train/validation/test set"""
    df = pd.DataFrame(columns=['id'])
    mpr_id = []
    # collect all image id
    for file_name in os.listdir(path=image_dir):
        if file_name.split('.')[-1] == 'nii':
            image_id = file_name.split('_')[0]
            mpr_id.append(image_id)
    df['id'] = mpr_id
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    assert df.duplicated(subset=['id']).sum() == 0

    # create train/val/test set
    df_train = df.iloc[:900, :]
    df_val = df.iloc[900: 1000, :]
    df_test = df.iloc[1000:, :]
    print(f'Train set: {len(df_train)}, validation set: {len(df_val)}, test set: {len(df_test)}')

    train_ids = df_train['id'].tolist()
    val_ids = df_val['id'].tolist()
    test_ids = df_test['id'].tolist()
    os.makedirs(image_dir + 'train', exist_ok=True)
    os.makedirs(image_dir + 'validation', exist_ok=True)
    os.makedirs(image_dir + 'test', exist_ok=True)

    # copy images to train/validation/test folder
    for file_name in os.listdir(path=image_dir):
        if file_name.split('.')[-1] == 'nii':
            image_id = file_name.split('_')[0]
            if image_id in train_ids:
                shutil.copy(image_dir + file_name, image_dir + 'train')
            elif image_id in val_ids:
                shutil.copy(image_dir + file_name, image_dir + 'validation')
            elif image_id in test_ids:
                shutil.copy(image_dir + file_name, image_dir + 'test')


# -------------- Generate LR images using FFT
def generate_lr_images_in_fft(pad_option='mask', participant_size=100, scale_factor=4,
                              image_dir='../../../Downloads/HCP_T1w_mpr_ex/', dataset_type='train/',
                              first_index=0, secondary_index=10):
    """generate lr images using fft and inverse fft"""
    # NOTE: we use float16 to save images to save disk space
    total_hr = []
    total_lr = []
    i = 0
    # NOTE: change participant size accordingly when generating different data sets
    for file_name in sorted(os.listdir(path=image_dir + dataset_type))[first_index * participant_size: (first_index+1) * participant_size]:
        i += 1
        logger.info(i)
        hr_image = get_img(os.path.join(image_dir, dataset_type), file_name)
        assert hr_image.shape == (256, 320, 320)

        # remove some useless black background to save spaces
        # hr_image = hr_image[:, :, 40: 280]

        # only generate hr images
        if pad_option == 'only_hr':
            total_hr.append(hr_image)
        # only generate lr images
        else:
            lr_image = fft_lr_hcp(image=hr_image, scale_factor=scale_factor, pad_option=pad_option)
            total_lr.append(lr_image)

        if i >= participant_size:
            break

    if pad_option == 'only_hr':
        depth, height, width = total_hr[0].shape

        total_hr = np.reshape(total_hr, (len(total_hr), depth, height, width)).astype(np.float16)
        logger.info(f'total_hr shape: {total_hr.shape}')
        # For training data, we use two-level index to save images.
        # For validation and test data, we do not use index.
        if dataset_type == 'train/':
            num_image_per_batch = int(participant_size / secondary_index)
            for k in range(secondary_index):
                np.save(f'{dataset_type[:-1]}_hcp_hr_{first_index}_{k}.npy',
                        total_hr[num_image_per_batch * k: num_image_per_batch * (k+1)])
        elif dataset_type in ['validation/', 'test/']:
            np.save(f'{dataset_type[:-1]}_hcp_hr.npy', total_hr)

    elif pad_option in ['mask', 'downscale_3d']:
        depth, height, width = total_lr[0].shape

        total_lr = np.reshape(total_lr, (len(total_lr), depth, height, width)).astype(np.float16)
        logger.info(f'total_lr shape: {total_lr.shape}')
        # For training data, we use two-level index to save images.
        # For validation and test data, we do not use index.
        if dataset_type == 'train/':
            num_image_per_batch = int(participant_size / secondary_index)
            for k in range(secondary_index):
                np.save(f'{dataset_type[:-1]}_hcp_lr_scale_{scale_factor}_{pad_option}_{first_index}_{k}.npy',
                        total_lr[num_image_per_batch * k: num_image_per_batch * (k+1)])
        elif dataset_type in ['validation/', 'test/']:
            np.save(f'{dataset_type[:-1]}_hcp_lr_scale_{scale_factor}_{pad_option}.npy', total_lr)


def get_img(image_dir, file_name):
    """return a normalized hr image"""
    image = nib.load(image_dir + file_name)
    image = np.array(image.dataobj).astype(np.float32)
    image /= np.max(image)
    return image


def fft_lr_hcp(image, scale_factor, pad_option):
    """
    image shape: (256, 320, 240)
    pad_option: mask, downscale_3d
    """
    # assert image.shape == (256, 320, 240)

    radius_1d = int(image.shape[0] // (scale_factor * 2))
    residual_1d = int(image.shape[0] // 2) - radius_1d
    radius_2d = int(image.shape[1] // (scale_factor * 2))
    residual_2d = int(image.shape[1] // 2) - radius_2d
    radius_3d = int(image.shape[2] // (scale_factor * 2))
    residual_3d = int(image.shape[2] // 2) - radius_3d
    assert radius_1d + residual_1d == int(image.shape[0] // 2)
    assert radius_2d + residual_2d == int(image.shape[1] // 2)
    assert radius_3d + residual_3d == int(image.shape[2] // 2)

    # get k-space data
    image_fft = np.fft.fftshift(np.fft.fftn(image))

    x_center = image_fft.shape[0] // 2
    y_center = image_fft.shape[1] // 2
    z_center = image_fft.shape[2] // 2
    
    if pad_option == 'downscale_3d':
        # truncate all three dimensions
        image_fft = image_fft[x_center - radius_1d: x_center + radius_1d,
                              y_center - radius_2d: y_center + radius_2d,
                              z_center - radius_3d: z_center + radius_3d]
    
    elif pad_option == 'mask':
        # mask out on height and width dimension: mDCSRN implementation
        image_fft = image_fft[:, y_center-radius_2d: y_center+radius_2d,
                              z_center-radius_3d: z_center+radius_3d]
        image_fft = np.pad(image_fft, ((0, 0), (residual_2d, residual_2d), (residual_3d, residual_3d)),
                           'constant', constant_values=0)

    # transform back to image space
    lr_image = abs(np.fft.ifftn(image_fft))
    lr_image /= lr_image.max()
    return lr_image


if __name__ == '__main__':
    # extract_images_hcp(image_dir='../../Downloads',
    #                    temp_dir='../../Downloads/temp',
    #                    target_dir='../../Downloads/HCP_T1w_mpr')

    # extract_brains_hcp(image_dir='../../../Downloads/HCP_T1w_mpr/brain_extracted/',
    #                    target_dir='../../../Downloads/HCP_T1w_mpr_ex/')

    # split_dataset_hcp(image_dir='../../../Downloads/HCP_T1w_mpr_ex/')
    # generate_lr_images_in_fft()
    pass
