import logging
import gc

import torch
import numpy as np
import random
import nibabel as nib

from utils.common_utils import TrainDataset, ValidationDataset, TestDataset, calculate_padding_size

results_folder = 'model_ckpt_results'
logger = logging.getLogger(__name__)


# ----Build Dataset ---- #
def build_data_loader(args, *idx):
    """make train or validation data loader"""
    dataset = build_dataset(args, *idx)

    # data-loader configs
    if len(idx) == 2:
        # train args
        kwargs = {'batch_size': args.train_batch_size, 'shuffle': True}
    elif len(idx) == 1:
        # validation args
        kwargs = {'batch_size': args.validation_batch_size, 'shuffle': False}
    # common configs
    if torch.cuda.is_available():
        cuda_kwargs = {'pin_memory': True}
        kwargs.update(cuda_kwargs)

    # initialize data loader
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return data_loader


def build_dataset(args, *idx):
    """build train or validation dataset"""
    if args.dataset in ['hcp', 'hcp_ce', 'hcp_all']:
        dataset = build_dataset_hcp(args, *idx)
    return dataset


def build_dataset_hcp(args, *idx):
    """load train or validation hcp MRI data"""
    dataset_type = 'train' if len(idx) == 2 else 'validation'
    data_dir = args.train_data_dir if len(idx) == 2 else args.validation_data_dir
    idx_name = f'{idx[0]}_{idx[1]}' if len(idx) == 2 else f'{idx[0]}'

    # load whole brain data-set
    if args.dataset == 'hcp':
        hr_images = torch.from_numpy(np.load(data_dir + f'{dataset_type}_hcp_hr_{idx_name}.npy'))
        lr_images = torch.from_numpy(np.load(
            data_dir + f'{dataset_type}_hcp_lr_scale_{args.scale_factor}_zero_out_{idx_name}.npy'))
    # load cerebellum data-set
    elif args.dataset == 'hcp_ce':
        hr_images = torch.from_numpy(np.load(data_dir + f'{dataset_type}_hcp_hr_{idx_name}_ce.npy'))
        lr_images = torch.from_numpy(np.load(
                data_dir + f'{dataset_type}_hcp_lr_scale_{args.scale_factor}_zero_out_{idx_name}_ce.npy'))
    # load all hcp data (whole brain and cerebellum)
    elif args.dataset == 'hcp_all':
        hr_images_whole_brain = torch.from_numpy(np.load(data_dir + f'{dataset_type}_hcp_hr_{idx_name}.npy'))
        lr_images_whole_brain = torch.from_numpy(np.load(
            data_dir + f'{dataset_type}_hcp_lr_scale_{args.scale_factor}_zero_out_{idx_name}.npy'))

        hr_images_cerebellum = torch.from_numpy(np.load(data_dir + f'{dataset_type}_hcp_hr_{idx_name}_ce.npy'))
        lr_images_cerebellum = torch.from_numpy(np.load(
                data_dir + f'{dataset_type}_hcp_lr_scale_{args.scale_factor}_zero_out_{idx_name}_ce.npy'))

    # train with image patches
    if args.dataset in ['hcp', 'hcp_ce']:
        hr_images_patch, lr_images_patch = get_image_patches(
            hr_images=hr_images, lr_images=lr_images, patch_size=args.patch_size,
            stride=args.stride, is_training=True, args=args)

        del hr_images, lr_images
        gc.collect()

    elif args.dataset in ['hcp_all']:
        hr_images_patch_whole_brain, lr_images_patch_whole_brain = \
            get_image_patches(hr_images=hr_images_whole_brain, lr_images=lr_images_whole_brain,
                              patch_size=args.patch_size, stride=args.stride, is_training=True, args=args)
        del hr_images_whole_brain, lr_images_whole_brain
        gc.collect()

        hr_images_patch_cerebellum, lr_images_patch_cerebellum = \
            get_image_patches(hr_images=hr_images_cerebellum, lr_images=lr_images_cerebellum,
                              patch_size=args.patch_size, stride=args.stride, is_training=True, args=args)
        del hr_images_cerebellum, lr_images_cerebellum
        gc.collect()

        # when using HCP_all to train the model, we need to find a way to balance the number of patches of two sets
        rnd_idx = list(range(len(hr_images_patch_whole_brain)))
        random.shuffle(rnd_idx)
        # Based on the size of cerebellum dataset
        if args.hcp_all_balancing == 'small':
            rnd_idx = rnd_idx[:len(hr_images_patch_cerebellum)]

            # select a subset of whole brain patches
            hr_images_patch_whole_brain = torch.stack([hr_images_patch_whole_brain[i] for i in rnd_idx])
            lr_images_patch_whole_brain = torch.stack([lr_images_patch_whole_brain[i] for i in rnd_idx])

            assert len(hr_images_patch_whole_brain) == len(hr_images_patch_cerebellum)

            # combine two types of image
            hr_images_patch = np.concatenate([hr_images_patch_whole_brain, hr_images_patch_cerebellum])
            lr_images_patch = np.concatenate([lr_images_patch_whole_brain, lr_images_patch_cerebellum])

            del hr_images_patch_whole_brain, hr_images_patch_cerebellum, \
                lr_images_patch_whole_brain, lr_images_patch_cerebellum
            gc.collect()
        # Based on the size of the whole brain dataset
        elif args.hcp_all_balancing == 'balanced':
            rnd_idx = rnd_idx[:len(hr_images_patch_whole_brain) // 2]

            hr_images_patch_whole_brain = torch.stack([hr_images_patch_whole_brain[i] for i in rnd_idx])
            lr_images_patch_whole_brain = torch.stack([lr_images_patch_whole_brain[i] for i in rnd_idx])

            multiplier = len(hr_images_patch_whole_brain) // len(hr_images_patch_cerebellum)
            hr_images_patch_cerebellum = torch.cat([hr_images_patch_cerebellum for i in range(multiplier)])
            lr_images_patch_cerebellum = torch.cat([lr_images_patch_cerebellum for i in range(multiplier)])

            # combine two types of image
            hr_images_patch = np.concatenate([hr_images_patch_whole_brain, hr_images_patch_cerebellum])
            lr_images_patch = np.concatenate([lr_images_patch_whole_brain, lr_images_patch_cerebellum])

            del hr_images_patch_whole_brain, hr_images_patch_cerebellum, \
                lr_images_patch_whole_brain, lr_images_patch_cerebellum
            gc.collect()

    # Build Pytorch Data-set
    logger.info(f'{dataset_type} images patch shape: {lr_images_patch.shape}, '
                f'{dataset_type} labels patch shape: {hr_images_patch.shape}')
    if dataset_type == 'train':
        dataset = TrainDataset(images=lr_images_patch, labels=hr_images_patch)
    elif dataset_type == 'validation':
        dataset = ValidationDataset(images=lr_images_patch, labels=hr_images_patch)

    del lr_images_patch, hr_images_patch
    gc.collect()

    return dataset


# ---- build test set ---- #
def build_test_data_loader(args, i):
    """make test data loader"""
    dataset_test, test_hr_images, test_lr_images = build_test_dataset(args, i)
    # data-loader configurations
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    if torch.cuda.is_available():
        cuda_kwargs = {'pin_memory': True}
        test_kwargs.update(cuda_kwargs)

    # initialize loader
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    return test_loader, test_hr_images, test_lr_images


def build_test_dataset(args, i):
    """build test dataset"""
    if args.dataset in ['hcp', 'hcp_ce', 'hcp_all']:
        dataset_test, test_hr_images, test_lr_images = build_test_dataset_hcp(args, i)
    return dataset_test, test_hr_images, test_lr_images


def build_test_dataset_hcp(args, i):
    """load hcp test MRI data (50 images)"""
    # NOTE: when we train on HCP and HCP_all, we could evaluate whole brain and
    #  cerebellum (using --only-evaluate-cerebellum option);
    #  when we train on HCP_ce, we could only evaluate cerebellum.

    # load whole brain data-set
    if args.dataset in ['hcp', 'hcp_all']:
        test_hr_images = torch.from_numpy(np.load(args.test_data_dir + f'test_hcp_hr_{i}.npy')).float()
        test_lr_images = torch.from_numpy(np.load(
            args.test_data_dir + f'test_hcp_lr_scale_{args.scale_factor}_zero_out_{i}.npy')).float()
    # load cerebellum data-set
    elif args.dataset == 'hcp_ce':
        test_hr_images = torch.from_numpy(np.load(args.test_data_dir + f'test_hcp_hr_{i}_ce.npy')).float()
        test_lr_images = torch.from_numpy(np.load(
                args.test_data_dir + f'test_hcp_lr_scale_{args.scale_factor}_zero_out_{i}_ce.npy')).float()

    # train with image patches
    test_hr_images_patch = get_image_patches_solo(images=test_hr_images, patch_size=args.patch_size, args=args)
    logger.info(1)
    del test_hr_images
    gc.collect()
    test_lr_images_patch = get_image_patches_solo(images=test_lr_images, patch_size=args.patch_size, args=args)
    del test_lr_images
    gc.collect()

    logger.info(f'Test hr image patches shape: {test_hr_images_patch.shape}, '
                f'test lr image patches shape: {test_lr_images_patch.shape}.')

    # Build Pytorch Data-set for test set
    dataset_test = TestDataset(images=test_lr_images_patch, labels=test_hr_images_patch)
    del test_lr_images_patch, test_hr_images_patch
    gc.collect()
    exit(0)

    return dataset_test, test_hr_images, test_lr_images


def get_external_img(image_dir):
    """return a normalized external image"""
    image = nib.load(image_dir)
    image = np.array(image.dataobj).astype(np.float32)
    image /= np.max(image)
    return image


def test_external_image(args):
    """test external image in nii format"""
    external_test_image_hr = torch.from_numpy(np.expand_dims(get_external_img(image_dir=args.external_image_dir_hr), 0))
    external_test_image_lr = torch.from_numpy(np.expand_dims(get_external_img(image_dir=args.external_image_dir_lr), 0))

    # we do not need to get hr image patch actually
    _, test_lr_images_patch = \
        get_image_patches(hr_images=external_test_image_lr, lr_images=external_test_image_lr,
                          patch_size=args.patch_size, stride=args.patch_size, is_training=False, args=args)
    logger.info(f'Test lr image patches shape: {test_lr_images_patch.shape}.')

    dataset_test = TestDataset(images=test_lr_images_patch, labels=test_lr_images_patch)
    del test_lr_images_patch
    gc.collect()

    # data-loader configurations
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    if torch.cuda.is_available():
        cuda_kwargs = {'pin_memory': True}
        test_kwargs.update(cuda_kwargs)

    # initialize loader
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    return test_loader, external_test_image_hr, external_test_image_lr


# ---- image patches ---- #
def get_image_patches(hr_images, lr_images, patch_size, stride, is_training, args):
    """split images into patches"""
    if is_training:
        # By default, we use random patching in training
        start_idx = np.random.randint(low=0, high=stride, size=1)[0]
        hr_image_patches = hr_images[:, start_idx:, start_idx:, start_idx:].unfold(1, patch_size, stride).\
            unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        lr_image_patches = lr_images[:, start_idx:, start_idx:, start_idx:].unfold(1, patch_size, stride).\
            unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        hr_image_patches = hr_image_patches.contiguous().view(-1, patch_size, patch_size, patch_size)
        lr_image_patches = lr_image_patches.contiguous().view(-1, patch_size, patch_size, patch_size)
    else:
        if args.dataset in ['hcp', 'hcp_all']:
            image_size = (256, 320, 240)
            padding = calculate_padding_size(image_size, args.margin, patch_size)
        elif args.dataset == 'hcp_ce':
            image_size = (160, 90, 100)
            padding = calculate_padding_size(image_size, args.margin, patch_size)

        stride = patch_size - 2 * args.margin

        lr_images_padded = torch.zeros([lr_images.shape[0],
                                        lr_images.shape[1] + 2 * padding[0],
                                        lr_images.shape[2] + 2 * padding[1],
                                        lr_images.shape[3] + 2 * padding[2]])
        hr_images_padded = torch.zeros([hr_images.shape[0],
                                        hr_images.shape[1] + 2 * padding[0],
                                        hr_images.shape[2] + 2 * padding[1],
                                        hr_images.shape[3] + 2 * padding[2]])
        lr_images_padded[:, padding[0]: lr_images.shape[1] + padding[0],
                         padding[1]: lr_images.shape[2] + padding[1],
                         padding[2]: lr_images.shape[3] + padding[2]] = lr_images
        hr_images_padded[:, padding[0]: hr_images.shape[1] + padding[0],
                         padding[1]: hr_images.shape[2] + padding[1],
                         padding[2]: hr_images.shape[3] + padding[2]] = hr_images
        lr_image_patches = lr_images_padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride).\
            unfold(3, patch_size, stride)
        hr_image_patches = hr_images_padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride).\
            unfold(3, patch_size, stride)
        lr_image_patches = lr_image_patches.contiguous().view(-1, patch_size, patch_size, patch_size)
        hr_image_patches = hr_image_patches.contiguous().view(-1, patch_size, patch_size, patch_size)

    return hr_image_patches, lr_image_patches


def get_image_patches_solo(images, patch_size, args):
    """split images into patches"""
    if args.dataset in ['hcp', 'hcp_all']:
        image_size = (256, 320, 240)
        padding = calculate_padding_size(image_size, args.margin, patch_size)
    elif args.dataset == 'hcp_ce':
        image_size = (160, 90, 100)
        padding = calculate_padding_size(image_size, args.margin, patch_size)

    stride = patch_size - 2 * args.margin

    images_padded = torch.zeros([images.shape[0],
                                 images.shape[1] + 2 * padding[0],
                                 images.shape[2] + 2 * padding[1],
                                 images.shape[3] + 2 * padding[2]])
    images_padded[:, padding[0]: images.shape[1] + padding[0],
                     padding[1]: images.shape[2] + padding[1],
                     padding[2]: images.shape[3] + padding[2]] = images
    image_patches = images_padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride).\
        unfold(3, patch_size, stride)
    image_patches = image_patches.contiguous().view(-1, patch_size, patch_size, patch_size)

    return image_patches


def depatching(patches, args):
    """merge patches into images"""
    if args.dataset in ['hcp', 'hcp_all']:
        image_size = (256, 320, 240)
        padding = calculate_padding_size(image_size, args.margin, args.patch_size)
    elif args.dataset == 'hcp_ce':
        image_size = (160, 90, 100)
        padding = calculate_padding_size(image_size, args.margin, args.patch_size)

    patch_size = patches.shape[-1]
    batch_size = patches.shape[0]
    tmp = patches.view(batch_size, -1, patch_size, patch_size, patch_size)
    real_tmp = tmp[:, :, args.margin:-args.margin, args.margin:-args.margin, args.margin:-args.margin]
    patch_size_cropped = real_tmp.shape[-1]
    merged_image_size = [image_size[0] + 2 * (padding[0] - args.margin), image_size[1] + 2 * (padding[1] - args.margin),
                         image_size[2] + 2 * (padding[2] - args.margin)]
    merged_image = torch.zeros(batch_size, merged_image_size[0], merged_image_size[1], merged_image_size[2])
    nz = int(merged_image_size[0] / patch_size_cropped)
    nx = int(merged_image_size[1] / patch_size_cropped)
    ny = int(merged_image_size[2] / patch_size_cropped)
    real_tmp = real_tmp.view(batch_size, nz, -1, patch_size_cropped, patch_size_cropped, patch_size_cropped)
    real_tmp = real_tmp.view(batch_size, nz, nx, -1, patch_size_cropped, patch_size_cropped, patch_size_cropped)

    for i in range(nz):
        for j in range(nx):
            for k in range(ny):
                merged_image[:, patch_size_cropped * i:patch_size_cropped * (i + 1),
                             patch_size_cropped * j:patch_size_cropped * (j + 1),
                             patch_size_cropped * k:patch_size_cropped * (k + 1)] = real_tmp[:, i, j, k, :, :, :]
    images = merged_image[:, (padding[0] - args.margin):-(padding[0] - args.margin),
                          (padding[1] - args.margin):-(padding[1] - args.margin),
                          (padding[2] - args.margin):-(padding[2] - args.margin)]
    return images
