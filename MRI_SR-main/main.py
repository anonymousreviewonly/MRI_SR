import argparse
import logging
import os
import torch

from models.mDCSRN.solver import mDCSRNTrainer
from models.mDCSRN_GAN.solver import mDCSRN_GANTrainer
from models.mDCSRN_WGAN.solver import mDCSRN_WGANTrainer
from models.volume_net.solver import volume_netTrainer

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results'
os.makedirs(results_folder, exist_ok=True)


def build_parser():
    """build parser for MRI Super Resolution"""
    parser = argparse.ArgumentParser(description='MRI Super Resolution')
    parser.add_argument('--model', type=str, default='mDCSRN',
                        choices=['mDCSRN', 'mDCSRN_GAN', 'mDCSRN_WGAN', 'volume_net'], help='model architectures')
    # frequently used settings
    parser.add_argument('--only-evaluate-cerebellum', action='store_true', default=False,
<<<<<<< HEAD
                        help='only evaluate cerebellum when training model using the whole brain')
=======
                        help='only evaluate model performance on the cerebellum when '
                             'the models are trained using the whole brain images')
>>>>>>> 17582ec65338df64d704c39b04cbc5ce6d0ca6bb
    parser.add_argument('--ssim-loss', action='store_true', default=False, help='add ssim loss')
    parser.add_argument('--test-mode', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='hcp', choices=['hcp', 'hcp_ce', 'hcp_all'])
    parser.add_argument('--hcp-all-balancing', type=str, default='balanced', choices=['small', 'balanced'])
    # when using hcp_all , set patch size and stride to 16, batch size to 256
    parser.add_argument('--scale-factor', type=int, default=4, help='specify which scale factor to use')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--validation-batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=32, help='input patch size')
    parser.add_argument('--stride', type=int, default=32, help='stride size when generating patches')
    parser.add_argument('--comment', type=str, default='run0',
                        help='comments to distinguish different runs')
    parser.add_argument('--external-image', action='store_true', default=False)
    parser.add_argument('--external-image-dir-hr', type=str, default='external_data/image2_ex.nii',
                        help='external_image_dir')
    parser.add_argument('--external-image-dir-lr', type=str, default='external_data/image4_ex.nii',
                        help='external_image_dir')
    parser.add_argument('--external-image-marker', type=str, default='marker',
                        help='external-image-marker')
    # default settings
    parser.add_argument('--margin', type=int, default=3, help='image margin when super-resolve images')
    parser.add_argument('--train-data-dir', type=str, default='../shared/hanzhi/train/')
    parser.add_argument('--validation-data-dir', type=str, default='../shared/hanzhi/validation/')
    parser.add_argument('--test-data-dir', type=str, default='../shared/hanzhi/test/')
    # hcp settings
    parser.add_argument('--n-split-dataset', type=int, default=8, help='split dataset into smaller batches')
    parser.add_argument('--n-split-per-batch', type=int, default=10, help='secondary split on each batch')
    parser.add_argument('--n-split-validation', type=int, default=5, help='split validation set into smaller batches')
    parser.add_argument('--n-split-test', type=int, default=5, help='split test set into smaller batches')
    return parser


def main():
    """overall workflow of MRI super resolution"""
    # build parser
    args = build_parser().parse_args()
    logger.info(f'Parser arguments are {args}')

    # check GPU
    logger.info(f"Found device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # build model
    if args.model == 'mDCSRN':
        model = mDCSRNTrainer(args)
    elif args.model == 'mDCSRN_GAN':
        model = mDCSRN_GANTrainer(args)
    elif args.model == 'mDCSRN_WGAN':
        model = mDCSRN_WGANTrainer(args)
    elif args.model == 'volume_net':
        model = volume_netTrainer(args)

    if not args.test_mode:
        model.run(args)
    else:
        model.super_resolve(args)


if __name__ == '__main__':
    main()
