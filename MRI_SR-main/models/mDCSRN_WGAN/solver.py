from collections import OrderedDict
import logging
import os
import gc
from pytorch_msssim import ssim

import torch
import torch.backends.cudnn as cudnn

from models.mDCSRN_WGAN.model import Generator, Discriminator
from utils.common_utils import RunManager, calculate_psnr, calculate_ssim
from utils.other_utils import analyze_sr_image_quality, stats2csv
from dataset.dataset import build_data_loader, build_test_data_loader, test_external_image


results_folder = 'model_ckpt_results'
logger = logging.getLogger(__name__)


class mDCSRN_WGANTrainer:
    def __init__(self, args):
        super().__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.netG = None
        self.netD = None
        self.lr = args.lr
        self.epochs = args.epochs
        self.epoch_pretrain = 1
        self.criterionG = None
        self.criterionD = None
        self.optimizerG = None
        self.optimizerD = None
        self.scheduler = None
        self.scale_factor = args.scale_factor
        self.train_loader = None
        self.validation_loader = None
        self.model_config = f'{args.model}_{args.dataset}_scale_{args.scale_factor}_{args.comment}'
        self.m = RunManager(args)
        self.cube_size = args.patch_size
        self.lmda = 0.001
        self.n_critic = 5

    def wasserstein_loss(self, d_fake, d_real=torch.Tensor([0.0])):
        """
        This function calculate the Earth Mover (EM) distance for the wasserstein loss.
        (Input) D_fake: the Discriminator's output digit for SR images.
        (Input) D_real: the Discriminator's output digit for HR images.
        (Output) G_loss: the Generator's loss only for WGAN's part.
        (Output) D_loss: the Discriminator's loss.
        """
        d_real = d_real.cuda(self.device)
        d_loss = - (torch.mean(d_real) - torch.mean(d_fake))
        g_loss = - torch.mean(d_fake)
        return g_loss, d_loss

    def run(self, args):
        """overall training process"""
        best_loss = 1
        # initialize the model
        self.build_model()
        for epoch in range(1, self.epoch_pretrain + 1):
            self.m.begin_epoch()
            # training part. Nested for loop to save memory usage
            for i in range(args.n_split_dataset):
                for j in range(args.n_split_per_batch):
                    logger.info(f'{i}-{j}')
                    self.train_loader = build_data_loader(args, i, j)
                    self.m.begin_run_train(self.train_loader)
                    self.pretrain_g(args)
                    self.pretrain_d()
                    gc.collect()
            # validation part. Nested for loop to save memory usage
            for k in range(args.n_split_validation):
                logger.info(f'validation-{k}')
                self.validation_loader = build_data_loader(args, k)
                self.m.begin_run_validation(self.validation_loader)
                self.validate()
                gc.collect()

            self.m.end_epoch()
            self.m.display_epoch_results()

            assert epoch == self.m.epoch_num_count

            # save the model with the best validation loss
            if self.m.epoch_stats['validation_loss'] < best_loss:
                logger.info(f'Lower validation loss found at epoch {self.m.epoch_num_count}')
                best_loss = self.m.epoch_stats['validation_loss']
                self.save_model()

        logger.info('Generator pre-train finished!')

        for epoch in range(1, self.epochs + 1):
            self.m.begin_epoch()
            # training part. Nested for loop to save memory usage
            for i in range(args.n_split_dataset):
                for j in range(args.n_split_per_batch):
                    logger.info(f'{i}-{j}')
                    self.train_loader = build_data_loader(args, i, j)
                    self.m.begin_run_train(self.train_loader)
                    self.train(args)
                    gc.collect()
            # validation part
            for k in range(args.n_split_validation):
                logger.info(f'validation-{k}')
                self.validation_loader = build_data_loader(args, k)
                self.m.begin_run_validation(self.validation_loader)
                self.validate()
                gc.collect()

            self.m.end_epoch()
            self.m.display_epoch_results()

            assert epoch == self.m.epoch_num_count - self.epoch_pretrain

            # save the model with the best validation loss
            if self.m.epoch_stats['validation_loss'] < best_loss:
                logger.info(f'Lower validation loss found at epoch {self.m.epoch_num_count}')
                best_loss = self.m.epoch_stats['validation_loss']
                self.save_model()
        self.m.end_run()
        self.m.save(os.path.join(results_folder, f'{self.model_config}_runtime_stats'))
        logger.info('Model finished!')

    def build_model(self):
        """initialize model"""
        self.netG = Generator().to(self.device)
        self.netD = Discriminator(cube_size=self.cube_size).to(self.device)
        self.netG.weight_init()
        self.netD.weight_init()
        self.criterionG = torch.nn.L1Loss()
        self.criterionD = torch.nn.BCELoss()

        if self.CUDA:
            cudnn.benchmark = True
            self.criterionG.cuda()
            self.criterionD.cuda()

        logger.info(f'Available GPUs: {torch.cuda.device_count()}')
        if torch.cuda.device_count() > 1:
            self.netG = torch.nn.DataParallel(self.netG, list(range(torch.cuda.device_count())))
            self.netD = torch.nn.DataParallel(self.netD, list(range(torch.cuda.device_count())))

        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr)
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.lr)

    def load_trained_model(self):
        """load pre-trained model for testing"""
        self.netG = Generator().to(self.device)

        logger.info(f'Available GPUs: {torch.cuda.device_count()}')
        if torch.cuda.device_count() > 1:
            self.netG = torch.nn.DataParallel(self.netG, list(range(torch.cuda.device_count())))
            self.netG.load_state_dict(
                state_dict=torch.load(os.path.join(results_folder, f'{self.model_config}_G_Best_Model.pt')))
            logger.info('trained model loaded!')
        else:
            # re-load weights trained on multiple GPUs
            saved_weight = torch.load(os.path.join(results_folder, f'{self.model_config}_G_Best_Model.pt'))
            keys = saved_weight.keys()
            values = saved_weight.values()

            new_keys = []
            for key in keys:
                # remove the 'module.'
                new_key = key[7:]
                new_keys.append(new_key)
            self.netG.load_state_dict(state_dict=OrderedDict(list(zip(new_keys, values))))
            logger.info('trained model loaded!')

    def save_model(self):
        torch.save(self.netG.state_dict(), os.path.join(results_folder, f"{self.model_config}_G_Best_Model.pt"))
        torch.save(self.netD.state_dict(), os.path.join(results_folder, f"{self.model_config}_D_Best_Model.pt"))

    def pretrain_g(self, args):
        """pre-train generator"""
        for p in self.netG.parameters():
            p.requires_grad = True
        for p in self.netD.parameters():
            p.requires_grad = False

        self.netG.train()
        for batch_num, (lr_patches, hr_patches) in enumerate(self.train_loader):

            # add channel dimension. The preprocessed images do not have the channel dimension
            lr_patches = torch.unsqueeze(lr_patches, 1)
            hr_patches = torch.unsqueeze(hr_patches, 1)

            # zero out gradient
            self.optimizerG.zero_grad()

            lr_patches, hr_patches = lr_patches.float().to(self.device), hr_patches.float().to(self.device)
            sr_patches = self.netG(lr_patches)

            # update parameters
            loss = self.criterionG(sr_patches, hr_patches)
            # add ssim loss if needed
            if args.ssim_loss:
                ssim_loss = 1 - ssim(sr_patches, hr_patches, data_range=1.0, nonnegative_ssim=True)
                loss += 0.1 * ssim_loss

            loss.backward()
            self.optimizerG.step()

    def pretrain_d(self):
        """pre-train discriminator"""
        for p in self.netG.parameters():
            p.requires_grad = False
        for p in self.netD.parameters():
            p.requires_grad = True

        self.netD.train()
        for batch_num, (lr_patches, hr_patches) in enumerate(self.train_loader):

            # add channel dimension. The preprocessed images do not have the channel dimension
            lr_patches = torch.unsqueeze(lr_patches, 1)
            hr_patches = torch.unsqueeze(hr_patches, 1)

            # zero out gradient
            self.optimizerD.zero_grad()

            # setup noise
            real_label = torch.ones(lr_patches.size(0)).to(self.device)
            fake_label = torch.zeros(lr_patches.size(0)).to(self.device)

            lr_patches, hr_patches = lr_patches.float().to(self.device), hr_patches.float().to(self.device)

            d_real = torch.sigmoid(self.netD(hr_patches))
            assert d_real.shape == real_label.shape
            d_real_loss = self.criterionD(d_real, real_label)

            d_fake = torch.sigmoid(self.netD(self.netG(lr_patches)))
            assert d_fake.shape == fake_label.shape
            d_fake_loss = self.criterionD(d_fake, fake_label)

            d_total = (d_real_loss + d_fake_loss) / 2
            d_total.backward()
            self.optimizerD.step()

    def train(self, args):
        """training"""
        for p in self.netG.parameters():
            p.requires_grad = True
        for p in self.netD.parameters():
            p.requires_grad = True

        self.netG.train()
        self.netD.train()
        for batch_num, (lr_patches, hr_patches) in enumerate(self.train_loader):

            # add channel dimension. The preprocessed images do not have the channel dimension
            lr_patches = torch.unsqueeze(lr_patches, 1)
            hr_patches = torch.unsqueeze(hr_patches, 1)

            lr_patches, hr_patches = lr_patches.float().to(self.device), hr_patches.float().to(self.device)
            sr_patches = self.netG(lr_patches)

            # Train Discriminator using W-loss
            self.optimizerD.zero_grad()
            d_real = self.netD(hr_patches)
            d_fake = self.netD(sr_patches)
            d_loss = - (torch.mean(d_real) - torch.mean(d_fake))
            d_loss.backward()
            self.optimizerD.step()

            for p in self.netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            if batch_num % self.n_critic == 0:
                self.optimizerG.zero_grad()
                # Train generator
                sr_patches = self.netG(lr_patches)
                d_fake = self.netD(sr_patches)
                L1_loss = self.criterionG(sr_patches, hr_patches)
                g_loss = -torch.mean(d_fake)

                # add ssim loss if needed
                if args.ssim_loss:
                    ssim_loss = 1 - ssim(sr_patches, hr_patches, data_range=1.0, nonnegative_ssim=True)
                    L1_loss += 0.1 * ssim_loss

                loss = L1_loss + self.lmda * g_loss
                loss.backward()
                self.optimizerG.step()

    def validate(self):
        """validation"""
        validation_psnr_list = []
        validation_ssim_list = []

        self.netG.eval()
        with torch.no_grad():
            for batch_num, (lr_patches, hr_patches) in enumerate(self.validation_loader):

                # add channel dimension
                lr_patches = torch.unsqueeze(lr_patches, 1)
                hr_patches = torch.unsqueeze(hr_patches, 1)

                lr_patches, hr_patches = lr_patches.float().to(self.device), hr_patches.float().to(self.device)
                sr_patches = self.netG(lr_patches)
                assert sr_patches.shape == hr_patches.shape

                # track validation loss
                loss = self.criterionG(sr_patches, hr_patches)
                self.m.track_validation_loss(loss=loss)

                # calculate stats from this batch
                psnr_val, _ = calculate_psnr(y_true=hr_patches, y_test=sr_patches.detach())
                ssim_val, _ = calculate_ssim(y_true=hr_patches, y_test=sr_patches.detach())

                validation_psnr_list.append(psnr_val)
                validation_ssim_list.append(ssim_val)

            self.m.collect_validation_psnr(psnr=sum(validation_psnr_list) / len(validation_psnr_list))
            self.m.collect_validation_ssim(ssim=sum(validation_ssim_list) / len(validation_ssim_list))

    def super_resolve(self, args):
        """super resolve lr images"""
        # load trained model
        self.load_trained_model()

        ori_psnr_list, sr_psnr_list, ori_ssim_list, sr_ssim_list, ori_nrmse_list, sr_nrmse_list \
            = [], [], [], [], [], []

        if not args.external_image:
            # HCP test image
            for i in range(args.n_split_test):
                logger.info(f'test-{i}')
                # load test set
                test_loader, test_hr_images, test_lr_images = build_test_data_loader(args, i)

                # super resolve images/patches. Here we use sr_output to include both patches and images
                sr_outputs = []
                with torch.no_grad():
                    for lr_input, _ in test_loader:
                        sr_output = self.netG(torch.unsqueeze(lr_input, 1).float().to(self.device))
                        sr_outputs.append(sr_output)

                del test_loader
                gc.collect()

                # merge super_resolved patches into images and perform analysis
                stats = analyze_sr_image_quality(args, test_hr_images, test_lr_images, sr_outputs, i)

                del test_hr_images, test_lr_images
                gc.collect()

                stats2csv(stats, ori_psnr_list, sr_psnr_list, ori_ssim_list, sr_ssim_list,
                          ori_nrmse_list, sr_nrmse_list, model_config=self.model_config, args=args)

        else:
            logger.info(f'Test external image')
            # load test set
            test_loader, test_hr_images, test_lr_images = test_external_image(args=args)

            # super resolve images/patches. Here we use sr_output to include both patches and images
            sr_outputs = []
            with torch.no_grad():
                for lr_input, _ in test_loader:
                    sr_output = self.netG(torch.unsqueeze(lr_input, 1).float().to(self.device))
                    sr_outputs.append(sr_output)

            del test_loader
            gc.collect()

            # merge super_resolved patches into images and perform analysis
            stats = analyze_sr_image_quality(args, test_hr_images, test_lr_images, sr_outputs)

            del test_hr_images, test_lr_images
            gc.collect()

            # stats2csv(stats, ori_psnr_list, sr_psnr_list, ori_ssim_list, sr_ssim_list,
            #           ori_nrmse_list, sr_nrmse_list, model_config=self.model_config, args=args)
