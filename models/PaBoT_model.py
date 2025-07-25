import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util
import os
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import normalize as l2_normalize
from util.image_pool import ImagePool
from util.util import rgb_to_grayscale
from util.SobelOperator import SobelOperator
import torchvision.transforms.functional as TF


class PaBoTModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')        
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for rec loss')        
        parser.add_argument('--lambda_idt', type=float, default=5.0, help='weight for idt loss')        
        parser.add_argument('--lambda_bone', type=float, default=5.0, help='weight for bone loss')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for kl loss')
        parser.add_argument('--lambda_path', type=float, default=0.1, help='weight for path loss)')
        parser.add_argument('--lambda_a', type=float, default=1.0, help='controls the importance of the attention mechanism')
        parser.add_argument('--path_layers', type=str, default='0,3,6,10,14', help='compute NCE loss on which layers')
        parser.add_argument('--style_dim', type=int, default=8, help='Dimensionality of the style vector')
        parser.add_argument('--path_interval_min', type=float, default=0.05, help='Minimum interval step size along the latent path during image translation.')
        parser.add_argument('--path_interval_max', type=float, default=0.10, help='Maximum interval step size along the latent path during image translation.')
        parser.add_argument('--noise_std', type=float, default=1.0, help='tandard deviation of the Gaussian noise added during encoding')
        parser.add_argument('--tag', type=str, default='debug', help='')

        
        parser.set_defaults(no_html=True, pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        if opt.phase != 'test':
            model_id = '%s' % opt.tag
            model_id += '/'+os.path.basename(opt.dataroot.strip('/')) + '_%s' % opt.direction
            model_id += '/lam%s_layers%s_dim%d_rec%d_idt%s_pool%d_noise%s_kl%s' % \
            (opt.lambda_path, opt.path_layers, opt.style_dim, opt.lambda_rec, opt.lambda_idt, opt.pool_size, opt.noise_std, opt.lambda_kl)
            parser.set_defaults(name=model_id)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G_rec', 'G_idt', 'G_kl', 'G_path','G_bone', 'd1', 'd2']
        self.visual_names = ['real_A', 'fake_B', 'real_B','G_Bone','Sobel_Bone']

        self.path_layers = [int(i) for i in self.opt.path_layers.split(',')]
        for l in self.path_layers:
            self.loss_names += ['energy_%d' % l]

        if self.isTrain:
            self.model_names = ['G', 'D', 'G_bone']
        else:  # during test time, load G and G_bone
            self.model_names = ['G', 'G_bone']

        # define networks (both generator and discriminator) 

        # G(x) 
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        
        # G_{bone}(x)  
        self.netG_bone = networks.define_G(opt.input_nc, 1, opt.ngf, "bone_G", opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        # Sobel(y)
        self.SobelOperator = SobelOperator().cuda()
        
        # theta = 0、theta = 1 source domain A and target domain B
        self.d_A = torch.zeros([1]).to(self.device)
        self.d_B = torch.ones([1]).to(self.device)
        

        if self.isTrain:
            self.epochs_num = self.opt.n_epochs + self.opt.n_epochs_decay


        if self.isTrain:
            # D(y) discriminator
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_G_bone = torch.optim.Adam(self.netG_bone.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))  
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G_bone)


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G、G_{bone}
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_G_bone.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        self.optimizer_G.step()
        self.optimizer_G_bone.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real = torch.cat([self.real_A, self.real_B], dim=0)
        latents = self.netG(real, mode='encode')
        if self.isTrain and self.opt.noise_std > 0:
            noise = torch.ones_like(latents).normal_(mean=0, std=self.opt.noise_std)
            self.mu = latents
            latents = latents + noise
        self.latent_A, self.latent_B = latents.chunk(2, dim=0)
        ds = torch.cat([self.d_A, self.d_B, self.d_B], 0).unsqueeze(-1)
        latents = torch.cat([self.latent_A, self.latent_A, self.latent_B], 0)
        images = self.netG((latents, ds), mode='decode')
        self.rec_A, self.fake_B, self.idt_B = images.chunk(3, dim=0)
        
        # 一、Generate bone structure from the source domain image  
        self.G_Bone = self.netG_bone(self.real_A)
        # 二、Extract bone contours from the generated target domain image using the Sobel operator
        self.Sobel_Bone = self.SobelOperator(self.fake_B)


    @torch.no_grad()
    def single_forward(self):
        latent = self.netG(self.real_A, mode='encode')
        out = self.netG((latent, self.d_B), mode='decode')


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B_pool.query(self.fake_B.detach())
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""

        self.loss_G_bone = self.criterionL1(self.G_Bone,self.Sobel_Bone).mean()
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True).mean()
        self.loss_G_rec = self.criterionIdt(self.rec_A, self.real_A).mean()
        self.loss_G_idt = self.criterionIdt(self.idt_B, self.real_B).mean()
        self.loss_d1 = torch.tensor(0.)
        self.loss_d2 = torch.tensor(0.)

        for l in self.path_layers:
            setattr(self, 'loss_energy_%d' % l, 0)

        if self.opt.noise_std>0:
            self.loss_G_kl = torch.pow(self.mu, 2).mean()
        else:
            self.loss_G_kl = 0

        self.loss_n_dots = 0

        if self.opt.lambda_path > 0:
            self.loss_G_path = self.compute_path_losses()
        else:
            self.loss_G_path = torch.tensor(0.)

        self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN +\
                      self.opt.lambda_rec * self.loss_G_rec + \
                      self.opt.lambda_idt * self.loss_G_idt + \
                      self.opt.lambda_kl * self.loss_G_kl + \
                      self.opt.lambda_path * self.loss_G_path + \
                      self.opt.lambda_bone * self.loss_G_bone
        return self.loss_G
    
    def compute_path_losses(self):
        d1_center = torch.ones(len(self.latent_A)).to(self.device).uniform_(0, 1)
        interval = torch.ones(len(self.latent_A)).to(self.device).uniform_(self.opt.path_interval_min,
                                                                           self.opt.path_interval_max)
        d1 = (d1_center + interval).clamp(0, 1)
        d2 = (d1_center - interval).clamp(0, 1)
        latents = torch.cat([self.latent_A, self.latent_A], 0)
        ds = torch.cat([d1, d2], 0).unsqueeze(-1)
        features = self.netG((latents, ds), layers=self.path_layers, mode='decode_and_extract')

        # dummy for outputs only
        self.loss_d1 = torch.mean(d1)
        self.loss_d2 = torch.mean(d2)

        loss_path = 0
        cam = self.G_Bone.detach()  
        cam = TF.gaussian_blur(cam, kernel_size=(5,5), sigma=(1.0,1.0))
    
        for id, feats in enumerate(features):
            x_d1, x_d2 = torch.chunk(feats, 2, dim=0)
            cams = torch.nn.functional.interpolate(cam, size=x_d1.shape[2:], mode='bilinear')
            jacobian = (x_d1 - x_d2) / (torch.maximum(d1 - d2, torch.ones_like(d1)*0.1))
            energy = ((1 + self.opt.lambda_a * cams) * (jacobian ** 2)).mean()
            setattr(self, 'loss_energy_%d' % self.path_layers[id], energy.item())
            loss_path += energy
        loss_path = loss_path / len(features)
        return loss_path

  
    # Path visualization for Fig. 6 with θ = 0.2
    @torch.no_grad()
    def interpolation(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'AtoB':
            x = x_a
        else:
            x = x_b
        interps = []
        
        for i in range(min(x.size(0),2)):
            h_a = self.netG(x[i].unsqueeze(0), mode='encode')
            d = 0
            local_interps = []
            #local_interps.append(x[i].unsqueeze(0))

            while d <= 1:
                d_t = torch.tensor([d]).to(x_a.device).unsqueeze(-1)
                local_interps.append(self.netG((h_a, d_t), mode='decode'))
                d += 0.2
            local_interps = torch.cat(local_interps, 0)
            interps.append(local_interps)
        self.netG.train()
        return interps

    # Only infer/generate the target domain image
    @torch.no_grad()
    def translate(self, x):
        self.netG.eval()
        h = self.netG(x, mode='encode')
        out = self.netG((h, self.d_B), mode='decode')
        self.netG.train()
        return out

    @torch.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'BtoA':
            x_a, x_b = x_b, x_a
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a = self.netG(x_a[i].unsqueeze(0), mode='encode')
            h_b = self.netG(x_b[i].unsqueeze(0), mode='encode')
            x_a_recon.append(self.netG((h_a, self.d_A), mode='decode'))
            x_b_recon.append(self.netG((h_b, self.d_B), mode='decode'))
            x_ab.append(self.netG((h_a, self.d_B), mode='decode'))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ab = torch.cat(x_ab)
        self.netG.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon
