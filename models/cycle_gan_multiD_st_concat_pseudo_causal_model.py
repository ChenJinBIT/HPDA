import torch
import torch.nn as nn
import numpy as np
import itertools
import os
from torch.nn import functional as F
#from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable

def calc_ent(x):
    ent=torch.tensor(0.0).cuda()
    for i in range(x.shape[0]):
        logp=torch.log(x[i])/torch.log(torch.tensor(2.0).cuda())
        #logp=np.log2(x[i])
        ent=ent-x[i]*logp
    return ent



class CycleGANMultiDstconcatpseudocausalModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        #if is_train:
        #    parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for cycle loss (A -> B -> A)')
        #    parser.add_argument('--lambda_B', type=float, default=100.0, help='weight for cycle loss (B -> A -> B)')
        #    parser.add_argument('--lambda_C', type=float, default=100.0, help='weight for classification loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        #self.print_flag=True
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.rgb_path = os.path.join(self.opt.dataroot,self.opt.rgb_path,'rgb.npy')
        #self.flow_path = os.path.join(self.opt.dataroot,self.opt.flow_path,'flow.npy')
        self.loss_names = ['D_A','D_A_T', 'G_A','G_A_T', 'cycle_A', 'D_B','G_B' ,'cycle_B','C_real_A','C_rec_A','C_fake_B']
        visual_names_A = ['fake_B_s','fake_B_t','fake_B']
        visual_names_B = ['real_B','rec_B','flow','rgb']
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        #self.visual_names = visual_names_A  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_A_T','G_B','G_B_T','D_A','D_A_T', 'D_B','C_A','C_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B','C_A','C_B','G_A_T','G_B_T']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_is+opt.context_dim, opt.output_vs,opt.context_dim, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #print("netG_A input_dim %d, output_dim %d"%(opt.input_is+opt.context_dim, opt.output_vs))
        self.netG_A_T = networks.define_G(opt.output_vs+opt.input_noise+opt.context_dim, opt.output_vt, opt.context_dim,opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #print("netG_A_T input_dim %d, output_dim %d"%(opt.output_vs+opt.input_noise+opt.context_dim, opt.output_vt))
        self.netG_B_T = networks.define_G(opt.output_vt+opt.context_dim, opt.output_vs, opt.context_dim,opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #print("netG_B_T input_dim %d, output_dim %d"%(opt.output_vt+opt.context_dim, opt.output_vs))


        self.netG_B = networks.define_G(opt.output_vs+opt.output_vt+opt.context_dim, opt.input_is,opt.context_dim, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #print("netG_B input_dim %d, output_dim %d"%(opt.output_vs+opt.output_vt, opt.input_is))

        self.netC_A = networks.define_C(opt.output_vt+opt.output_vs, opt.ncf,opt.netC,opt.class_num,opt.init_type, opt.init_gain, self.gpu_ids)   
        #print("netC_A input_dim %d, output_dim %d"%(opt.output_vt+opt.output_vs, opt.class_num))   
        self.netC_B = networks.define_C(opt.input_is, opt.ncf,opt.netC,opt.class_num,opt.init_type, opt.init_gain, self.gpu_ids)  

        if self.isTrain:
            parameter_list_netD_A = []
            parameter_list_netD_B = []
            parameter_list_netD_A_T = []
            #parameter_list_netD_B = []
            self.netD_A = []
            self.netD_B = []
            self.netD_A_T = []
            #self.netD_B_T = []
            for i in range(opt.class_num):
                netD_A = networks.define_D(opt.output_vs+opt.output_vt, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                #print("netD_A input_dim %d"%(opt.output_vs+opt.output_vt))
                netD_A_T = networks.define_D(opt.output_vt, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

                #print("netD_A_T input_dim %d"%(opt.output_vt))
                netD_B = networks.define_D(opt.input_is, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                #print("netD_B input_dim %d"%(opt.input_is))
                ##netD_B_T = networks.define_D(opt.output_vt, opt.ndf, opt.netD,
                ##                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                #self.parameter_list_netD_A.append(netD_A.parameters())
                #self.parameter_list_netD_B.append(netD_B.parameters())
                #itertools.chain(netD_A.parameters(),netD_B.parameters())
                parameter_list_netD_A.append({"params":netD_A.parameters()})
                parameter_list_netD_B.append({"params":netD_B.parameters()})
                parameter_list_netD_A_T.append({"params":netD_A_T.parameters()})
                #parameter_list_netD_B_T.append({"params":netD_B_T.parameters()})
                self.netD_A.append(netD_A)
                self.netD_A_T.append(netD_A_T)
                self.netD_B.append(netD_B)


        if self.isTrain:
            #self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            #self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            ##self.criterionCycle = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.MSELoss()
            #self.criterionCycle = torch.nn.MSELoss(reduce=True, size_average=True)
            ##self.criterionCycle = torch.nn.HingeEmbeddingLoss(size_average=True)
            #self.criterionIdt = torch.nn.L1Loss()
            self.criterioncls = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_A_T.parameters(),self.netG_B.parameters(),self.netG_B_T.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(parameter_list_netD_A+parameter_list_netD_B+parameter_list_netD_A_T, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_C = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),\
                self.netC_A.parameters(),self.netC_B.parameters(),self.netG_A_T.parameters(),self.netG_B_T.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_C)
    def set_pseudo(self,pseudo):
        self.frame_pseudo = pseudo
        self.print_flag = True
    def set_context(self,s_context,t_context):
        self.s_context = s_context
        self.t_context = t_context
    def generate_context(self,labels,context,context_t):
        context_new=torch.FloatTensor(np.zeros([labels.shape[0],context.shape[1]])).to(self.device)
        context_new_t=torch.FloatTensor(np.zeros([labels.shape[0],context_t.shape[1]])).to(self.device)
        #print("labels:",labels)
        for i in range(labels.shape[0]):
            if labels[i]==-1:
                continue
            else:
                context_new[i]=context[labels[i]]
                context_new_t[i]=context_t[labels[i]]
        return context_new,context_new_t
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_labels = input['A_label' if AtoB else 'B_label']
        self.image_labels = self.image_labels.cuda()
        #print("self.real_A.shape",self.real_A.shape)
        #print("image_paths",self.image_paths)
        #print("image_labels",self.image_labels.shape)   

        self.F = input['F'].to(self.device)
        self.F_paths = input['F_paths']
        self.F_labels = input['F_label'].cuda()
        #print("self.F.shape",self.F.shape)
        #print("F_paths",self.F_paths)
        #print("F_labels",self.F_labels)        

        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths_t = input['B_paths' if AtoB else 'A_paths']
        self.image_labels_t = input['B_label' if AtoB else 'A_label']
        self.image_labels_t = self.image_labels_t.cuda()


        #rgb_feature = torch.from_numpy(np.load(self.rgb_path)).to(self.device)
        #flow_feature = torch.from_numpy(np.load(self.flow_path)).to(self.device)
        rgb_0 = torch.from_numpy(np.zeros([1,1024])).to(self.device)
        rgb_0 = rgb_0.float()
        flow_0 = torch.from_numpy(np.zeros([1,1024])).to(self.device)
        flow_0=flow_0.float()
        #print("rgb_feature.type:",type(rgb_feature))
        #print("self.real_B.type:",type(self.real_B))
        self.flow,self.rgb = self.real_B.split(int(self.real_B.size(1)/2),1)

        #rgb_feature = rgb_feature.repeat(flow.size(0),1)
        #flow_feature = flow_feature.repeat(rgb.size(0),1)
        #self.causal_B_z = torch.cat((flow, rgb_feature),1)
        #self.causal_B_x = torch.cat((flow_feature, rgb),1)

        self.rgb_0 = rgb_0.repeat(self.flow.size(0),1)
        self.flow_0 = flow_0.repeat(self.rgb.size(0),1)
        self.causal_B_z_0 = torch.cat((self.flow, self.rgb_0),1)
        self.causal_B_x_0 = torch.cat((self.flow_0,self.rgb),1)    

        #print("flow.shape:",flow.shape)
        #print("rgb.shape:",rgb.shape)

        #print("flow_feature.shape:",flow_feature.shape)
        #print("rgb_feature.shape:",rgb_feature.shape)

        #print("image_paths_t",self.image_paths_t)
        #print("F_labels",self.image_labels_t)    

        ##if not self.isTrain:
            ##self.image_labels_t = input['B_label' if AtoB else 'A_label']
            ##self.image_labels_t = self.image_labels_t.cuda()

        #print("self.image_paths is %s, image_paths_t is %s"%(self.image_paths,self.image_paths_t))
        #print("self.images_labels is %d, image_labels_t is %d"%(self.image_labels.item(),self.image_labels_t.item()))
        #print(self.real_A.shape)
        #print(self.image_labels)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #print("s_context:",s_context.shape)
        #print("t_context:",t_context.shape)
        self.d_labels = []
        if self.frame_pseudo:
            #self.notrain_num = 0
            if self.print_flag: 
                self.print_flag = False               
                print("frame_pseudo",self.frame_pseudo)
            for i in range(self.F.shape[0]):
                #print(self.F[i].shape)
                temp = self.netC_A(self.F[i])
                #print("temp is",temp)
                score_real_F = nn.Softmax(dim=1)(self.netC_A(self.F[i])).detach()
                #print("after softmax is",score_real_F)

                score_real_F_sum = torch.sum(score_real_F,dim=0)
                #print("score_real_F_sum is",score_real_F_sum)
                pred_real_F = score_real_F_sum/score_real_F.shape[0]
                pred_value_real_F,pred_label_real_F =torch.max(pred_real_F,0)
                #print("final label is ",pred_label_real_F)
                if pred_value_real_F >=self.opt.c_threshold:
                    self.d_labels.append(pred_label_real_F)
                else:
                    #self.notrain_num =self.notrain_num+1
                    self.d_labels.append(int(-1)) 
            
        else:

            #self.train_num = 0
            temp = nn.Softmax(dim=1)(self.netC_B(self.real_B)).detach()

            pred_value_real_B,pred_label_real_B =torch.max(temp,1)
            if self.print_flag:
                self.print_flag = False
                print("frame_pseudo",self.frame_pseudo)
                print(self.netC_B(self.real_B))
                #print(temp)
                #print(pred_label_real_B)            
            for i in range(self.real_B.shape[0]):
                if pred_value_real_B[i] >=self.opt.c_threshold:
                    #self.train_num =self.train_num+1
                    self.d_labels.append(pred_label_real_B[i])
                else:
                    self.d_labels.append(int(-1))
        self.d_labels = torch.IntTensor(self.d_labels).to(self.device)
        #image to video direction and the corresponding cycle
        #temp = self.generate_context(self.image_labels,self.s_context)
        #print("self.image_labels:",self.image_labels)
        #print("self.generate_context(self.image_labels,self.s_context):",self.s_context)
        img_s_context,img_t_context = self.generate_context(self.image_labels,self.s_context,self.t_context)
        self.fake_B_s = self.netG_A(torch.cat((self.real_A,img_s_context),1))  # G_A(A)

        noise = Variable(torch.randn(self.fake_B_s.size())).cuda()
        self.fake_B_t = self.netG_A_T(torch.cat((self.fake_B_s,noise,img_t_context),1))
        self.fake_B = torch.cat((self.fake_B_t,self.fake_B_s),1)
        

        self.fakeB_ts = self.netG_B_T(torch.cat((self.fake_B_t,img_t_context),1))
        self.rec_A = self.netG_B(torch.cat((self.fake_B_s,self.fakeB_ts,img_s_context),1))   # G_B(G_A(A))
        vid_s_context,vid_t_context = self.generate_context(self.d_labels,self.s_context,self.t_context)
        #video to image direction and the corresponding cycle
        self.fakeA_ts = self.netG_B_T(torch.cat((self.flow,vid_t_context),1))
        self.fake_A = self.netG_B(torch.cat((self.rgb,self.fakeA_ts,vid_s_context),1))   # G_B(G_A(A))

        self.rec_B_s = self.netG_A(torch.cat((self.fake_A,vid_s_context),1))  # G_A(A)
        noise = Variable(torch.randn(self.rec_B_s.size())).cuda()
        self.rec_B_t = self.netG_A_T(torch.cat((self.rec_B_s,noise,vid_t_context),1))
        
        self.rec_B = torch.cat((self.rec_B_s,self.rec_B_t),1)   # G_A(G_B(B))

        #performing causal inference
        ##self.causal_rec_B_z_0 = torch.cat((self.rec_B_t, self.rgb_0),1)
        ##self.causal_rec_B_x_0 = torch.cat((self.flow_0, self.rec_B_s),1) 
        #self.fake_A_z = self.netG_B(self.causal_B_z)  # G_B(B_z)
        #self.rec_B_z = self.netG_A(self.fake_A_z) 

        #self.fake_A_x = self.netG_B(self.causal_B_x)  # G_B(B_z)
        #self.rec_B_x = self.netG_A(self.fake_A_x) 

 
            #print("there are %d/%d instances no join training"%(j,self.opt.batch_size))

    def get_correct_num(self,sample_model):
        correct = 0
        if sample_model=='real_A':
            predict_score_real_A = self.netC_A(self.real_A)
            _,predict_label_real_A =torch.max(predict_score_real_A,1)
            correct+=(predict_label_real_A==self.image_labels).sum()
            return correct.item(),predict_score_real_A.size(0)
        elif sample_model=='rec_A':
            predict_score_rec_A = self.netC_A(self.rec_A)
            _,predict_label_rec_A =torch.max(predict_score_rec_A,1)
            correct+=(predict_label_rec_A==self.image_labels).sum()
            return correct.item(),predict_score_rec_A.size(0)
        elif sample_model=='fake_B':
            predict_score_fake_B = self.netC_B(self.fake_B)
            _,predict_label_fake_B =torch.max(predict_score_fake_B,1)
            correct+=(predict_label_fake_B==self.image_labels).sum()
            return correct.item(),predict_score_fake_B.size(0)
        elif sample_model=='fake_A':
            predict_score_fake_A = self.netC_A(self.fake_A)
            #print("predict_score_fake_A.shape",predict_score_fake_A.shape)
            #print("predict_score_fake_A",predict_score_fake_A)
            _,predict_label_fake_A =torch.max(predict_score_fake_A,1)
            #print("predict_label_fake_A",predict_label_fake_A)
            correct+=(predict_label_fake_A==self.image_labels_t).sum()
            return correct.item(),predict_score_fake_A.size(0)
        elif sample_model=='real_B':
            predict_score_real_B,fc = self.netC_B(self.real_B,True)
            #print("fc.shape",fc.shape)
            _,predict_label_real_B =torch.max(predict_score_real_B,1)
            correct+=(predict_label_real_B==self.image_labels_t).sum()

            _,fc_t = self.netC_B(self.causal_B_z_0,True)
            #print("fc_t.shape",fc_t.shape)
            s_context = fc-fc_t

            _,fc_s = self.netC_B(self.causal_B_x_0,True)
            t_context = fc-fc_s

            #print("predict_label_real_B is %d, image_labels_t is %s"%(predict_label_real_B.item(),self.image_labels_t.item()))
            return correct.item(),predict_score_real_B.size(0),predict_label_real_B.detach(),s_context.detach(),t_context.detach()
        elif sample_model=='rec_B':
            predict_score_rec_B = self.netC_B(self.rec_B)
            _,predict_label_rec_B =torch.max(predict_score_rec_B,1)
            correct+=(predict_label_rec_B==self.image_labels_t).sum()
            return correct.item(),predict_score_rec_B.size(0)
        elif sample_model=='fake_A':
            predict_score_fake_A = self.netC_A(self.fake_A)
            #print("predict_score_fake_A.shape",predict_score_fake_A.shape)
            #print("predict_score_fake_A",predict_score_fake_A)
            _,predict_label_fake_A =torch.max(predict_score_fake_A,1)
            #print("predict_label_fake_A",predict_label_fake_A)
            correct+=(predict_label_fake_A==self.image_labels_t).sum()
            return correct.item(),predict_score_fake_A.size(0)
        elif sample_model=='F':
            total_num = 0
            for i in range(self.F_labels.shape[0]):
                #print("%d-th testing F"%(i))
                predict_score_F = self.netC_A(self.F[i])
                #print("self.F[i].shape",self.F[i].shape)
                #print("predict_score_F.shape",predict_score_F.shape)
                _,predict_label_F =torch.max(predict_score_F,1)
                #print("predict_label_F is ",predict_label_F)
                total_num = total_num +predict_score_F.size(0)
                correct+=(predict_label_F==self.F_labels[i]).sum()
            return correct.item(),total_num
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward(retain_graph=True)
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = 0
        self.loss_D_A_T = 0
        Flag = False
        for i in range(self.real_B.shape[0]):
            if self.d_labels[i]!=-1:
                Flag = True
                #self.loss_D_A = self.loss_D_A+self.backward_D_basic(self.netD_A[], self.real_B, self.fake_B)
                self.loss_D_A = self.loss_D_A+(self.criterionGAN(self.netD_A[self.image_labels[i]](self.fake_B[i].detach()), False)\
                    +self.criterionGAN(self.netD_A[self.d_labels[i]](self.real_B[i]), True))*0.5
                self.loss_D_A_T = self.loss_D_A_T+(self.criterionGAN(self.netD_A_T[self.image_labels[i]](self.fake_B_t[i].detach()), False)\
                    +self.criterionGAN(self.netD_A_T[self.d_labels[i]](self.flow[i]), True))*0.5
                self.loss_D = self.loss_D_A+self.loss_D_A_T
        if Flag:
            self.loss_D.backward(retain_graph=True)
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = 0
        Flag = False
        for i in range(self.real_A.shape[0]):
            #self.loss_D_B = self.loss_D_B+self.backward_D_basic(self.netD_B[self.image_labels[i]], self.real_A, self.fake_A)
            if self.d_labels[i]!=-1:
                Flag = True
                self.loss_D_B = self.loss_D_B+(self.criterionGAN(self.netD_B[self.d_labels[i]](self.fake_A[i].detach()), False)\
                    +self.criterionGAN(self.netD_B[self.image_labels[i]](self.real_A[i]), True))*0.5
        if Flag:
            self.loss_D_B.backward(retain_graph=True)
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        self.loss_G_A = 0
        self.loss_G_A_T = 0
        self.loss_G_B = 0
        # Identity loss
        for i in range(self.real_A.shape[0]):
            self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A[self.image_labels[i]](self.fake_B[i]), True)
            self.loss_G_A_T = self.loss_G_A_T + self.criterionGAN(self.netD_A_T[self.image_labels[i]](self.fake_B_t[i]), True)
            if self.d_labels[i]!= -1:
                self.loss_G_B = self.loss_G_B + self.criterionGAN(self.netD_B[self.d_labels[i]](self.fake_A[i]), True)
            #else:
            #    print("no join training")
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B +self.loss_C
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B +self.loss_G_A_T
        self.loss_G.backward(retain_graph=True)
    def backward_C(self):
        lambda_C = self.opt.lambda_C
        #lambda_idt = self.opt.lambda_identity
        self.loss_C_real_A = self.criterioncls(self.netC_A(self.real_A),self.image_labels)
        self.loss_C_rec_A = self.criterioncls(self.netC_A(self.rec_A),self.image_labels)
        self.loss_C_fake_B = self.criterioncls(self.netC_B(self.fake_B),self.image_labels)

        #self.loss_C_fake_B = 0
        self.loss_C = (self.loss_C_real_A+self.loss_C_rec_A+self.loss_C_fake_B)*lambda_C
        self.loss_C.backward(retain_graph=True)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        #print("s_context in optimize_parameters",s_context)
        self.forward()      # compute fake images and reconstruction images.
        # D_A and D_B
        self.set_requires_grad([self.netD_A,self.netD_A_T,self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights\
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B,self.netD_A_T], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad_G([self.netG_A, self.netG_B,self.netG_A_T, self.netG_B_T], True)
        self.set_requires_grad_G([self.netC_A], False)
        self.set_requires_grad_G([self.netC_B], False)

        self.optimizer_C.zero_grad()
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # C_A
        self.set_requires_grad([self.netD_A, self.netD_B,self.netD_A_T], False)
        self.set_requires_grad_G([self.netC_A], True)
        self.set_requires_grad_G([self.netC_B], True)
        self.optimizer_C.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_C() 
        self.optimizer_C.step() 