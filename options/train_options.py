from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #print(parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--S_list', dest='S_list', default='stanford-image-resnet50.txt', help='img_list are saved here')########
        parser.add_argument('--T_list', dest='T_list', default='trainlist01.txt', help='vid_list are saved here')
        parser.add_argument('--F_path', dest='F_path', default='ucf_resnet50_frame_feature', help='vid_list are saved here')
        parser.add_argument('--rgb_path', dest='rgb_path', default='ucf_i3d_meanfeature', help='vid_list are saved here')
        parser.add_argument('--flow_path', dest='flow_path', default='ucf_i3d_meanfeature', help='vid_list are saved here')
        parser.add_argument('--result_path', dest='result_path', default='ucf_i3d_meanfeature', help='vid_list are saved here')
        parser.add_argument('--c_threshold',type=float, default=0.7, help='the threshold of curriculum learning')
        parser.add_argument('--lambda_A',type=float, default=100, help='the threshold of curriculum learning')
        parser.add_argument('--lambda_B',type=float, default=100, help='the threshold of curriculum learning')
        parser.add_argument('--lambda_C',type=float, default=100, help='the threshold of curriculum learning')        
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        parser.add_argument('--frame_pseudo', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--test_accuracy_freq', type=int, default=50, help='frequency of test')
        parser.add_argument('--test_accuracy_freq_S', type=int, default=100, help='frequency of test')
        # training parameters
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. lsgan is MSEloss.vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser
