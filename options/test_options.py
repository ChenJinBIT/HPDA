from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--result_path', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--c_threshold',type=float, default=0.7, help='the threshold of curriculum learning')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--S_list', dest='S_list', default='Stanford12_resnet50_list.txt', help='img_list are saved here')########
        parser.add_argument('--T_list', dest='T_list', default='ucf_i3d_feature_frame_num_list.txt', help='vid_list are saved here')
        parser.add_argument('--F_path', dest='F_path', default='ucf_resnet50_frame_feature', help='vid_list are saved here')
        parser.add_argument('--rgb_path', dest='rgb_path', default='ucf_i3d_meanfeature', help='vid_list are saved here')
        parser.add_argument('--flow_path', dest='flow_path', default='ucf_i3d_meanfeature', help='vid_list are saved here')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        print(parser)
        self.isTrain = False
        return parser
