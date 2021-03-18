import os.path
from data.base_dataset import BaseDataset, get_transform
import numpy as np
#from data.feature_list import make_dataset
from PIL import Image
import scipy.io as sio 
#from skimage import io
import random

Feature_EXTENSIONS = [
    '.txt', '.mat'
]


def get_feature_vector_from_binary(feature_file):
    f = open(feature_file, 'rb')
    s = f.read()
    f.close()
    # num * channel * length * height * width
    (n, c, l, h, w) = array.array('i', s[:20])
    feature_vector = np.array(array.array('f', s[20:]))
    return feature_vector
def make_dataset(list_path):
    #print("list_path",list_path)
    feats=[]
    labels=[]
    frame_num=[]
    for vid in open(list_path):
        img=vid.split(' ')
        if len(img)==2:
            imgPath=img[0].replace('/media/ir4t','/media/mcislab3d/ir4t')
            label=int(img[1])
        else:
            imgPath=img[0].replace('/media/ir4t','/media/mcislab3d/ir4t')
            label=int(img[1])
            frame_num.append(int(img[2]))
        #print label
        feats.append(imgPath)
        labels.append(label)

    return feats, labels, frame_num

def load_feat(path):
    # print(path)
    if path.split('.')[-1]=="mat":
        data=sio.loadmat(path)
        return np.array(data['pool5'][0]).astype(np.float32)
    elif path.split('.')[-1]=="txt":
        fid=open(path)
        feat=fid.readline()
        if feat.count(',')==0:
            feat=feat.split()
        else:
            feat=feat.split(',')
        return np.array(feat).astype(np.float32)
    elif path.split('.')[-1]=="npy":
        return np.load(path).reshape(-1).astype(np.float32)
    else:
        raise(RuntimeError("Supported image extensions are: " +
                               ",".join(Feature_EXTENSIONS)))
    '''
    elif path.split('.')[-1]=="jpg":
        image = io.imread(path)

        return image

        #return transforms(image, input_height='--loadSize', input_width='--loadSize',
        #           resize_height=256, resize_width=256)
    else:
        return get_feature_vector_from_binary(path) 
    '''


class FeaturenoFDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, test):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #print("---------this is a feature dataset--------------")
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot,'list',opt.S_list)  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot,'list',opt.T_list)  # create a path '/path/to/data/trainB'

        self.test = test

        print(self.dir_A)
        print(self.dir_B)


        self.A_paths,self.A_labels,_ = make_dataset(self.dir_A)   # load images from '/path/to/data/trainA'
        self.B_paths,self.B_labels,self.B_frame_nums = make_dataset(self.dir_B)    # load images from '/path/to/data/trainB'
        #self.F_paths,self.F_labels = make_dataset(self.F_list)
        #self.B_paths,self.B_labels = make_dataset(self.dir_B_test)
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        #self.F_size = len(self.F_paths)

        btoA = self.opt.direction == 'BtoA'
        #input_nc = self.opt.output_nc if btoA else self.opt.input_nc     # get the number of channels of input image
        #output_nc = self.opt.input_nc if btoA else self.opt.output_nc    # get the number of channels of output image
        #self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        #self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        print("featurenoF")

    '''
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        #print("index is ",index)
        A_path= self.A_paths[index % self.A_size]
        A_label=self.A_labels[index % self.A_size]

        #A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        A_feature = load_feat(A_path)

        #print("index_B is ",index_B)
        B_path = self.B_paths[index_B]
        B_label=self.B_labels[index_B]
        B_feature = load_feat(B_path)
        #print("A_feature.shape",A_feature.shape)
        #print("B_feature.shape",B_feature.shape)
        #print("A_path",A_path)
        #print("B_path",B_path)        

        #print("A_label",A_label)
        #print("B_label",B_label) 

        item = {}
        item.update({'A':A_feature,
                    'A_paths':A_path,
                    'A_label':A_label
                    })
        item.update({'B':B_feature,
                    'B_paths':B_path,
                    'B_label':B_label
                    })        
        return item

        #A_img = Image.open(A_path).convert('RGB')
        #B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        #A = self.transform_A(A_img)
        #B = self.transform_B(B_img)

        #return {'A': A_feature, 'B': B_feature, 'A_label': A_label, 'B_label': B_label,'A_paths': A_path, 'B_paths': B_path,}
    '''
    def __getitem__(self, index):
        #print("------------------index %d is--------------------"%(index))
        A_path= self.A_paths[index % self.A_size]
        A_label=self.A_labels[index % self.A_size]

        if self.opt.class_consist:
            #print("----------class_consist is true-------------")
            index_B = random.randint(0, self.B_size - 1)
            B_path  = self.B_paths[index_B]
            B_label = self.B_labels[index_B]
            B_frame_num = self.B_frame_nums[index_B]
            while B_label!= A_label:
                index_B = random.randint(0, self.B_size - 1)
                B_path  = self.B_paths[index_B]
                B_label = self.B_labels[index_B]
                B_frame_num = self.B_frame_nums[index_B]
        elif self.test:
            #print("----------for testing-------------")
            index_B = index % self.B_size
            B_path  = self.B_paths[index_B]
            B_label = self.B_labels[index_B]
            #print("self.B_labels[index_B] is",self.B_labels[index_B])
            B_frame_num = self.B_frame_nums[index_B]
        else:
            #print("----------random target sample-------------")
            index_B = random.randint(0, self.B_size - 1)
            B_path  = self.B_paths[index_B]
            B_label = self.B_labels[index_B]
            B_frame_num = self.B_frame_nums[index_B]            
        A_feature = load_feat(A_path)
        B_feature = load_feat(B_path)

        item = {}
        item.update({'A':A_feature,
                    'A_paths':A_path,
                    'A_label':A_label
                    })
        item.update({'B':B_feature,
                    'B_paths':B_path,
                    'B_label':B_label
                    })         
        return item

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        print("A.size:",self.A_size)
        print("B_size:",self.B_size)
        return max(self.A_size, self.B_size)
        #return self.A_size,self.B_size
