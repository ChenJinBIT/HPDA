"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    print("dataset_name is :",dataset_name)
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    #print(datasetlib)
    print("dataset_filename",dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    print("target_dataset_name is,",target_dataset_name)
    for name, cls in datasetlib.__dict__.items():
        print("name is",name)
        #print("cls is ",cls)
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            print("-----------------------------------")
            dataset = cls
    print("dataset is ",dataset)

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    print("begin create_dataset")
    print("opt.batch_size is",opt.batch_size)
    print("opt.serial_batches is",opt.serial_batches)
    data_loader = CustomDatasetDataLoader(opt)
    #dataset = data_loader.load_data()
    S_number,T_number = data_loader.len_ST()
    return data_loader.dataloader,data_loader.dataloader_test,S_number,T_number
    #return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        print("hello")
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        print(dataset_class)

        self.dataset = dataset_class(opt,test=False)
        self.dataset_test = dataset_class(opt,test=True)
        #print("!!!!!!!!!!!!!!!!!")
        print("dataset [%s] was created" % type(self.dataset).__name__)
        print("opt.batch_size is ",opt.batch_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

        self.dataloader_test = torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def len_ST(self):
        """Return the number of data in the dataset"""
        return self.dataset.A_size,self.dataset.B_size

    '''
    def __iter__(self,test):
        """Return a batch of data"""
        if test:
            for i, data in enumerate(self.dataloader_test):
                yield data
        else:
            for i, data in enumerate(self.dataloader):
                #print("i",i)
                #print(self.opt.batch_size)
                #print(self.opt.max_dataset_size)
                if i * self.opt.batch_size >= self.opt.max_dataset_size:
                    break
                #print(data['A_paths'])
                #print(data['B_paths'])
                yield data
    '''
