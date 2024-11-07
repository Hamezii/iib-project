"""Module for getting MNIST datasets and generating data loaders."""

from torchvision import datasets
from torchvision.transforms import ToTensor


# ToTensor documentation:
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
# if the PIL Image belongs to one of the modes 
# (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray 
# has dtype = np.uint8

def get_train_data():
    return datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )

def get_test_data():
    return datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor(),
        download = True
    )