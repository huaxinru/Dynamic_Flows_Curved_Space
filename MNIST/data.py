import os
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
import numpy as np
import torch


# TODO create a class for each dataset EXCEPT MNIST (this is already built into pytorch)
# If the dataset comes pre-split into train/test we should write a separate class for each.

# For example the street view house number might look something like this...
# The cropped version of the dataset is in a weird *.mat format, see https://stackoverflow.com/questions/29185493/read-svhn-dataset-in-python for instructions to load with numpy

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, x, y, transform=None):
        assert x.shape[0]==y.shape[0]
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x_t = self.transform(self.x[index])
        y_t = self.y[index]

        return x_t, y_t

    def __len__(self):
        return self.x.size(0)
    
class ResizeTransform(object):
    ''' Resizes a PIL image to (size, size) to feed into OpenFace net and returns a torch tensor.'''
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):
        img = sample.resize((self.size, self.size), Image.BILINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)
    
class ZeroPadBottom(object):
    ''' Zero pads batch of image tensor Variables on bottom to given size. Input (B, C, H, W) - padded on H axis. '''
    def __init__(self, size, use_gpu=True):
        self.size = size
        self.use_gpu = use_gpu
        
    def __call__(self, sample):
        B, C, H, W = sample.size()
        diff = self.size - H
        padding = Variable(torch.zeros(B, C, diff, W), requires_grad=False)
        if self.use_gpu:
            padding = padding.cuda()
        zero_padded = torch.cat((sample, padding), dim=2)
        return zero_padded
    
class NormalizeRangeTanh(object):
    ''' Normalizes a tensor with values from [0, 1] to [-1, 1]. '''
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample = sample * 2.0 - 1.0
        return sample
    
class UnNormalizeRangeTanh(object):
    ''' Unnormalizes a tensor with values from [-1, 1] to [0, 1]. '''
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample = (sample + 1.0) * 0.5
        return sample
        
    
class UnNormalize(object):
    ''' from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3'''
    def __init__(self, mean, std):
        mean_arr = []
        for dim in range(len(mean)):
            mean_arr.append(dim)
        std_arr = []
        for dim in range(len(std)):
            std_arr.append(dim)
        self.mean = torch.Tensor(mean_arr).view(1, len(mean), 1, 1)
        self.std = torch.Tensor(std_arr).view(1, len(std), 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor *= self.std
        tensor += self.mean
        return tensor
    