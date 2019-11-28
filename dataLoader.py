from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import glob
import cv2
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm

class DirtyDocumentsDataset(Dataset):
    def __init__(self, clean_dirs, dirty_dirs, transform=None):
        self.clean_files = []
        for clean_dir in clean_dirs:
            self.clean_files += sorted(glob.glob(clean_dir + '*.png'))
            
        self.dirty_files = []
        for dirty_dir in dirty_dirs:
            self.dirty_files += sorted(glob.glob(dirty_dir + '*.png'))
        
        self.H = 256 # Height
        self.W = 540 # Width
        self.transform = transform
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() # to list

        clean_img_name = self.clean_files[idx]
        clean_img = cv2.imread(clean_img_name)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        clean_img = cv2.resize(clean_img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        clean_img = np.reshape(clean_img, (H, W, 1))
        
        dirty_img_name = self.dirty_files[idx]
        dirty_img = cv2.imread(dirty_img_name)
        dirty_img = cv2.cvtColor(dirty_img, cv2.COLOR_BGR2GRAY)
        dirty_img = cv2.resize(dirty_img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        dirty_img = np.reshape(dirty_img, (H, W, 1))

        sample = {'clean_img' : clean_img, 'dirty_img': dirty_img}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class RandomCrop():
    def __init__(self, size, probability=0.2):
        # size should be a form of (H, W)
        assert isinstance(size, (tuple))
        self.size = size
        self.threshold = probability

    def __call__(self, sample):
        p = np.random.rand(1)

        if p <= self.threshold:
            clean_image = sample['clean_img']
            dirty_image = sample['dirty_img']

            clean_img = np.array(clean_image)
            dirty_img = np.array(dirty_image)

            x, y = clean_img.shape[:2]

            x = x - 1 - self.size[0]
            y = y - 1 - self.size[1]
            min_x = int(np.random.rand(1) * x)
            min_y = int(np.random.rand(1) * y)
            max_x = min_x + self.size[0]
            max_y = min_y + self.size[1]

            clean_crop = clean_img[min_x:max_x, min_y:max_y]
            dirty_crop = dirty_img[min_x:max_x, min_y:max_y]
            return {'clean_img' : clean_crop, 'dirty_img' : dirty_crop}
        else:
            return sample

        
    
class ImgAugTransform():
    def __init__(self, aug_probability=0.25, gaussian_blur_sigma=1.5):
        self.sometimes = lambda aug: iaa.Sometimes(aug_probability, aug)
        self.aug = iaa.Sequential([
            self.sometimes(
                iaa.GaussianBlur(sigma=(0, gaussian_blur_sigma))
            )
        ])    

    def __call__(self, sample):
        clean_image = sample['clean_img']
        dirty_image = sample['dirty_img']

        clean_img = np.array(clean_image)
        dirty_img = np.array(dirty_image)

        return {'clean_img' : clean_img, 'dirty_img' : self.aug.augment_image(dirty_img)}

class Rescale(object):
    def __init__(self, output_size):
        # output_size shoud be a form of (H, W).
        assert isinstance(output_size, (tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        clean_image = sample['clean_img']
        dirty_image = sample['dirty_img']

        h, w = self.output_size
        clean_image = transform.resize(clean_image, (h, w))
        dirty_image = transform.resize(dirty_image, (h, w))

        return {'clean_img':clean_image, 'dirty_img':dirty_image}
    
class ToTensor(object):
    def __call__(self, sample):
        clean_image = sample['clean_img']
        dirty_image = sample['dirty_img']

        clean_image = clean_image.transpose((2,0,1))
        dirty_image = dirty_image.transpose((2,0,1))

        return {'clean_img' : torch.from_numpy(clean_image), 'dirty_img' : torch.from_numpy(dirty_image)}
    
class DirtyDocumentsDataset_Test(Dataset):
    def __init__(self, test_dir, transform=None):
        self.img_files = glob.glob(test_dir + '*.png')
        self.transform = transform
        self.H = 256
        self.W = 540
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        test_img_name = self.clean_files[idx]
        test_img = cv2.imread(test_img_name)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_img = cv2.resize(test_img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        
        if self.transform:
            test_img = self.transform(test_img)
            
        test_img = test_image.transpose((2,0,1)) 
        return torch.from_numpy(test_image)

