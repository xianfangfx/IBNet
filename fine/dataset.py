import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std = std
    
    def __call__(self, image, mask=None, body1=None, detail1=None, body2=None, detail2=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        return image, mask/255, body1/255, detail1/255, body2/255, detail2/255


class RandomCrop(object):
    def __call__(self, image, mask=None, body1=None, detail1=None, body2=None, detail2=None):
        H,W,_ = image.shape
        randw = np.random.randint(W/8)
        randh = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], body1[p0:p1, p2:p3], detail1[p0:p1, p2:p3], body2[p0:p1, p2:p3], detail2[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None, body1=None, detail1=None, body2=None, detail2=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), body1[:, ::-1].copy(), detail1[:, ::-1].copy(), body2[:, ::-1].copy(), detail2[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, body1, detail1, body2, detail2


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body1=None, detail1=None, body2=None, detail2=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body1 = cv2.resize(body1, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        detail1 = cv2.resize(detail1, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body2 = cv2.resize(body2, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        detail2 = cv2.resize(detail2, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, body1, detail1, body2, detail2


class ToTensor(object):
    def __call__(self, image, mask=None, body1=None, detail1=None, body2=None, detail2=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)
        body1 = torch.from_numpy(body1)
        detail1 = torch.from_numpy(detail1)
        body2 = torch.from_numpy(body2)
        detail2 = torch.from_numpy(detail2)
        return image, mask, body1, detail1, body2, detail2


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]
        ##
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:, :, ::-1].astype(np.float32)
        # image = cv2.imread(self.cfg.datapath+'/image/'+name+'.png')[:, :, ::-1].astype(np.float32)
        ##
        if self.cfg.mode == 'train':
            mask = cv2.imread(self.cfg.datapath+'/mask/'+name+'.png', 0).astype(np.float32)
            body1 = cv2.imread(self.cfg.datapath+'/body1/'+name+'.png', 0).astype(np.float32)
            detail1 = cv2.imread(self.cfg.datapath+'/detail1/'+name+'.png', 0).astype(np.float32)
            body2 = cv2.imread(self.cfg.datapath+'/body2/'+name+'.png', 0).astype(np.float32)
            detail2 = cv2.imread(self.cfg.datapath+'/detail2/'+name+'.png', 0).astype(np.float32)
            image, mask, body1, detail1, body2, detail2 = self.normalize(image, mask, body1, detail1, body2, detail2)
            image, mask, body1, detail1, body2, detail2 = self.randomcrop(image, mask, body1, detail1, body2, detail2)
            image, mask, body1, detail1, body2, detail2 = self.randomflip(image, mask, body1, detail1, body2, detail2)
            return image, mask, body1, detail1, body2, detail2
        else:
            shape = image.shape[:2]
            image = self.normalize(image)
            image = self.resize(image)
            image = self.totensor(image)
            return image, shape, name

    def __len__(self):
        return len(self.samples)
    
    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, body1, detail1, body2, detail2 = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            body1[i] = cv2.resize(body1[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            detail1[i] = cv2.resize(detail1[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            body2[i] = cv2.resize(body2[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            detail2[i] = cv2.resize(detail2[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        body1 = torch.from_numpy(np.stack(body1, axis=0)).unsqueeze(1)
        detail1 = torch.from_numpy(np.stack(detail1, axis=0)).unsqueeze(1)
        body2 = torch.from_numpy(np.stack(body2, axis=0)).unsqueeze(1)
        detail2 = torch.from_numpy(np.stack(detail2, axis=0)).unsqueeze(1)
        return image, mask, body1, detail1, body2, detail2
