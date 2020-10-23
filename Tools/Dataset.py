from PIL import Image, ImageFilter
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import random


"""
    Cityscapes
"""


class CityscapesConvertor():

    def __init__(self):

        self.reclass_mapper = {

             7  :   0 ,
             8  :   1 ,
            11  :   2 ,
            12  :   3 ,
            13  :   4 ,
            17  :   5 ,
            19  :   6 ,
            20  :   7 ,
            21  :   8 ,
            22  :   9 ,
            23  :  10 ,
            24  :  11 ,
            25  :  12 ,
            26  :  13 ,
            27  :  14 ,
            28  :  15 ,
            31  :  16 ,
            32  :  17 ,
            33  :  18

        }

        self.rgb_mapper = {

              0 : (128, 64,128),
              1 : (244, 35,232),
              2 : ( 70, 70, 70),
              3 : (102,102,156),
              4 : (190,153,153),
              5 : (153,153,153),
              6 : (250,170, 30),
              7 : (220,220,  0),
              8 : (107,142, 35),
              9 : (152,251,152),
             10 : ( 70,130,180),
             11 : (220, 20, 60),
             12 : (255,  0,  0),
             13 : (  0,  0,142),
             14 : (  0,  0, 70),
             15 : (  0, 60,100),
             16 : (  0, 80,100),
             17 : (  0,  0,230),
             18 : (119, 11, 32)

        }


    def toNewLabels(self, image):

        size = list(image.size())
        size[0] = len(self.rgb_mapper.keys())   # Nx1xHxW -> NxCxHxW, where C â€“ number of classes
        newimage = torch.zeros(size, dtype=torch.uint8)

        for key in self.reclass_mapper:
            newimage[self.reclass_mapper[key]][image[0] == key] = 1

        return newimage


    def trainIdsToRGB(self, image):

        size = list(image.size())
        size[1] = 3
        newimage = torch.zeros(size)

        for key in self.rgb_mapper:

            newimage[:,0][image[:,key].round() == 1] = self.rgb_mapper[key][0]
            newimage[:,1][image[:,key].round() == 1] = self.rgb_mapper[key][1]
            newimage[:,2][image[:,key].round() == 1] = self.rgb_mapper[key][2]

        return newimage



class Cityscapes(Dataset):


    def __init__(self, split='train', transform=None, crop_size=(1024,  512)):

        self.image_pathes = sorted(Path('./cityscapes/leftImg8bit/{}'.format(split)).rglob('*.png'))
        self.target_pathes = sorted(Path('./cityscapes/gtFine/{}'.format(split)).rglob('*labelIds.png'))

        self.transform = transform
        self.crop_size = crop_size

    def par_transform(self, img, mask):

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        w, h = img.size
        newsize = random.randint(h//2, w//2)

        if w >= h:
            newh = newsize
            neww = int(w * newh / h)
        elif w < h:
            neww = newsize
            newh = int(h * neww / w)

        img = img.resize((neww, newh), Image.BILINEAR)
        mask = mask.resize((neww, newh), Image.NEAREST)

        x1 = random.randint(0, neww - self.crop_size[0])
        y1 = random.randint(0, newh - self.crop_size[1])

        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask


    def __getitem__(self, index):

        image_path = self.image_pathes[index % len(self.image_pathes)]
        target_path = self.target_pathes[index % len(self.target_pathes)]

        image = Image.open(image_path).convert('RGB')
        target = Image.open(target_path).convert('L')
        image, target = self.par_transform(image, target)

        if self.transform != None:
            image = self.transform(image)
            target = self.transform(target) * 255
        else:
            image = transforms.ToTensor()(image)
            target = transforms.ToTensor()(target) * 255

        convertor = CityscapesConvertor()

        return image, convertor.toNewLabels(target).float()


    def __len__(self):

        return max(len(self.image_pathes), len(self.target_pathes))


"""
    CamVid Dataset
"""


class CamVidConvertor():

    def __init__(self):

        df = pd.read_csv("./CamVid/class_dict.csv")
        self.mapper = dict()

        for i,elm in enumerate(df[df.class_11 == 1][['r','g','b']].values):
            self.mapper[i] = tuple(elm)

    def rgbToLabel(self, image):

        size = list(image.shape)
        size[0] = len(self.mapper)
        newimage = torch.zeros(size, dtype=torch.uint8)

        for key in self.mapper:
            mapp = (image[0] == self.mapper[key][0]) *\
                    (image[1] == self.mapper[key][1]) *\
                    (image[2] == self.mapper[key][2])

            newimage[key][mapp] = 1

        return newimage.float()

    def trainIdsToRGB(self, image):

        size = list(image.size())
        size[1] = 3
        newimage = torch.zeros(size)

        for key in self.mapper:
            newimage[:,0][image[:,key].round() == 1] = self.mapper[key][0]
            newimage[:,1][image[:,key].round() == 1] = self.mapper[key][1]
            newimage[:,2][image[:,key].round() == 1] = self.mapper[key][2]

        return newimage


class CamVid(Dataset):

    def __init__(self, split='train', transform=None, crop_size=(256,256), maxanle=10):

        self.image_pathes = sorted(Path('./CamVid/{}'.format(split)).rglob('*.png'))
        self.target_pathes = sorted(Path('./CamVid/{}_labels'.format(split)).rglob('*.png'))

        self.transform = transform
        self.crop_size = crop_size
        self.maxanle = maxanle

    def par_transform(self, img, mask):

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        ang = random.randint(0, self.maxanle)

        if random.random() < 0.5: ang = -ang

        img = img.rotate(ang)
        mask = mask.rotate(ang)

        w, h = img.size
        newsize = random.randint(256,450)

        if w >= h:
            newh = newsize
            neww = int(w * newh / h)
        elif w < h:
            neww = newsize
            newh = int(h * neww / w)

        img = img.resize((neww, newh), Image.BILINEAR)
        mask = mask.resize((neww, newh), Image.NEAREST)

        x1 = random.randint(0, neww - self.crop_size[0])
        y1 = random.randint(0, newh - self.crop_size[1])

        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return img, mask


    def __getitem__(self, index):

        image_path = self.image_pathes[index % len(self.image_pathes)]
        target_path = self.target_pathes[index % len(self.target_pathes)]

        image = Image.open(image_path)
        target = Image.open(target_path).convert('RGB')
        image, target = self.par_transform(image, target)

        if self.transform != None:
            image = self.transform(image)
            target = self.transform(target) * 255
        else:
            image = transforms.ToTensor()(image)
            target = transforms.ToTensor()(target) * 255

        return image.float(), CamVidConvertor().rgbToLabel(target.float())


    def __len__(self):

        return max(len(self.image_pathes), len(self.target_pathes))
