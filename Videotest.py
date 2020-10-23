from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from pathlib import Path

from tqdm import tqdm, trange
import imageio
import argparse
from time import time
import os

import warnings
warnings.filterwarnings("ignore")

from Tools.Models import *
from Tools.Dataset import CamVidConvertor, CamVid, Cityscapes, CityscapesConvertor


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="SegNet", help="Choose testing model")
opt = parser.parse_args()

os.makedirs("./%s/Video"%opt.model, exist_ok=True)

class CamVidVideo(Dataset):

    def __init__(self):

        self.image_pathes = sorted(Path('./CamVid/val/').rglob('*.png'))

    def __getitem__(self, index):

        transform = transforms.Compose([transforms.Resize((350,512)),
                                        transforms.CenterCrop((256,512)),
                                        transforms.ToTensor()])
        image_path = self.image_pathes[index % len(self.image_pathes)]
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)

        return image.float()

    def __len__(self):

        return len(self.image_pathes)

class CityscapesVideo(Dataset):


    def __init__(self):

        self.image_pathes = sorted(Path('./cityscapes/demoVideo/stuttgart_02').rglob('*.png'))


    def __getitem__(self, index):

        image_path = self.image_pathes[index % len(self.image_pathes)]

        transform = transforms.Compose([transforms.Resize((768,1536)),
                                        transforms.CenterCrop((512,1024)),
                                        transforms.ToTensor()])
        image = Image.open(image_path).convert('RGB')#.resize((1024,  512))
        image = transform(image).unsqueeze(0)
        #image = transforms.ToTensor()(image)

        return image.float()


    def __len__(self):

        return len(self.image_pathes)


def get_mask(img, mask, alpha=0.4):
    return (img*(1-alpha) + alpha*mask)

def save_samples(photos, generated, epoch_number, convert):

    unloader = transforms.ToPILImage()
    photos = photos.cpu().data
    generated = convert.trainIdsToRGB(generated.cpu().data) / 255

    mask = get_mask(photos, generated)

    save_image(mask, "./%s/Video/%s.png" %(opt.model,epoch_number),
                nrow=1, normalize=True)

if __name__ == "__main__":

    if opt.model in ["SegNet", "UNet"]:

        data = DataLoader(
            CamVidVideo(),
            batch_size=1,
            shuffle=False)

        convert = CamVidConvertor()

    elif opt.model == "FastSCNN":

        data = DataLoader(
            CityscapesVideo(),
            batch_size=1,
            shuffle=False)

        convert = CityscapesConvertor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('./%s/saved_model/SCNN.pth'%opt.model, map_location=device)
    model.eval()

    cnt = 0
    pbar = tqdm(data, ncols=70)

    for batch in pbar:

        cnt += 1
        photos = batch[0].to(device)

        start = time()
        fake = model(photos)
        finish = time()

        speed = finish - start
        pbar.set_description("FPS: %s"%str(1/speed))

        save_samples(photos, fake, cnt, convert)

    images = []
    for filename in trange(1,len(data)+1, desc='Generating mp4'):
        images.append(imageio.imread('./%s/Video/%i.png'%(opt.model,filename)))

    imageio.mimsave('./%s/Video/%s.mp4'%(opt.model,opt.model), images, fps=20)
