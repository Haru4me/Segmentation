from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.utils import save_image

from tqdm import tqdm
import pickle
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

from Tools.Models import *
from Tools.Dataset import CityscapesConvertor, Cityscapes
from Tools.Metrics import *
from Tools.Loss import TverskyLoss

parser = argparse.ArgumentParser()

parser.add_argument("--pretrain", type=int, default=False, help="Use pretrained model")
parser.add_argument("--train", type=int, default=True, help="Training or just test")
parser.add_argument("--epoch", type=int, default=1000, help="Epoch number")
parser.add_argument("--batch_size", type=int, default=12, help="Size of the batches")
parser.add_argument("--saving_interval", type=int, default=1, help="Second momentum of gradient")
parser.add_argument("--lr", type=float, default=0.045, help="Learning rate")
parser.add_argument("--a", type=float, default=0.9, help="Momentum of gradient")
opt = parser.parse_args()


os.makedirs("./FastSCNN/train_imgs", exist_ok=True)
os.makedirs("./FastSCNN/test_imgs", exist_ok=True)
os.makedirs("./FastSCNN/saved_model", exist_ok=True)
os.makedirs("./FastSCNN/saved_tests", exist_ok=True)


def save_samples(photos, groud_truth, generated, epoch_number, mode='train'):

    unloader = transforms.ToPILImage()
    convert = CityscapesConvertor()
    photos = photos.cpu().data

    groud_truth = convert.trainIdsToRGB(groud_truth.cpu().data)/255
    generated = convert.trainIdsToRGB(generated.cpu().data)/255



    img_sample = torch.cat((photos, groud_truth, generated), -2)
    save_image(img_sample[:4], "./FastSCNN/%s_imgs/%s.png" % \
               (mode, epoch_number), nrow=4, normalize=True)


def train(train_data, val_data, model, optimizer, lossfn, num_epoch=opt.epoch, k=opt.saving_interval, pretrain=opt.pretrain):

    print("Star training...")

    mean_loss = np.inf
    pbar = tqdm(range(1,num_epoch+1), ncols=70)
    labda = lambda epoch:  (1 - epoch/1000) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=labda)

    if pretrain:
        sch_dict = pickle.load(open("./FastSCNN/saved_model/scheduler_dict.p", "rb" ))
        scheduler.load_state_dict(sch_dict)
        scheduler.get_last_lr()

    for epoch in pbar:

        loss_run = []
        model.train()

        for batch in train_data:

            photos = batch[0].to(device)
            real = batch[1].to(device)

            optimizer.zero_grad()
            fake = model(photos)
            loss = lossfn(fake.squeeze(1), real.squeeze(1))
            loss.backward()
            optimizer.step()
            loss_run.append(loss.item())

        photos_val, real_val = next(iter(val_data))
        model.eval()
        fake_val = model(photos_val.to(device))
        scheduler.step()

        save_samples(photos_val, real_val,
                        fake_val, epoch, mode='train')

        pbar.set_description("Loss: %s" %str(np.mean(loss_run)))

        if mean_loss > np.mean(loss_run):#epoch % k == 0:

            mean_loss = np.mean(loss_run)
            torch.save(model, './FastSCNN/saved_model/SCNN.pth')
            pickle.dump(scheduler.state_dict(),
                        open("./FastSCNN/saved_model/scheduler_dict.p", "wb" ))

    print("Finished!")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.pretrain: model = torch.load('./FastSCNN/saved_model/SCNN.pth', map_location=device)
    else: model = FastSCNN().to(device)

    if opt.train:

        optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr, momentum=opt.a, weight_decay=4e-5)

        dataloader = DataLoader(
            Cityscapes(),
            batch_size=opt.batch_size,
            shuffle=True)

        validator = DataLoader(
            Cityscapes(split='val'),
            batch_size=opt.batch_size,
            shuffle=True)

        loss = nn.BCELoss().to(device)

        train(dataloader, validator, model, optimizer, loss)

    """
        Testing model
    """


    test_data = DataLoader(
                    Cityscapes(split='val'),
                    batch_size=opt.batch_size,
                    shuffle=True)

    acc = Accurasy()
    dice = DiceCoef()
    iou = IoU()

    acc_score = torch.zeros(19)
    dice_score = torch.zeros(19)
    iou_score = torch.zeros(19)

    cnt = 0

    for batch in tqdm(test_data, desc='Test', ncols=70):

        cnt += 1
        photos = batch[0].to(device)
        real = batch[1].to(device)

        model.eval()
        fake = model(photos)

        save_samples(photos, real, fake, cnt, mode='test')

        acc_score += acc(fake.cpu(), real.cpu())
        dice_score += dice(fake.cpu(), real.cpu())
        iou_score += iou(fake.cpu(), real.cpu())

    acc_score /= len(test_data)
    dice_score /= len(test_data)
    iou_score /= len(test_data)

    np.savetxt('./FastSCNN/saved_tests/%s.txt' %'Accurasy',
                    acc_score, delimiter=',')
    np.savetxt('./FastSCNN/saved_tests/%s.txt' %'Dice',
                    dice_score, delimiter=',')
    np.savetxt('./FastSCNN/saved_tests/%s.txt' %'IoU',
                    iou_score, delimiter=',')
