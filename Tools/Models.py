from torchvision.models import vgg16
import torch.nn as nn
import torch
import copy




"""
    SegNet Decoder Layer
"""


class SegNetDecoder(nn.Module):

    def __init__(self, inp, outp, up=False):

        super(SegNetDecoder, self).__init__()

        net = []

        if up == True:
            net.append(nn.Upsample(scale_factor=2))

        net.append(nn.Conv2d(inp,outp,3, stride=1,padding=1))
        net.append(nn.BatchNorm2d(outp))

        net.append(nn.ReLU())

        self.layer = nn.Sequential(*net)

    def forward(self,img):
        return self.layer(img)




"""
    SegNet Model
"""


class SegNet(nn.Module):

    def __init__(self):

        super(SegNet, self).__init__()

        encoder_cnn =  copy.deepcopy(vgg16().features)

        self.encoder = nn.Sequential()

        i = 0
        for layer in encoder_cnn.children():

            if isinstance(layer, nn.Conv2d):
                i += 1
                value = int(str(layer).split(',')[1])
                self.encoder.add_module('conv_{}'.format(i), layer)
                self.encoder.add_module('norm_{}'.format(i), nn.BatchNorm2d(value))

            elif isinstance(layer, nn.ReLU):
                self.encoder.add_module('relu_{}'.format(i), layer)

            elif isinstance(layer, nn.MaxPool2d):
                self.encoder.add_module('pool_{}'.format(i), layer)

        self.decoder = nn.Sequential(
            SegNetDecoder(512,512, up=True),
            SegNetDecoder(512,512),
            SegNetDecoder(512,512),
            SegNetDecoder(512,256, up=True),
            SegNetDecoder(256,256),
            SegNetDecoder(256,256),
            SegNetDecoder(256,128, up=True),
            SegNetDecoder(128,128),
            SegNetDecoder(128,128),
            SegNetDecoder(128,64, up=True),
            SegNetDecoder(64,64),
            SegNetDecoder(64,64, up=True),
            SegNetDecoder(64,11),
            nn.Softmax2d()
        )


    def forward(self, img):

        enc = self.encoder(img)
        dec = self.decoder(enc)

        return dec

#============================================================================

"""
    U-Net Encoder
"""


class UnetEncoder(nn.Module):

    def __init__(self, inp, outp, norm=True, stride=1):

        super(UnetEncoder, self).__init__()

        net = [nn.Conv2d(inp,outp,3, stride=stride, padding=1)]
        net.append(nn.BatchNorm2d(outp))

        net.append(nn.ReLU())

        self.layer = nn.Sequential(*net)

    def forward(self,img):
        return self.layer(img)




"""
    U-Net Decoder
"""

class UnetDecoder(nn.Module):

    def __init__(self, inp, outp, norm=True, stride=1, padding=1):

        super(UnetDecoder, self).__init__()

        net = [nn.ConvTranspose2d(inp,outp,3, stride=stride, padding=1, output_padding=stride-1)]
        net.append(nn.BatchNorm2d(outp))

        net.append(nn.ReLU())

        self.layer = nn.Sequential(*net)

    def forward(self,img):
        return self.layer(img)




"""
    U-net model
"""


class UNet(nn.Module):


    def __init__(self):

        super(UNet, self).__init__()

        """
            Make encoder
        """

        self.enc0 = nn.Sequential(
            UnetEncoder(3  ,  64),
            UnetEncoder(64 ,  64)
        )

        self.enc1 = nn.Sequential(
            UnetEncoder(64 , 128, stride=2),
            UnetEncoder(128, 128),
            UnetEncoder(128, 128)
        )

        self.enc2 = nn.Sequential(
            UnetEncoder(128, 256, stride=2), #pool
            UnetEncoder(256, 256),
            UnetEncoder(256, 256)
        )

        self.enc3 = nn.Sequential(
            UnetEncoder(256, 512, stride=2), #pool
            UnetEncoder(512, 512),
            UnetEncoder(512, 512)
        )


        """
            Bottleneck
        """

        self.neck = nn.Sequential(
            UnetEncoder(512 , 1024, stride=2),
            UnetEncoder(1024, 1024, ),
            UnetEncoder(1024, 1024, ),
            UnetDecoder(1024, 512, stride=2)
        )


        """
            Make decoder
        """

        self.dec0 = nn.Sequential(
            UnetDecoder(1024, 512),
            UnetDecoder(512 , 512),
            UnetDecoder(512 , 256, stride=2)
        )

        self.dec1 = nn.Sequential(
            UnetDecoder(512 , 256),
            UnetDecoder(256 , 256),
            UnetDecoder(256 , 128, stride=2)
        )

        self.dec2 = nn.Sequential(
            UnetDecoder(256 , 128),
            UnetDecoder(128 , 128),
            UnetDecoder(128 ,  64, stride=2)
        )

        self.dec3 = nn.Sequential(
            UnetDecoder(128 ,  64),
            UnetDecoder(64  ,  64),
            UnetDecoder(64  ,  11),
            nn.Softmax2d()
        )


    def forward(self, img):

        e0 = self.enc0(img)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        neck = self.neck(e3)

        d0 = self.dec0(torch.cat((e3,neck),dim=1))
        d1 = self.dec1(torch.cat((e2,d0),dim=1))
        d2 = self.dec2(torch.cat((e1,d1),dim=1))
        d3 = self.dec3(torch.cat((e0,d2),dim=1))

        return d3
#============================================================================

"""
    PPM ( PYRAMID POOLING MODULE )
"""

"""
class PPM(nn.Module):

    def __init__(self, output_size):

        super(PPM, self).__init__()

        self.w, self.h = 32, 64
        self.net = []

        for i in output_size:

            self.net.append(nn.Sequential(nn.AvgPool2d((self.w//i, self.h//i),
                                                        stride=(self.w//i, self.h//i)),
                                          nn.Conv2d(128, 128//len(output_size),
                                                    3, stride=1, padding=1)))

    def forward(self, input: torch.Tensor):

        output = input.clone()

        for pool in self.net:

            cop = pool(input)
            cop = nn.functional.interpolate(cop, size=(self.w,self.h))
            output = torch.cat((output, cop), dim=1)

        return output
"""
class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class PPM(nn.Module):


    def __init__(self, in_channels, out_channels, **kwargs):
        super(PPM, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


"""
    Fast-SCNN Conv (Standart/Separable/Deepwise)
"""

class FastConv2d(nn.Module):

    def __init__(self,
                    in_channels, out_channels, kernel_size=3, stride=1,
                    padding=0, dilation=1, bias=True, padding_mode='zeros',
                    depth_multiplier=1, type=None):

        super(FastConv2d, self).__init__()

        if type == None:

            self.Conv = nn.Conv2d(in_channels, out_channels,
                        kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias,
                        padding_mode=padding_mode)

        elif type == 'DS':

            self.Conv = nn.Sequential(

                nn.Conv2d(in_channels, in_channels * depth_multiplier,
                            kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=bias,
                            groups=in_channels, padding_mode=padding_mode),

                nn.Conv2d(in_channels * depth_multiplier, out_channels,
                            1, stride=1,
                            padding=0, dilation=1, bias=bias,
                            padding_mode=padding_mode))

        elif type == 'DW':

            self.Conv = nn.Conv2d(in_channels, in_channels * depth_multiplier,
                        kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=bias,
                        groups=in_channels, padding_mode=padding_mode)


    def forward(self, input: torch.Tensor):

        return self.Conv(input)


"""
    bottleneck residual block
"""

class Bottleneck(nn.Module):
    """docstring fo Bottleneck."""

    def __init__(self, in_channels, out_channels, t=1, s=1):

        super(Bottleneck, self).__init__()

        net = []

        net.append(FastConv2d(in_channels, t*in_channels, 1))
        net.append(nn.BatchNorm2d(t*in_channels))
        net.append(nn.ReLU())
        net.append(FastConv2d(t*in_channels, t*in_channels, 3, stride=s, padding=1, type='DW'))
        net.append(nn.BatchNorm2d(t*in_channels))
        net.append(nn.ReLU())
        net.append(FastConv2d(t*in_channels, out_channels, 1))

        self.layer = nn.Sequential(*net)


    def forward(self, input: torch.Tensor):

        return self.layer(input)



class FastSCNN(nn.Module):

    def __init__(self):

        super(FastSCNN, self).__init__()

        #           Learning to Down-sample
        self.ds1 = FastConv2d(3,32,3,stride=2,padding=1)
        self.ds2 = FastConv2d(32,48,3,stride=2,padding=1,type='DS')
        self.ds3 = FastConv2d(48,64,3,stride=2,padding=1,type='DS')

        #           Global Feature Extractor
        self.gf1 = nn.Sequential(Bottleneck(64,64,t=6,s=2),
                                 Bottleneck(64,64,t=6,s=1),
                                 Bottleneck(64,64,t=6,s=1))
        self.gf2 = nn.Sequential(Bottleneck(64,128,t=6,s=2),  #96
                                 Bottleneck(128,128,t=6,s=1),
                                 Bottleneck(128,128,t=6,s=1))
        """
        self.gf3 = nn.Sequential(Bottleneck(96,128,t=6,s=1),
                                 Bottleneck(128,128,t=6,s=1),
                                 Bottleneck(128,128,t=6,s=1))
        """
        self.gf4 = PPM(128,128)

        #           Feature Fusion
        self.ff1 = FastConv2d(64,64,1)
        self.ff2 = nn.Sequential(nn.Upsample(scale_factor=4),
                                 FastConv2d(128,128,1, type='DW'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 FastConv2d(128,64,1))
        self.ff3 = nn.Sequential(nn.BatchNorm2d(128),
                                 nn.ReLU())

        #           Classifier
        self.clf1 = nn.Sequential(nn.Upsample(scale_factor=2),
                                  FastConv2d(128,128,3,stride=1,padding=1,type='DS'),
                                  nn.BatchNorm2d(128),
                                  nn.Dropout(0.3),
                                  nn.ReLU())
        self.clf2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                  FastConv2d(128,128,3,stride=1,padding=1,type='DS'),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU())
        self.clf3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                  FastConv2d(128,19,1,stride=1,padding=0),
                                  nn.Softmax2d())

    def forward(self, input):

        d1 = self.ds1(input)
        d2 = self.ds2(d1)
        d3 = self.ds3(d2)

        g1 = self.gf1(d3)
        g2 = self.gf2(g1)
        #g3 = self.gf3(g2)
        g4 = self.gf4(g2)

        f1 = self.ff1(d3)
        f2 = self.ff2(g4)
        f3 = self.ff3(torch.cat((f1,f2),dim=1))

        c1 = self.clf1(f3)
        c2 = self.clf2(c1)
        c3 = self.clf3(c2)

        return c3
