

import torch.nn.functional as F

from model.unet_parts import *
def C(y1,y2):

    diffY = torch.tensor([y2.size()[2] - y1.size()[2]])
    diffX = torch.tensor([y2.size()[3] - y1.size()[3]])

    x1 = F.pad(y1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([y2, x1], dim=1)

    return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #卷积
        self.inc = DoubleConv(n_channels,8)
        self.pool = pool()
        self.conv1 = DoubleConv(8, 16)
        self.conv2 = DilDoubleConv(8, 16)
        self.SPP1 = SPPLayer()
        self.pool = pool()
        self.conv3 = ThreefoldConv(128, 128)
        self.conv4 = DilThreefoldConv(128, 128)
        self.zyl = SelfAttention(8,128,128,0.4,0.4)


      
        self.up = Up()
        self.CT1 = ConvTranspose2d3(256,128)
        self.CT2 = ConvTranspose2d3(256,128)
        self.zyl2 = SelfAttention(8, 256, 256, 0.4, 0.4)
        self.up = Up()
        self.CT3 = ConvTranspose2d3(256,64)
        self.CT4 = ConvTranspose2d3(256,64)
        self.CT5 = ConvTranspose2d3(128, 32)
        self.conv = ConvTranspose2d3(32, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):

        x1 =self.inc(x)
        x2 =self.pool(x1)
        x3 =self.conv1(x2)
        x4 =self.conv2(x2)
        y = C(x3, x4)
        z =self.SPP1(y)
        u = self.zyl(z)
        x5 =self.pool(u)
        x6 = self.conv3(x5)
        x7 = self.conv4(x5)
        n = C(x6, x7)
        m = self.up(n)
        z1 = self.CT1(m)
        z2 = self.CT2(m)
        y = C(z1, z2)
        p =self.zyl2(y)
        z3 =self.up(p)

        z4 =self.CT3(z3)
        z5 =self.CT4(z3)
        z6 =C(z4, z5)
        z7 =self.CT5(z6)

        x = self.conv(z7)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
