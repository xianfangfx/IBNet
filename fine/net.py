import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../backbone/resnet50-19c8e357.pth'), strict=False)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out3
    
    def initialize(self):
        weight_init(self)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv1b1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b1 = nn.BatchNorm2d(64)
        self.conv2b1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b1 = nn.BatchNorm2d(64)
        self.conv3b1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b1 = nn.BatchNorm2d(64)
        self.conv4b1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b1 = nn.BatchNorm2d(64)
        self.conv1d1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d1 = nn.BatchNorm2d(64)
        self.conv2d1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.conv3d1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d1 = nn.BatchNorm2d(64)
        self.conv4d1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d1 = nn.BatchNorm2d(64)
        self.conv1b2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b2 = nn.BatchNorm2d(64)
        self.conv2b2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b2 = nn.BatchNorm2d(64)
        self.conv3b2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b2 = nn.BatchNorm2d(64)
        self.conv4b2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b2 = nn.BatchNorm2d(64)
        self.conv1d2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d2 = nn.BatchNorm2d(64)
        self.conv2d2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv3d2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d2 = nn.BatchNorm2d(64)
        self.conv4d2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d2 = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)
        out1b1 = F.relu(self.bn1b1(self.conv1b1(out1)), inplace=True)
        out2b1 = F.relu(self.bn2b1(self.conv2b1(out2)), inplace=True)
        out3b1 = F.relu(self.bn3b1(self.conv3b1(out3)), inplace=True)
        out4b1 = F.relu(self.bn4b1(self.conv4b1(out4)), inplace=True)
        out1d1 = F.relu(self.bn1d1(self.conv1d1(out1)), inplace=True)
        out2d1 = F.relu(self.bn2d1(self.conv2d1(out2)), inplace=True)
        out3d1 = F.relu(self.bn3d1(self.conv3d1(out3)), inplace=True)
        out4d1 = F.relu(self.bn4d1(self.conv4d1(out4)), inplace=True)
        out1b2 = F.relu(self.bn1b2(self.conv1b2(out1)), inplace=True)
        out2b2 = F.relu(self.bn2b2(self.conv2b2(out2)), inplace=True)
        out3b2 = F.relu(self.bn3b2(self.conv3b2(out3)), inplace=True)
        out4b2 = F.relu(self.bn4b2(self.conv4b2(out4)), inplace=True)
        out1d2 = F.relu(self.bn1d2(self.conv1d2(out1)), inplace=True)
        out2d2 = F.relu(self.bn2d2(self.conv2d2(out2)), inplace=True)
        out3d2 = F.relu(self.bn3d2(self.conv3d2(out3)), inplace=True)
        out4d2 = F.relu(self.bn4d2(self.conv4d2(out4)), inplace=True)
        return (out4b1, out3b1, out2b1, out1b1), (out4d1, out3d1, out2d1, out1d1), (out4b2, out3b2, out2b2, out1b2), (out4d2, out3d2, out2d2, out1d2)

    def initialize(self):
        weight_init(self)


class IBNet_Res50(nn.Module):
    def __init__(self, cfg):
        super(IBNet_Res50, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet(cfg)
        self.conv5b1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5d1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5b2 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b2 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5d2 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d2 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d2 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.encoder = Encoder()
        self.decoderb1 = Decoder()
        self.decoderd1 = Decoder()
        self.decoderb2 = Decoder()
        self.decoderd2 = Decoder()
        self.linearb1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linearb2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.initialize()

    def forward(self, x, shape=None):
        out1, out2, out3, out4, out5 = self.bkbone(x)
        out2b1, out3b1, out4b1, out5b1 = self.conv2b1(out2), self.conv3b1(out3), self.conv4b1(out4), self.conv5b1(out5)
        out2d1, out3d1, out4d1, out5d1 = self.conv2d1(out2), self.conv3d1(out3), self.conv4d1(out4), self.conv5d1(out5)
        out2b2, out3b2, out4b2, out5b2 = self.conv2b2(out2), self.conv3b2(out3), self.conv4b2(out4), self.conv5b2(out5)
        out2d2, out3d2, out4d2, out5d2 = self.conv2d2(out2), self.conv3d2(out3), self.conv4d2(out4), self.conv5d2(out5)
        outb11 = self.decoderb1([out5b1, out4b1, out3b1, out2b1])
        outd11 = self.decoderd1([out5d1, out4d1, out3d1, out2d1])
        outb12 = self.decoderb2([out5b2, out4b2, out3b2, out2b2])
        outd12 = self.decoderd2([out5d2, out4d2, out3d2, out2d2])
        out1 = torch.cat([outb11, outd11, outb12, outd12], dim=1)
        outb21, outd21, outb22, outd22 = self.encoder(out1)
        outb21 = self.decoderb1([out5b1, out4b1, out3b1, out2b1], outb21)
        outd21 = self.decoderd1([out5d1, out4d1, out3d1, out2d1], outd21)
        outb22 = self.decoderb2([out5b2, out4b2, out3b2, out2b2], outb22)
        outd22 = self.decoderd2([out5d2, out4d2, out3d2, out2d2], outd22)
        out2 = torch.cat([outb21, outd21, outb22, outd22], dim=1)
        if shape is None:
            shape = x.size()[2:]
        outb11 = F.interpolate(self.linearb1(outb11), size=shape, mode='bilinear')
        outd11 = F.interpolate(self.lineard1(outd11), size=shape, mode='bilinear')
        outb12 = F.interpolate(self.linearb2(outb12), size=shape, mode='bilinear')
        outd12 = F.interpolate(self.lineard2(outd12), size=shape, mode='bilinear')
        out1 = F.interpolate(self.linear(out1), size=shape, mode='bilinear')
        outb21 = F.interpolate(self.linearb1(outb21), size=shape, mode='bilinear')
        outd21 = F.interpolate(self.lineard1(outd21), size=shape, mode='bilinear')
        outb22 = F.interpolate(self.linearb2(outb22), size=shape, mode='bilinear')
        outd22 = F.interpolate(self.lineard2(outd22), size=shape, mode='bilinear')
        out2 = F.interpolate(self.linear(out2), size=shape, mode='bilinear')
        return outb11, outd11, outb12, outd12, out1, outb21, outd21, outb22, outd22, out2


    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
