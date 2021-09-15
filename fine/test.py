import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import dataset
from net import IBNet_Res50
import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True


class Test(object):
    def __init__(self, Dataset, Network, Path):
        self.cfg = Dataset.Config(datapath=Path, snapshot='./out/model-40', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape = image.cuda().float(), (H, W)
                outb11, outd11, outb12, outd12, out1, outb21, outd21, outb22, outd22, out2 = self.net(image, shape)
                out = out2
                pred = torch.sigmoid(out[0, 0]).cpu().numpy()*255
                head = '../output/maps/'+self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__ == '__main__':
    ##
    for path in ['../data/ECSSD', '../data/DUT-OMRON', '../data/PASCAL-S', '../data/DUTS/DUTS-TE']:
    # for path in ['../data/HKU-IS']:
    ##
        t = Test(dataset, IBNet_Res50, path)
        t.save()
