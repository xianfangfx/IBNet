import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
import sys
import datetime
import dataset
from net import IBNet_Res50
sys.path.insert(0, '../')
sys.dont_write_bytecode = True


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def train(Dataset, Network):
    cfg = Dataset.Config(datapath='../data/DUTS/DUTS-TR', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=40)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)
    net = Network(cfg)
    net.train(True)
    net.cuda()
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(cfg.savepath)
    global_step = 0
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        for step, (image, mask, body1, detail1, body2, detail2) in enumerate(loader):
            image, mask, body1, detail1, body2, detail2 = image.cuda(), mask.cuda(), body1.cuda(), detail1.cuda(), body2.cuda(), detail2.cuda()
            outb11, outd11, outb12, outd12, out1, outb21, outd21, outb22, outd22, out2 = net(image)
            lossb11 = F.binary_cross_entropy_with_logits(outb11, body1)
            lossd11 = F.binary_cross_entropy_with_logits(outd11, detail1)
            lossb12 = F.binary_cross_entropy_with_logits(outb12, body2)
            lossd12 = F.binary_cross_entropy_with_logits(outd12, detail2)
            loss1 = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)
            lossb21 = F.binary_cross_entropy_with_logits(outb21, body1)
            lossd21 = F.binary_cross_entropy_with_logits(outd21, detail1)
            lossb22 = F.binary_cross_entropy_with_logits(outb22, body2)
            lossd22 = F.binary_cross_entropy_with_logits(outd22, detail2)
            loss2 = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
            loss = (lossb11 + lossd11 + lossb12 + lossd12 + loss1 + lossb21 + lossd21 + lossb22 + lossd22 + loss2)/2
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'lossb11': lossb11.item(), 'lossd11': lossd11.item(), 'lossb12': lossb12.item(), 'lossd12': lossd12.item(), 'loss1': loss1.item(), 'lossb21': lossb21.item(), 'lossd21': lossd21.item(), 'lossb22': lossb22.item(), 'lossd22': lossd22.item(), 'loss2': loss2.item()}, global_step=global_step)
            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | lossb11=%.4f | lossd11=%.4f | lossb12=%.4f | lossd12=%.4f | loss1=%.4f | lossb21=%.4f | lossd21=%.4f | lossb22=%.4f | lossd22=%.4f | loss2=%.4f' % (datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], lossb11.item(), lossd11.item(), lossb12.item(), lossd12.item(), loss1.item(), lossb21.item(), lossd21.item(), lossb22.item(), lossd22.item(), loss2.item()))
        if epoch > cfg.epoch*3/4:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__ == '__main__':
    train(dataset, IBNet_Res50)
