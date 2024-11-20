import json
import torch
import os
import argparse
from torch import nn
import torch.nn.functional as F
import numpy as np
from toolbox.datasets.vaihingen import Vaihingen
from toolbox.datasets.potsdam import Potsdam
from toolbox.datasets.Dataset_4EO import Dataset4EO
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
from toolbox.loss.loss import MscCrossEntropyLoss, FocalLossbyothers, MscLovaszSoftmaxLoss,CE_IOU
# from toolbox.loss.kd_losses1.TFFD import TfFD
from toolbox.loss.loss_contrast import PixelContrastLoss
from toolbox.loss.IOU import IOU
from toolbox.loss.SSIM import SSIM
from toolbox.loss.NCELoss.contrast_loss import SupConLoss
from toolbox.loss.NCELoss.cl1 import ContrastiveLoss
from toolbox.loss.NCELoss.CAC import CACLoss
from toolbox.loss.DiceLoss.dice_loss import DiceLoss
from toolbox.loss.NCELoss.cl import cl1
from toolbox.loss.NCELoss.cluster_cl import cl2
from toolbox.loss.NCELoss.cl_cl import CL_CL
from toolbox.loss.vid import VID
from toolbox.loss.cosime import CosineSimilarityLoss,TripletLoss
from toolbox.loss.NCELoss.SCL import SCL
from log import get_logger
import time
T
# from toolbox.models.lyz.T_ADC import MyConv_resnet# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # import toolbox.utils.ClassWeight
#
# # from toolbox.models.lyz.lyz_stu import Mobilenet_S
# from toolbox.models.lyz.lyz_2 import MyConv_resnet__T

# torch.backends.cudnn.benchmark = True
from toolbox.models.SFAF.model.SFAFMA import SFAFMA

from toolbox.models.lyz.Module2.PPNet13_T import SFAFMA_T
# from toolbox.models.lyz.Module2.PPNet13_S import SFAFMA_S

from toolbox.models.lyz.paper4.foul_DIM import L_W
from toolbox.models.lyz.paper4.PROMPT_1 import L_W
from toolbox.models.CAINet.toolbox.models.cainet import mobilenetGloRe3_CRRM_dule_arm_bou_att
from toolbox.models.MSEDNet.net import Net
from toolbox.models.lyz.paper4.PROMPT_1_FFM import L_W
# from toolbox.models.lyz.paper4.PROMPT_1_DIM import L_W
# from toolbox.models.lyz.paper4.PROMPT_1_DIM_ndsm import L_W
from toolbox.models.lyz.paper4.Light_weight_27 import L_W
# from toolbox.models.lyz.paper4.two_DIM import L_W
# from toolbox.models.lyz.paper4.one_DIM import L_W
# from toolbox.models.LSNet.LSNet.LSNet import LSNet
# from toolbox.models.GeleNet.model.GeleNet_models import GeleNet
from toolbox.models.SeaNet.model.SeaNet_models import SeaNet
# from toolbox.models.CorrNet_main.model.CorrNet_models import CorrelationModel_VGG
# from toolbox.models.MAGNet.model.MAGNet import MAGNet

# from toolbox.models.lyz.paper3.pvt_resnet import paper3
# from toolbox.models.lyz.paper3.resnet_pvt import paper3
# from toolbox.models.lyz.paper3.resnet_p2t import paper3
# from toolbox.models.lyz.paper3.resnet_segformer import paper3
# from toolbox.models.lyz.paper3.resnet_resnet import paper3

from toolbox.models.DFMNet.net import DFMNet


DATASET = "Potsdam"
# DATASET = "Vaihingen"
# DATASET = "Dataset4EO"
batch_size = 10
import argparse
parser = argparse.ArgumentParser(description="config")
parser.add_argument(
    "--config",
    nargs="?",
    type=str,
    default="configs/{}.json".format(DATASET),
    help="Configuration file to use",
)
args = parser.parse_args()
with open(args.config, 'r') as fp:
    cfg = json.load(fp)
if DATASET == "Potsdam":
    train_dataloader = DataLoader(Potsdam(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(Potsdam(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
elif DATASET == "Vaihingen":
    train_dataloader = DataLoader(Vaihingen(cfg, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    test_dataloader = DataLoader(Vaihingen(cfg, mode='test'), batch_size=batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
elif DATASET == "Dataset4EO":
    train_dataloader = DataLoader(Dataset4EO(r'/media/panyi/dd67419f-4b06-4cc1-a57e-0706577875076/RSUSS2/', 'train'),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_dataloader = DataLoader(Dataset4EO(r'/media/panyi/dd67419f-4b06-4cc1-a57e-0706577875076/RSUSS2/', 'val'),
                                batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# weight = ClassWeight('median_freq_balancing')
criterion = nn.CrossEntropyLoss().cuda()
criterion_dice = DiceLoss().cuda()

criterion_without = MscCrossEntropyLoss().cuda()
iou = IOU()
ssim = SSIM()
# criterion1 = nn.CrossEntropyLoss(weight=weight.get_weight(test_dataloader, num_classes=5)).cuda()
print(DATASET)
criterion_focal1 = FocalLossbyothers().cuda()
criterion_Lovasz = MscLovaszSoftmaxLoss().cuda()
criterion_bce = nn.BCELoss().cuda()  # 边界监督
NCE = ContrastiveLoss().cuda()  # 边界监督
criterion_CAC = CACLoss(hard_sort_hardmining=True, easy_sort_hardmining=True).cuda()  # 边界监督

criterion_cl = PixelContrastLoss().cuda()
criterion_cl1 = cl1().cuda()
criterion_cl2 = cl2().cuda()
criterion_cl3 = CL_CL().cuda()
criterion_VID = VID(512,5).cuda()
criterion_CosineSimilarityLoss = CosineSimilarityLoss()

# net = SPNet(nclass=6, backbone='resnet101', pretrained='/home/maps/PycharmProjects/DFCN/pretrained/resnet101-2a57e44d.pth', criterion=criterion).cuda()

"""
pretrained_dict = model_zoo.load_url(model_urls["res2net101_26w_4s"])

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
"""

net = Net().cuda()
# net.load_pre('/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/backbone/p2t_segmentation/ADE20K/sem_fpn_p2t_s_ade20k_80k.pth')
# net = DFMNet().cuda()


# net = A3Net_Res2Net50().cuda()
# net.load_pre('/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/backbone/p2t_segmentation/ADE20K/sem_fpn_p2t_s_ade20k_80k.pth')
# print('Total params % .2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
# model_dict = net.state_dict()
# pretrained_dict = torch.load('./weight/JJNet6_1-Potsdam-loss.pth')
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
# net.load_state_dict(torch.load('./weight/FFNet9(best)-Vaihingen-loss.pth'))
# optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-4)
# net.load_pretrained_model()
#
# resume = True
# if resume:
#     net.state_dict(torch.load('./weight/MyConv_resnet_T(ADC)-Potsdam-loss.pth'))
# net.load_state_dict(torch.load('/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/models/lyz/paper4/run/2024-07-08-21-14(Light_weight_27_V)/_best.pth'))
# net.load_state_dict(torch.load('/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/models/lyz/paper4/run/2024-07-19-08-53(L_W_P)/_best.pth'))
def accuary(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size
best = [0.0]
size = (56, 56)
numloss = 0
nummae = 0
trainlosslist_nju = []
vallosslist_nju = []
iter_num = len(train_dataloader)
epochs = 350
# CAC_loss
# model = "paper3_26_resnet_pvt(lwa_cluster_VID(b*c_H*W))"

model = "MSEDNet_Potsdam"
# model = "paper3_24(lwa_cluster_VID(b*c_H*w))"
# model = "paper3_24(lwa_cluster)_Potsdam"
time = time.strftime("%Y-%m-%d-%H-%M")
logdir = f'/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/models/lyz/paper4/run/{time}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)
logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')
logger.info(f'Conf | use dataset {DATASET}')
logger.info(f'Conf | use batch_size {batch_size}')
logger.info(f'Conf | use epochs {epochs}')
logger.info(f'Conf | use model {model}')
#
bestpath = f'{logdir}/' + '_best.pth'
lastpath = f'{logdir}/' + '_last.pth'
# midpath = f'{logdir}/'

# schduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # setting the learning rate desend starage
for epoch in range(epochs):
    if epoch % 20 == 0 and epoch != 0:  # setting the learning rate desend starage
        for group in optimizer.param_groups:
            group['lr'] = 0.1 * group['lr']
    # for group in optimizer.param_groups:
    # 	group['lr'] *= 0.99
    # 	print(group['lr'])
    train_loss = 0
    net = net.train()
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['image'].cuda())  # [2, 3, 256, 256]
        ndsm = Variable(sample['dsm'].cuda())  # [2, 1, 256, 256]
        label = Variable(sample['label'].long().cuda())  # [2, 256, 256]
        # edge = Variable(sample['edge'].float().cuda())  # 边界监督 [12, 256, 256]
        ndsm = torch.repeat_interleave(ndsm, 3, dim=1)
        out = net(image,ndsm)


        # input = torch.cat((image,ndsm),dim=1)
        # out = net(image)
        # print(out.shape)
        # print(out.shape)
        # print('lab',out[0].shape)
        # loss1 = criterion_without(out[0],label) + iou(out[0],label)
        # loss2 = criterion_without(out[1],label) + iou(out[1],label)
        # loss3 = criterion_without(out[2],label) + iou(out[2],label)
        # loss4 = criterion_without(out[3],label) + iou(out[3],label)
        # loss5 = criterion_without(out[4],label) + iou(out[4],label)
        # # #
        # loss_avg = (loss5 + loss4 + loss3 + loss2 + loss1)/5
        # loss = loss_avg
        # loss_nce1 = NCE(out[5],out[6])
        # loss_nce2 = NCE(out[7],out[8])
        # # loss_nce3 = NCE(out[9],out[10])
        # # loss_nce4 = NCE(out[11],out[12])
        # loss_nce = loss_nce1 + loss_nce2
        # print(loss_nce)


        # loss = criterion_without(out[0:5], label)
        # print(out[0].shape)
        loss = criterion_without(out, label)
        # loss = loss + out[3]
        # loss1 = criterion_CosineSimilarityLoss(out[5],out[6])
        # loss2 = criterion_CosineSimilarityLoss(out[7],out[8])
        # loss3 = criterion_CosineSimilarityLoss(out[9],out[10])
        # loss4 = criterion_CosineSimilarityLoss(out[11],out[12])
        # loss5 = criterion_CosineSimilarityLoss(out[13],out[14])
        # loss = loss5 + loss4 + loss3 + loss2 + loss1 + loss
        # loss_cl1 = criterion_VID(out[6],out[7])
        #
        # loss = loss_cl1 + loss
        # print(out[6].shape)
        # print(out[7].shape)
        # print(out[12].shape)
        # print(out[13].shape)
        # loss_scl1 = SCL(out[6],out[11])
        # loss_scl2 = SCL(out[7],out[12])
        # loss = criterion_without(out[0:5], label)
        # weight_logit1 = torch.rand((9, 6, 256, 256), requires_grad=True).cuda()
        # weight_logit2 = torch.rand((7, 6, 64, 64), requires_grad=True).cuda()
        # loss_cac = criterion_CAC(out[1], out[6], label, out[1])
        # loss = loss + loss_cac
        # loss2 = criterion_without(out[1], label)
        # loss1 = criterion1(out[5],out[6],out[7],out[8],out[9],out[10])
        # loss = loss + loss_scl1 + loss_scl2
        # 边界监督
        # loss1 = criterion_without(out[0], label)
        # loss2 = criterion_bce(nn.Sigmoid()(out[1]), edge)
        # loss = (loss2 + loss1) / 2
        # 边界监督

        print('Training: Iteration {:4}'.format(i), 'Loss:', loss.item())
        if (i+1) % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                epoch+1, epochs, i+1, iter_num, train_loss / 100))
            train_loss = 0

        optimizer.zero_grad()

        loss.backward()  # backpropagation to get gradient
        # qichuangaaaxuexi
        optimizer.step()  # update the weight

        train_loss = loss.item() + train_loss

    net = net.eval()
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):
            imageVal = Variable(sampleTest['image'].float().cuda())
            ndsmVal = Variable(sampleTest['dsm'].float().cuda())
            labelVal = Variable(sampleTest['label'].long().cuda())
            # imageVal = F.interpolate(imageVal, (320, 320), mode="bilinear", align_corners=True)
            # ndsmVal = F.interpolate(ndsmVal, (320, 320), mode="bilinear", align_corners=True)
            # labelVal = F.interpolate(labelVal.unsqueeze(1).float(), (320, 320),
            #                          mode="bilinear", align_corners=True).squeeze(1).long()
            ndsmVal = torch.repeat_interleave(ndsmVal, 3, dim=1)
            # teacherVal, studentVal = net(imageVal, ndsmVal)
            # outVal = net(imageVal)
            # inputVal = torch.cat((imageVal, ndsmVal), dim=1)
            # outVal = net(imageVal)

            outVal = net(imageVal, ndsmVal)

            loss = criterion_without(outVal, labelVal)
            # loss = criterion_without(outVal[0:5], labelVal)

            outVal = outVal[0].max(dim=1)[1].data.cpu().numpy()
            # outVal = outVal.max(dim=1)[1].data.cpu().numpy()

            labelVal = labelVal.data.cpu().numpy()

            accval = accuary(outVal, labelVal)
            # print('accVal:', accval)
            print('Valid:    Iteration {:4}'.format(j), 'Loss:', loss.item())
            eval_loss = loss.item() + eval_loss
            acc = acc + accval

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f},Valid Loss: {:.5f},Valid Acc: {:.5f}'.format(
        epoch, train_loss / len(train_dataloader), eval_loss / len(test_dataloader), acc / len(test_dataloader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)
    logger.info(
        f'Epoch:{epoch + 1:3d}/{epochs:3d} || trainloss:{train_loss / len(train_dataloader):.8f} valloss:{eval_loss / len(test_dataloader):.8f} || '
        f'acc:{acc / len(test_dataloader):.8f} || spend_time:{time_str}')
    trainlosslist_nju.append(train_loss / len(train_dataloader))
    vallosslist_nju.append(eval_loss / len(test_dataloader))
    # if acc / len(test_dataloader) >= 86.0:
    #     torch.save(net.state_dict(), f'{logdir}/+{epoch}.pth')

    if acc / len(test_dataloader) >= max(best):
        best.append(acc / len(test_dataloader))
        numloss = epoch
        # torch.save(net.state_dict(), './weight/paper3(resnet_18+Mobilenet)-{}-loss.pth'.format(DATASET))
        # torch.save(net.state_dict(), './weight/Res_50_PVTB5-{}-loss.pth'.format(DATASET))
        torch.save(net.state_dict(), bestpath)
        # torch.save(net.state_dict(), './weight/MyConv_resnet_T(backbone)_1-{}-loss.pth'.format(DATASET))
        # torch.save(net.state_dict(), './weight/SAFA-{}-loss.pth'.format(DATASET))
        # torch.save(net.state_dict(), './toolbox/models/lyz/weight/ppnet-{}-loss.pth'.format(DATASET))
    # if epoch > 5 :
    #     torch.save(net.state_dict(), midpath + '{}.pth'.format(epoch))
    torch.save(net.state_dict(), lastpath)
    # torch.save(net.state_dict(), './weight/DGNet_last-{}-loss.pth'.format(DATASET))
    # torch.save(net.state_dict(), './weight/tem/SFAFMA_T(paper2)_{}_last-{}-loss.pth'.format(epoch,DATASET))
    # if eval_loss / len(test_dataloader) <= min(best):
    #     best.append(eval_loss / len(test_dataloader))
    #     numloss = epoch
    #     torch.save(net.state_dict(), './weight/JJNet14(3)-Potsdam-loss.pth')

    # print()
    #     torch.save(net.state_dict(), './weight/JJNet11-Potsdam-loss.pth')

    # print('best acc: ', best)

    print(max(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Accuracy', numloss)
    # print(min(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Loss', numloss)

    # trainlosslistnumpy = np.array(trainlosslist_nju)
    # vallosslistnumpy = np.array(vallosslist_nju)
    #
    # np.savetxt('trainloss_M2RESIZE_nlpr.txt', trainlosslistnumpy)
    # np.savetxt('valloss_M2RESIZE_nlpr.txt', vallosslistnumpy)
#
# """
# use class_weight in train_dataset is 85.04579784800706
# use class_weight in test_dataset is 85.62239859868022
#
# 84.89822811550565
# 82.3266911
# MyConv_resnet_T(image)
# MyConv_resnet_T(backbone)