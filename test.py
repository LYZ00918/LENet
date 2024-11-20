from datetime import datetime
import torch.utils.data as data
import torch.optim.lr_scheduler
import torch.nn.init
from dataprocessing import *
from tqdm import tqdm
from toolbox.datasets.Dataset_4EO import Dataset4EO_test
from torch.utils.data import DataLoader
from toolbox.models.ACCoNet.model.ACCoNet_Res_models import ACCoNet_Res

from toolbox.models.lyz.lyz_2 import MyConv_resnet_T
# from toolbox.models.lyz.T_ADC import MyConv_resnet_T
# from toolbox.models.lyz.lyz_stu import Mobilenet_S
# from toolbox.models.lyz.lyz_test_S1 import Mobilenet_S

from toolbox.models.lyz.Module2.PPNet13_T import SFAFMA_T
from toolbox.models.lyz.Module2.PPNet13_S import SFAFMA_S
# from toolbox.models.lyz.Module2.PPNet13_T import SFAFMA_T
# from toolbox.models.SFAF.model.SFAFMA import SFAFMA
# from toolbox.models.GeleNet.model.GeleNet_models import GeleNet
# from toolbox.models.CMTFNet.models.CMTFNet import CMTFNet
from toolbox.models.DGNet.lib_pytorch.lib.DGNet import DGNet
# from toolbox.models.lyz.paper3.ablation.paper3_26_resnet_pvt import paper3
from toolbox.models.lyz.paper4.one_DIM import L_W
# from toolbox.models.lyz.paper4.foul_DIM import L_W
from toolbox.models.lyz.paper4.Light_weight_27 import L_W
from toolbox.models.lyz.paper4.PROMPT_1 import L_W
from toolbox.models.lyz.paper4.PROMPT_1_FFM import L_W
from toolbox.models.CAINet.toolbox.models.cainet import mobilenetGloRe3_CRRM_dule_arm_bou_att
from toolbox.models.MSEDNet.net import Net
from toolbox.models.DGPInet.DGPINet_T import EnDecoderModel as DGPINetT
from toolbox.models.DGPInet.DGPINet_S import EnDecoderModel as DGPINetS
from toolbox.models.SXY.KD_new_model_1_T import teacher_model
from toolbox.models.SXY.KD_new_model_1_S import student_model
# from toolbox.models.lyz.paper4.PROMPT_1_DIM import L_W
# from toolbox.models.lyz.paper4.PROMPT_1_DIM_ndsm import L_W
# from toolbox.models.LSNet.LSNet.LSNet import LSNet
# from toolbox.models.GeleNet.model.GeleNet_models import GeleNet
# from toolbox.models.SeaNet.model.SeaNet_models import SeaNet
# from toolbox.models.MAGNet.model.MAGNet import MAGNet
# from toolbox.models.lyz.PPNet13_T import SFAFMA_T
# from toolbox.models.lyz.paper3.resnet_resnet import paper3
# from toolbox.models.lyz.paper3.resnet_pvt import paper3
# from toolbox.models.lyz.paper3.resnet_segformer import paper3
# from toolbox.models.lyz.paper3.resnet_p2t import paper3
from toolbox.models.DFMNet.net import DFMNet
if DATASET == 'Potsdam':
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    # print(all_files)
    # all_ids = ["".join(f.split('')[5:7]) for f in all_files]
    all_ids = ["".join(f.split('/')[-1].split('_')[2] + '_' + f.split('/')[-1].split('_')[3]) for f in all_files]
    # print(all_ids)
elif DATASET == 'Vaihingen':
    # all_ids =
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]

# train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
# test_ids = list(set(all_ids) - set(train_ids))

# Exemple of a train/test split on Vaihingen :
if DATASET == 'Potsdam':
    train_ids = ['2_10', '3_10', '3_11', '3_12', '4_11', '4_12', '5_10', '5_12', '6_8', '6_9', '6_10', '6_11', '6_12', '7_7', '7_9', '7_11', '7_12']
    test_ids = ['2_11', '2_12', '4_10']   # '2_11', '2_12', '4_10', '5_11', '6_7', '7_8', '7_10'
elif DATASET == 'Vaihingen':
    train_ids = ['1', '3', '5', '7', '13', '17', '21', '23', '26', '32', '37']
    test_ids = [ '11', '15', '28','30', '34'] # '11', '15', '28',
elif DATASET =="Dataset4EO":
    val_dataloader = DataLoader(Dataset4EO_test(r'/media/panyi/dd67419f-4b06-4cc1-a57e-0706577875076/RSUSS1/', 'test'),
                                batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


def test(net , test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    print("Tiles for training: ", train_ids)
    print("Tiles for testing: ", test_ids)
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_ndsm = (1 / 255 * np.asarray(io.imread(NDSM_FOLODER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)

    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        net.eval()

        for img, ndsm, gt, gt_e in tqdm(zip(test_images, test_ndsm, test_labels, eroded_labels), total=len(test_ids), leave=None):
            pred = np.zeros(img.shape[:2] + (N_CLASSES, ))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=None)):
                # Display in progress results
                # Build the tensor
                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                ndsm_patches = [np.copy(ndsm[np.newaxis, x:x+w, y:y+h]) for x, y, w, h in coords]
                ndsm_patches = np.asarray(ndsm_patches)
                ndsm_patches = Variable(torch.from_numpy(ndsm_patches).cuda(), volatile=True)
                # ndsm_patches = torch.repeat_interleave(ndsm_patches, 3, dim=1)

                # input = torch.cat((image_patches, ndsm_patches), dim=1)
                # outs = net(image_patches)
                outs = net(image_patches, ndsm_patches)
                outs = outs[0].data.cpu().numpy()
                # outs = outs.data.cpu().numpy()


                # Do the inference
                # outs = net(image_patches, ndsm_patches)
                # outs = net(image_patches)
                # outs = outs[0].data.cpu().numpy()

                # outimg = convert_to_color_(outs)
                # io.imsave('./result/inference_tile_{}.png'.format(i), outimg)

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):

                    out = out.transpose((1, 2, 0))
                    pred[x:x+w, y:y+h] += out
                del(outs)

            pred = np.argmax(pred, axis=-1)
            print('pred_1',pred.shape)
            all_preds.append(pred)
            all_gts.append(gt_e)

            metrics(pred.ravel(), gt_e.ravel())
            accuary = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())

    if all:
        # return accuary, all_preds, all_gts
        return accuary, all_preds, all_gts
    else:
        return accuary


if __name__ == "__main__":
    net = student_model().cuda()
    # net = DFMNet().cuda()
    net.load_state_dict(torch.load('/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/models/SXY/run/2024-10-09-08-39(SXY_kd_P)/_last.pth'))
    # net.load_state_dict(torch.load('/media/panyi/dd67419f-4b06-4cc1-a57e-0706577875076/weight/tem/SFAFMA_T(paper2)_28_last-Vaihingen-loss.pth'))
    # net.load_state_dict(torch.load('/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/models/DGPInet/run/2024-10-02-12-45(DGPINet_Vaihingen)/_best.pth'))
    # net.load_state_dict(torch.load('/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/models/xj/MyJokerbase_5_3_1_student-Vaihingen-loss.pth'))
    # net.load_state_dict(torch.load('./weight/PPNet_10-Dataset4EO-loss.pth'))
    # net.load_state_dict(torch.load('./weight/EAEFNet_1-Potsdam-loss.pth'))
    # net.load_state_dict(torch.load('./weight/SAFA-Potsdam-loss.pth'))
    # net.load_state_dict(torch.load('./toolbox/models/lyz/weight/PPNet_S_KD(CE[S,T]_AT))-Vaihingen-loss.pth'))
    vt800 = '/home/panyi/桌面/pytorch_segementation_Remote_Sensing/toolbox/models/SXY/run/2024-10-09-08-39(SXY_kd_P)/result_val/'
    # vt800 = '/home/panyi/桌面/pytorch_segementation_Remote_Sensing/results/Vaihingen/DGPINet_kd_1/'
    path1 = vt800
    isExist = os.path.exists(vt800)
    if not isExist:
        os.makedirs(vt800)
    else:
        print('path1 exist')

    _, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
    # _, all_preds = test_RSUSS(net, all=True, stride=32)
    # print(accuary)
    for p, id_ in zip(all_preds, test_ids):
        img = convert_to_color_(p)
        # io.imsave('./results/Vaihingen/MyConv_resnet_T(backbone)/top_mosaic_09cm_area{}.png'.format(id_), img)
        # io.imsave('./results/Vaihingen/paper3(Res_50_PVTB3)/top_mosaic_09cm_area{}.png'.format(id_), img)
        io.imsave(vt800 + 'top_mosaic_09cm_area{}.png'.format(id_), img)
        # io.imsave('./results/Potsdam/MobileNet_KD(all)_1/top_mosaic_09cm_area{}.png'.format(id_), img)
        # io.imsave('./results/Dataset4EO/PPNet_10/{}.png'.format(id_), img)
        # io.imsave('./results/Potsdam/SFAFMA/top_mosaic_09cm_area{}.png'.format(id_), img)
        # io.imsave('./toolbox/models/lyz/results/Potsdam/Mobilenet_S_at_1/top_mosaic_09cm_area{}.png'.format(id_), img)

# Mobilenet_S(image)_last
# MyConv_resnet_T(image)
# 25

# MobileNet_KD(all) 21 22 26 19 17

# MobileNet_KD(all)_2_last

# SFAFMA_T(paper2)_30_last  PPNet_S_KD(paper)_1

# 24 22