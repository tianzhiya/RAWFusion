import os

import numpy as np

from FusionLoss import Fusionloss
from TaskFusion_dataset import Fusion_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

random.seed(777)

import torch.nn as nn
import argparse

from model import RegionWareNet
from torchvision import transforms
import util

from models import U2NETP
from fastsam import PreFastSAM
from torch.utils.data import DataLoader
from torch.autograd import Variable


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def gradient_loss(pred, ref):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_ref = ref[:, :, :, 1:] - ref[:, :, :, :-1]
    dy_ref = ref[:, :, 1:, :] - ref[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_ref) + F.l1_loss(dy_pred, dy_ref)


def laplacian_loss(pred, ref):
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=pred.device).unsqueeze(
        0).unsqueeze(0)
    lap_pred = F.conv2d(pred, lap_kernel, padding=1)
    lap_ref = F.conv2d(ref, lap_kernel, padding=1)
    return F.l1_loss(lap_pred, lap_ref)


class VisForegroundEnhanceLoss(nn.Module):
    def __init__(self, lambda_grad=1.0, lambda_texture=1.0, lambda_bright=0.1):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_texture = lambda_texture
        self.lambda_bright = lambda_bright

    def forward(self, pred, input_fg):
        L_grad = gradient_loss(pred, input_fg)
        L_texture = laplacian_loss(pred, input_fg)
        L_bright = F.l1_loss(pred.mean(), input_fg.mean())
        return self.lambda_grad * L_grad + self.lambda_texture * L_texture + self.lambda_bright * L_bright


import torch
import torch.nn as nn
import torch.nn.functional as F


class VisBackgroundLoss(nn.Module):
    def __init__(self, lambda_g=1.0, lambda_c=0.1):
        super(VisBackgroundLoss, self).__init__()
        self.lambda_g = lambda_g
        self.lambda_c = lambda_c

    def gradient(self, x):
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad_x, grad_y

    def forward(self, F_bg_vi, I_bg_vi):
        Fgx, Fgy = self.gradient(F_bg_vi)
        Igx, Igy = self.gradient(I_bg_vi)
        L_grad = F.l1_loss(Fgx, Igx) + F.l1_loss(Fgy, Igy)

        std_F = torch.std(F_bg_vi, dim=[2, 3])
        std_I = torch.std(I_bg_vi, dim=[2, 3])
        L_std = F.l1_loss(std_F, std_I)

        # 总损失
        L_vis_bg = self.lambda_g * L_grad + self.lambda_c * L_std

        return L_vis_bg


def IRBackgroundEnhanceLoss(pred, input_bg, lambda_tv=1.0, lambda_bright=0.1):
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    L_tv = dx.abs().mean() + dy.abs().mean()

    L_bright = F.l1_loss(pred.mean(), input_bg.mean())

    loss = lambda_tv * L_tv + lambda_bright * L_bright
    return loss


def gradient_loss(pred, ref):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_ref = ref[:, :, :, 1:] - ref[:, :, :, :-1]
    dy_ref = ref[:, :, 1:, :] - ref[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_ref) + F.l1_loss(dy_pred, dy_ref)


def laplacian_loss(pred, ref):
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=pred.device).unsqueeze(
        0).unsqueeze(0)
    lap_pred = F.conv2d(pred, lap_kernel, padding=1)
    lap_ref = F.conv2d(ref, lap_kernel, padding=1)
    return F.l1_loss(lap_pred, lap_ref)


class IRForegroundEnhanceLoss(nn.Module):
    def __init__(self, lambda_grad=1.0, lambda_texture=1.0, lambda_bright=0.1):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_texture = lambda_texture
        self.lambda_bright = lambda_bright

    def forward(self, pred, input_fg):
        L_grad = gradient_loss(pred, input_fg)
        L_texture = laplacian_loss(pred, input_fg)
        L_bright = F.l1_loss(pred.mean(), input_fg.mean())
        return self.lambda_grad * L_grad + self.lambda_texture * L_texture + self.lambda_bright * L_bright


def train(config):
    preU2NETP, sal_transform = loadU2netP()
    seg_model_path = './checkpoints/FastSAM-x.pt'
    model_sam = PreFastSAM(seg_model_path)
    DEVICE = torch.device("cuda")

    regionWareNet = RegionWareNet().cuda()
    if config.load_pretrain == True:
        regionWareNet.load_state_dict(torch.load(config.pretrain_dir))

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    train_dataset = Fusion_dataset('train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)

    fusionLosss = Fusionloss()

    optimizer = torch.optim.Adam(regionWareNet.parameters(),
                                 lr=config.lr, weight_decay=config.weight_decay)

    if len(device_ids) > 1:
        regionWareNet = nn.DataParallel(regionWareNet, device_ids=device_ids)

    print('==> Training start: ')

    for epoch in range(config.num_epochs):
        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            regionWareNet.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            vis_input_Y = image_vis_ycrcb[:, :1]

            vis_input_ori = image_vis.detach().cpu().numpy()

            img_seg = []
            for i in range(vis_input_ori.shape[0]):
                img = vis_input_ori[i, 0]
                img = img * 255.0
                img = img.clip(0, 255).astype(np.uint8)

                img = np.stack([img, img, img], axis=-1)

                img_seg.append(img)

            masks_sam = model_sam(img_seg, device=DEVICE,
                                  retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, )
            img_sal = sal_transform(image_vis)
            d, _, _, _, _, _, _ = preU2NETP(img_sal)
            d = F.interpolate(util.normPRED(d[:, 0, :, :]).unsqueeze(1), size=(config.image_width, config.image_height),
                              mode='bilinear', align_corners=True)

            masks = util.get_mask(masks_sam, d)
            masks_sal = masks[0].unsqueeze(0)
            for id in range(len(masks)):
                if id > 0:
                    masks_sal = torch.cat((masks_sal, masks[id].unsqueeze(0)), dim=0)

            vis_qg_pred, vis_bg_pred, ir_bg_pred, ir_qg_pred, fuseImageY = regionWareNet(
                masks_sal, vis_input_Y, image_ir)

            optimizer.zero_grad()
            loss_fusion = fusionLosss(
                vis_input_Y, image_ir, fuseImageY
            )

            criterion = VisForegroundEnhanceLoss(lambda_grad=1.0, lambda_texture=1.0, lambda_bright=0.1)
            Y_fg = vis_input_Y * masks_sal
            visLossQg = criterion(vis_qg_pred, Y_fg)
            Y_bg = vis_input_Y * (~masks_sal)
            criterion_vis_bg = VisBackgroundLoss(
                lambda_g=1.0,
                lambda_c=0.1
            )
            visLossBg = criterion_vis_bg(vis_bg_pred, Y_bg)

            IR_qg = image_ir * (masks_sal)
            IR_bg = image_ir * (~masks_sal)
            bgloss_bg = IRBackgroundEnhanceLoss(ir_bg_pred, IR_bg, lambda_tv=1.0, lambda_bright=0.1)
            ircriterion = IRForegroundEnhanceLoss(lambda_grad=1.0, lambda_texture=1.0, lambda_bright=0.1)
            irloss_fg = ircriterion(ir_qg_pred, IR_qg)

            loss = loss_fusion[0] + visLossQg + visLossBg + bgloss_bg + irloss_fg

            loss.backward()
            optimizer.step()
            loss_print = 0
            loss_print = loss_print + loss.item()
            if epoch % 2 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                                                                                            epoch,
                                                                                            len(train_loader),
                                                                                            loss_print,
                                                                                            optimizer.param_groups[0][
                                                                                                'lr']))
                print(f"loss_fusion: {loss_fusion[0].item():.6f}")
                print(f"visLossQg : {visLossQg.item():.6f}")
                print(f"visLossBg : {visLossBg.item():.6f}")
                print(f"bgloss_bg : {bgloss_bg.item():.6f}")
                print(f"irloss_fg : {irloss_fg.item():.6f}")
                print(f"total_loss: {loss.item():.6f}")
                torch.save(regionWareNet.state_dict(), 'checkpoints/RAWFusion_{}.pth'.format(epoch))


def loadU2netP():
    import os
    model_name = 'u2netp'  # u2netp
    print("...load U2NEP---4.7 MB")
    preU2NETP = U2NETP(3, 1)
    model_dir = os.path.join(os.getcwd(), './checkpoints/', model_name + '.pth')
    sal_transform = transforms.Compose([
        transforms.Resize((320, 320), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if torch.cuda.is_available():
        preU2NETP.load_state_dict(torch.load(model_dir))
        preU2NETP.cuda()
    else:
        preU2NETP.load_state_dict(torch.load(model_dir, map_location='cpu'))
    preU2NETP.eval()
    return preU2NETP, sal_transform


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lowlight_images_path', type=str, default="./train")
    parser.add_argument('--val_images_path', type=str, default="./val")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--val_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--image_height', type=int, default=640)
    parser.add_argument('--image_width', type=int, default=480)
    parser.add_argument('--checkpoints', type=str, default="checkpoints/GPU/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="checkpoints/RAWFusion.pth")
    parser.add_argument('--val_height', type=int, default=400)
    parser.add_argument('--val_width', type=int, default=320)

    config = parser.parse_args()

    for epoch in range(0, 3):
        train(config)
