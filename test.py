import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TaskFusion_dataset import Fusion_dataset
import numpy as np
from PIL import Image
from model import RegionWareNet
from torchvision import transforms
import util
from models import U2NETP
from fastsam import PreFastSAM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(777)


def RGB2YCbCr(x):
    R, G, B = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 0.564 * (B - Y) + 0.5
    Cr = 0.713 * (R - Y) + 0.5
    return torch.cat([Y, Cb, Cr], dim=1)


def YCbCr2RGB(x):
    Y, Cb, Cr = x[:, 0:1, :, :], x[:, 1:2, :, :] - 0.5, x[:, 2:3, :, :] - 0.5
    R = Y + 1.403 * Cr
    G = Y - 0.344 * Cb - 0.714 * Cr
    B = Y + 1.773 * Cb
    return torch.cat([R, G, B], dim=1).clamp(0, 1)


def run_fusion(type='val'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("...load U2NETP---4.7 MB")
    preU2NETP = U2NETP(3, 1).to(device)
    u2net_path = './checkpoints/u2netp.pth'
    preU2NETP.load_state_dict(torch.load(u2net_path, map_location=device))
    preU2NETP.eval()

    model_sam = PreFastSAM('./checkpoints/FastSAM-x.pt')

    regionWareNet = RegionWareNet().to(device)
    fusion_model_path = 'checkpoints/RAWFusion.pth'
    regionWareNet.load_state_dict(torch.load(fusion_model_path, map_location=device))
    regionWareNet.eval()

    fused_dir = './test/result/Fuse'
    os.makedirs(fused_dir, exist_ok=True)

    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        for it, (image_vis, image_ir, names) in enumerate(test_loader):
            print(f"Processing batch {it}, image names: {names}")
            image_vis = image_vis.to(device)
            image_ir = image_ir.to(device)

            image_vis_ycbcr = RGB2YCbCr(image_vis)
            vis_input_Y = image_vis_ycbcr[:, 0:1, :, :]

            vis_input_ori = (image_vis * 255.0).cpu().numpy().transpose(0, 2, 3, 1)
            img_seg = [img.astype(np.uint8) for img in vis_input_ori]
            masks_sam = model_sam(img_seg, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

            img_sal = transforms.functional.resize(image_vis, (320, 320))
            img_sal = transforms.functional.normalize(img_sal, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            d, _, _, _, _, _, _ = preU2NETP(img_sal)
            d = F.interpolate(util.normPRED(d[:, 0:1, :, :]), size=image_vis.shape[2:], mode='bilinear',
                              align_corners=True)

            masks = util.get_mask(masks_sam, d)
            masks_sal = torch.stack(masks, dim=0).to(device)

            vis_qg_pred, vis_bg_pred, ir_bg_pred, ir_qg_pred, fuseImageY = regionWareNet(
                masks_sal, vis_input_Y, image_ir
            )

            Cb = image_vis_ycbcr[:, 1:2, :, :]
            Cr = image_vis_ycbcr[:, 2:3, :, :]

            fusion_ycbcr = torch.cat([fuseImageY, Cb, Cr], dim=1)

            fusion_rgb = YCbCr2RGB(fusion_ycbcr)

            fused_image = (fusion_rgb.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            for k in range(len(names)):
                img_pil = Image.fromarray(fused_image[k])
                save_path = os.path.join(fused_dir, names[k])
                img_pil.save(save_path)
                print(f'Fusion {save_path} successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fusion with pytorch')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()

    run_fusion('val')
