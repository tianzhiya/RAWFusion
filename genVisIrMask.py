import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from models import U2NETP
from fastsam import PreFastSAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIS_DIR = "./QianJInMask/images/vis"
IR_DIR = "./QianJInMask/images/ir"
SAVE_VIS_FG_DIR = "./QianJInMask/output/vis/fg"
SAVE_VIS_BG_DIR = "./QianJInMask/output/vis/bg"
SAVE_IR_FG_DIR = "./QianJInMask/output/ir/fg"
SAVE_IR_BG_DIR = "./QianJInMask/output/ir/bg"

for d in [SAVE_VIS_FG_DIR, SAVE_VIS_BG_DIR, SAVE_IR_FG_DIR, SAVE_IR_BG_DIR]:
    os.makedirs(d, exist_ok=True)

sal_net = U2NETP(3, 1)
sal_net.load_state_dict(torch.load("./checkpoints/u2netp.pth", map_location=DEVICE))
sal_net.to(DEVICE).eval()

sal_transform = transforms.Compose([
    transforms.Resize((320, 320)),
])

sam_model = PreFastSAM("./checkpoints/FastSAM-x.pt")


def process_image_pair(vis_path, ir_path):
    name = os.path.basename(vis_path)

    img_bgr = cv2.imread(vis_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = sal_transform(img_tensor).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d, _, _, _, _, _, _ = sal_net(img_tensor)
        d = F.interpolate(d[:, 0:1, :, :], size=(H, W), mode='bilinear', align_corners=True)
        d = torch.sigmoid(d)
    masks_sam = sam_model([img_rgb], device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    fg_mask = torch.zeros((H, W), dtype=torch.float32)
    masks = masks_sam[0].masks
    if masks is not None and masks.data.shape[0] > 0:
        for i in range(masks.data.shape[0]):
            m = masks.data[i].cpu()
            fg_mask = torch.max(fg_mask, m)

    d = d.squeeze(0).squeeze(0).cpu()
    fg_mask = torch.clamp(fg_mask * d, 0, 1)
    fg_mask = (fg_mask > 0.5).float()  # 二值化

    fg_vis = img_rgb * fg_mask[:, :, None].numpy()
    bg_vis = img_rgb * (1 - fg_mask[:, :, None].numpy())

    cv2.imwrite(os.path.join(SAVE_VIS_FG_DIR, name), cv2.cvtColor(fg_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(SAVE_VIS_BG_DIR, name), cv2.cvtColor(bg_vis, cv2.COLOR_RGB2BGR))

    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    fg_ir = (ir_img * fg_mask.numpy()).astype(np.uint8)
    bg_ir = (ir_img * (1 - fg_mask.numpy())).astype(np.uint8)

    cv2.imwrite(os.path.join(SAVE_IR_FG_DIR, name), fg_ir)
    cv2.imwrite(os.path.join(SAVE_IR_BG_DIR, name), bg_ir)

    print(f"Processed {name}: vis fg/bg and ir fg/bg saved.")


if __name__ == "__main__":
    vis_list = [f for f in os.listdir(VIS_DIR) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    for vis_file in vis_list:
        vis_path = os.path.join(VIS_DIR, vis_file)
        ir_path = os.path.join(IR_DIR, vis_file)
        if os.path.exists(ir_path):
            process_image_pair(vis_path, ir_path)
    print("All image pairs processed.")
