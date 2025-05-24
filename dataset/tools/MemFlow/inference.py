from __future__ import print_function, division
import argparse
from loguru import logger as loguru_logger
import random
from core.Networks import build_network
import sys
sys.path.append('core')
from PIL import Image
import os
import numpy as np
import torch
import cv2
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from inference import inference_core_skflow as inference_core


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def inference_video(cfg):
    # 加载模型
    model = build_network(cfg).cuda()
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in list(ckpt_model.keys()):
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)
    model.eval()

    # 打开视频
    cap = cv2.VideoCapture(cfg.video)
    assert cap.isOpened(), f"Failed to open video: {cfg.video}"

    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    cap.release()
    print(f"Total frames: {len(frame_list)}")

    # 视频保存器初始化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frame_list[0].shape[:2]
    writer = cv2.VideoWriter(cfg.out_mp4, fourcc, 25, (w, h))  # 25 FPS，可根据实际设置

    # 逐对帧处理
    padder = None
    processor = inference_core.InferenceCore(model, config=cfg)
    flow_prev = None

    for idx in range(len(frame_list) - 1):
        img1 = frame_list[idx]
        img2 = frame_list[idx + 1]

        # 确保输入为3通道
        if img1.ndim == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
        if img2.ndim == 2:
            img2 = np.tile(img2[..., None], (1, 1, 3))
        img1 = img1[..., :3]
        img2 = img2[..., :3]

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in [img1, img2]]
        images = torch.stack(imgs).unsqueeze(0).cuda()  # 1,2,3,H,W
        if padder is None:
            padder = InputPadder(images.shape)
        images = padder.pad(images)
        images = 2 * (images / 255.0) - 1.0

        # 网络推理
        flow_low, flow_pre = processor.step(
            images, end=(idx == len(frame_list) - 2),
            add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev
        )
        flow_pre = padder.unpad(flow_pre[0]).cpu()
        if 'warm_start' in cfg and cfg.warm_start:
            flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        # 可视化光流结果
        flow_img = flow_viz.flow_to_image(flow_pre.permute(1, 2, 0).numpy())
        # 写入视频（可以换成img2/光流混合等）
        writer.write(flow_img.astype(np.uint8))

    writer.release()
    print(f"Output video saved at {cfg.out_mp4}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet', choices=['MemFlowNet', 'MemFlowNet_T'], help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--out_mp4', required=True, help='Output mp4 file')

    args = parser.parse_args()
    if args.name == "MemFlowNet":
        if args.stage == 'things':
            from configs.things_memflownet import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet import get_cfg
        elif args.stage == 'spring_only':
            from configs.spring_memflownet import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet import get_cfg
        else:
            raise NotImplementedError
    elif args.name == "MemFlowNet_T":
        if args.stage == 'things':
            from configs.things_memflownet_t import get_cfg
        elif args.stage == 'things_kitti':
            from configs.things_memflownet_t_kitti import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet_t import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet_t import get_cfg
        else:
            raise NotImplementedError

    cfg = get_cfg()
    cfg.update(vars(args))

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    inference_video(cfg)
