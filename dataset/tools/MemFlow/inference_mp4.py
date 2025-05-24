import os
import sys
import argparse
import multiprocessing as mp

def list_pending_videos(input_dir, output_dir):
    mp4s = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')])
    pendings = []
    for f in mp4s:
        out_mp4 = os.path.join(output_dir, os.path.splitext(f)[0] + ".mp4")
        if not os.path.exists(out_mp4):
            pendings.append(f)
    return pendings

def worker(gpu_id, task_queue, args, progress_queue):
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.CRITICAL)
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import random
    import numpy as np
    import cv2
    sys.path.append('core')
    from core.Networks import build_network
    from utils import flow_viz
    from utils.utils import InputPadder, forward_interpolate
    from inference import inference_core_skflow as inference_core

    def read_video_frames(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    @torch.no_grad()
    def process_one_video_to_mp4(cfg, model, mp4_path, out_mp4_path):
        if os.path.exists(out_mp4_path):
            return
        print(mp4_path)
        imgs = read_video_frames(mp4_path)
        if len(imgs) < 2:
            return
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        images = torch.stack(imgs)
        processor = inference_core.InferenceCore(model, config=cfg)
        images = images.cuda().unsqueeze(0)
        padder = InputPadder(images.shape)
        images = padder.pad(images)
        images = 2 * (images / 255.0) - 1.0

        flow_prev = None
        results = []
        for ti in range(images.shape[1] - 1):
            flow_low, flow_pre = processor.step(
                images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
            flow_pre = padder.unpad(flow_pre[0]).cpu()
            results.append(flow_pre)
            if 'warm_start' in cfg and cfg.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        vis_frames = []
        for idx in range(len(results)):
            flow_img = flow_viz.flow_to_image(results[idx].permute(1, 2, 0).numpy())
            vis_frames.append(flow_img)
        vis_frames.append(vis_frames[-1].copy())
        os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
        h, w, _ = vis_frames[0].shape
        writer = cv2.VideoWriter(out_mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
        for img in vis_frames:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

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

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    model = build_network(cfg).cuda()
    if cfg.restore_ckpt is not None:
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in list(ckpt_model.keys()):
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)
    model.eval()

    while True:
        try:
            vfile = task_queue.get(timeout=3)
        except Exception:
            break
        vpath = os.path.join(args.input_dir, vfile)
        out_mp4 = os.path.join(args.output_dir, os.path.splitext(vfile)[0] + ".mp4")
        try:
            process_one_video_to_mp4(cfg, model, vpath, out_mp4)
        except RuntimeError as e:
            import traceback
            if 'CUDA out of memory' in str(e):
                with open("oom.log", "a") as f:
                    f.write(f"OOM: {vfile} {str(e)}\n")
            else:
                with open("error.log", "a") as f:
                    f.write(f"Error: {vfile} {traceback.format_exc()}\n")
        except Exception as e:
            import traceback
            with open("error.log", "a") as f:
                f.write(f"Error: {vfile} {traceback.format_exc()}\n")
        progress_queue.put(1)

def main():
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet', choices=['MemFlowNet', 'MemFlowNet_T'])
    parser.add_argument('--stage', required=True)
    parser.add_argument('--restore_ckpt', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    import torch
    from tqdm import tqdm

    n_gpu = torch.cuda.device_count()
    pendings = list_pending_videos(args.input_dir, args.output_dir)
    total = len(pendings)
    if total == 0:
        print("No pending videos.")
        return

    manager = mp.Manager()
    task_queue = manager.Queue()
    progress_queue = manager.Queue()
    for vfile in pendings:
        task_queue.put(vfile)

    processes = []
    for i in range(n_gpu):
        p = mp.Process(target=worker, args=(i, task_queue, args, progress_queue))
        p.start()
        processes.append(p)

    from time import sleep
    pbar = tqdm(total=total, desc="Total", ncols=66, position=0, leave=True, dynamic_ncols=True)
    finished = 0
    while finished < total:
        try:
            progress_queue.get(timeout=3)
            finished += 1
            pbar.update(1)
        except Exception:
            pass

    pbar.close()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()