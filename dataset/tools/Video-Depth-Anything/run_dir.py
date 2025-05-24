import os
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value, set_start_method
import time


def process_video(input_video, output_dir, args, video_depth_anything, DEVICE):
    import os
    from utils.dc_utils import read_video_frames, save_video
    video_name = os.path.basename(input_video)
    depth_vis_path = os.path.join(output_dir, video_name)
    if os.path.exists(depth_vis_path):
        return
    frames, target_fps = read_video_frames(input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(
        frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

def worker(video_list, gpu_id, args, model_configs, shared_counter):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    DEVICE = torch.device('cuda:0')

    from video_depth_anything.video_depth import VideoDepthAnything
    from utils.dc_utils import read_video_frames, save_video

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(
        torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    for input_video in video_list:
        process_video(input_video, args.output_dir, args, video_depth_anything, DEVICE)
        with shared_counter.get_lock():
            shared_counter.value += 1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Batch Video Depth Anything - Multi-GPU')
    parser.add_argument('--input_dir', type=str, default='./assets/example_videos', help='Input directory for videos')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--target_fps', type=int, default=-1)
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--procs_per_gpu', type=int, default=1,
                    help='Number of workers per GPU')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    import torch
    num_gpus = torch.cuda.device_count()
    print("num_gpus", num_gpus)

    video_exts = ['.mp4', '.MP4']
    all_videos = [os.path.join(args.input_dir, f)
                  for f in os.listdir(args.input_dir)
                  if os.path.splitext(f)[-1] in video_exts]

    total_videos = len(all_videos)

    procs_per_gpu = args.procs_per_gpu
    total_processes = num_gpus * procs_per_gpu

    # videos_per_worker = np.array_split(all_videos, total_processes)
    def split_videos_round_robin(all_videos, num_workers):
        split = [[] for _ in range(num_workers)]
        for i, v in enumerate(all_videos):
            split[i % num_workers].append(v)
        return split

    videos_per_worker = split_videos_round_robin(all_videos, total_processes)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    shared_counter = Value('i', 0)

    processes = []
    for idx in range(total_processes):
        gpu_id = idx // procs_per_gpu
        p = Process(target=worker,
                    args=(videos_per_worker[idx], gpu_id, args, model_configs, shared_counter))
        p.start()
        processes.append(p)

    with tqdm(total=total_videos, desc='Total Progress', ncols=100) as pbar:
        last = 0
        while pbar.n < total_videos:
            current = shared_counter.value
            pbar.update(current - last)
            last = current
            time.sleep(0.01)
        pbar.n = total_videos
        pbar.refresh()

    for p in processes:
        p.join()

    print('All videos processed.')