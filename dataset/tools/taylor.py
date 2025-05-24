#!/usr/bin/env python3
# ---------------------------------------------------------------
# GPU-accelerated Taylor-series color-encoding video converter
# Accelerations: streaming NVENC output, pinned-memory async upload,
# FP16 math, incremental difference-table update.
# ---------------------------------------------------------------

from __future__ import annotations
import argparse, math, subprocess, sys
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import torch
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def get_device(force_cpu: bool = False) -> torch.device:
    """Get CUDA device if available, else fallback to CPU. Force CPU if specified."""
    return torch.device("cpu") if (force_cpu or not torch.cuda.is_available()) else torch.device("cuda")

@torch.jit.script
def preprocess_tensor(t: torch.Tensor) -> torch.Tensor:          # -> uint8, on-device
    """Clamp and normalize tensor, then convert to uint8 for video writing."""
    t = torch.clamp(t, min=0)
    mx = torch.max(t)
    if float(mx) > 0:
        t = t * (255.0 / mx)
    return t.to(torch.uint8)

def open_ffmpeg_writer(path: Path | str, fps: int, h: int, w: int, use_nvenc=True):
    """
    Launch an FFmpeg subprocess to encode and save video frames.
    Input is raw RGB24 via stdin.
    """
    codec = "h264_nvenc" if use_nvenc else "libx264"
    cmd = (
        "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 "
        f"-s:v {w}x{h} -r {fps} -i - "
        f"-c:v {codec} -preset p5 -pix_fmt yuv420p {path}"
    ).split()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def to_gpu_async(frame_bgr, device, stream, dtype):
    """Convert BGR numpy array to grayscale, then asynchronously upload to GPU as float tensor."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    t_cpu = torch.from_numpy(gray).pin_memory().to(dtype=dtype).div_(255.0)  # (H,W)
    with torch.cuda.stream(stream):
        return t_cpu.to(device, non_blocking=True)

@torch.jit.script
def update_fd_inline(fd: torch.Tensor, length: int):
    """
    Update the forward-difference table (fd) in-place.
    fd shape: (l, l, H, W); update last column.
    """
    j = length - 1
    for k in range(1, length):
        fd[k, j - k].copy_(fd[k - 1, j - k + 1] - fd[k - 1, j - k])

# ────────────────────────────────────────────────────────────────
# Single-video fast converter
# ────────────────────────────────────────────────────────────────
def video_convert_fast(
    vid_path: str | Path,
    out_path: str | Path,
    *,
    terms: int = 3,
    tprime: int = 0,
    dtype: torch.dtype = torch.float16,
    device: torch.device | None = None,
) -> None:
    """
    Convert a video file using Taylor-series color encoding, optionally using GPU acceleration.
    """
    device = get_device() if device is None else device
    if tprime <= terms + 3:
        tprime = terms + 3

    cap = cv2.VideoCapture(str(vid_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {vid_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    vlen   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Prime the window ─────────────────────────────────────────
    length = terms + 3  # window size l
    frames: List[torch.Tensor] = []
    stream = torch.cuda.Stream() if device.type == "cuda" else None
    for _ in range(length):
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError("Unexpected EOF while priming buffer")
        t = to_gpu_async(fr, device, stream, dtype) if stream else \
            torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)).to(device, dtype).div_(255.0)
        frames.append(t)
    if stream: torch.cuda.current_stream().wait_stream(stream)

    h, w = frames[0].shape
    fd = torch.zeros((length, length, h, w), dtype=dtype, device=device)
    fd[0, :length] = torch.stack(frames)

    # Initial forward-difference table
    for k in range(1, length):
        fd[k, :length - k] = torch.diff(fd[k - 1, :length - k + 1], dim=0)

    factorial = torch.tensor([math.factorial(i) for i in range(terms)],
                             dtype=dtype, device=device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_ffmpeg_writer(out_path, int(round(fps_in)), h, w,
                                use_nvenc=(device.type=="cuda"))

    total = vlen - tprime + 1
    for idx in range(total):
        base  = fd[0, 0]                           # f(x0)
        xa    = fd[0, :length] - base              # (l,H,W)
        powers = torch.stack([(xa ** i) / factorial[i] for i in range(terms)])  # (k,l,H,W)
        powers_sum = torch.sum(powers, dim=1)      # (k,H,W)

        t1 = torch.sum(powers_sum * fd[1 : terms + 1, 0], dim=0)
        t2 = torch.sum(powers_sum * fd[2 : terms + 2, 0], dim=0)
        t3 = torch.sum(powers_sum * fd[3 : terms + 3, 0], dim=0)

        rgb = torch.stack((t1, t2, t3), dim=-1) / tprime
        writer.stdin.write(preprocess_tensor(rgb).cpu().numpy().tobytes())

        # Slide the window
        if idx + 1 == total:
            break
        ok, fr = cap.read()
        if not ok: break
        fd = torch.roll(fd, shifts=-1, dims=1)     # shift columns left
        if stream:
            fd[0, length - 1] = to_gpu_async(fr, device, stream, dtype)
            torch.cuda.current_stream().wait_stream(stream)
        else:
            fd[0, length - 1] = torch.from_numpy(
                cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ).to(device, dtype).div_(255.0)
        update_fd_inline(fd, length)

    writer.stdin.close()
    writer.wait()
    cap.release()

# ────────────────────────────────────────────────────────────────
# Batch helpers
# ────────────────────────────────────────────────────────────────
def _process_video(args: Tuple[Path, Path, Path, int, int, str, torch.dtype, bool]):
    """
    Helper for multiprocessing: processes a single video, catches exceptions.
    """
    vid, in_root, out_root, terms, tprime, codec, dtype, force_cpu = args
    rel  = vid.relative_to(in_root)
    outv = (out_root / rel).with_suffix(".mp4")
    try:
        video_convert_fast(vid, outv, terms=terms, tprime=tprime,
                           dtype=dtype, device=get_device(force_cpu))
    except Exception as exc:
        return f"[ERROR] {vid}: {exc}"
    return None

def iter_videos(root: Path, exts: Iterable[str], recurse: bool):
    """
    Recursively or non-recursively yield video files with given extensions under root.
    """
    for ext in exts:
        yield from (root.rglob(f"*{ext}") if recurse else root.glob(f"*{ext}"))

def batch_process(
    input_dir: Path, output_dir: Path,
    *, terms=3, tprime=0, extensions=(".mp4",".avi",".mov",".mkv"),
    recursive=False, workers=1, dtype=torch.float16, force_cpu=False,
):
    """
    Batch process all video files in a directory (optionally recursively) with multiprocessing.
    """
    vids = list(iter_videos(input_dir, extensions, recursive))
    if not vids:
        print("[WARN] no video files found."); return
    ctx = get_context("spawn")
    pool_args = [
        (v, input_dir, output_dir, terms, tprime, "h264_nvenc", dtype, force_cpu)
        for v in vids
    ]
    if workers <= 1:
        for a in tqdm(pool_args, desc="Batch", unit="video", ncols=80):
            msg = _process_video(a)
            if msg: tqdm.write(msg)
    else:
        with ctx.Pool(min(workers, cpu_count())) as pool:
            with tqdm(total=len(pool_args), desc="Batch", unit="video", ncols=80) as bar:
                for msg in pool.imap_unordered(_process_video, pool_args):
                    if msg: tqdm.write(msg)
                    bar.update()

# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Taylor-series color encoding (GPU-accelerated)")
    ap.add_argument("input",  help="input video or directory")
    ap.add_argument("output", help="output video or directory")
    ap.add_argument("-t", "--terms", type=int, default=3, help="Taylor series terms (default: 3)")
    ap.add_argument("--tprime", type=int, default=0, help="normalization constant (default: terms+3)")
    ap.add_argument("--fp32", action="store_true", help="force FP32 instead of FP16")
    ap.add_argument("-r", "--recursive", action="store_true", help="scan subfolders recursively")
    ap.add_argument("-w", "--workers", type=int, default=1, help="Number of CPU decode workers (default: 1)")
    ap.add_argument("--cpu", action="store_true", help="force CPU mode")
    args = ap.parse_args(argv)

    inp = Path(args.input).expanduser().resolve()
    outp = Path(args.output).expanduser().resolve()

    if inp.is_dir():
        batch_process(inp, outp, terms=args.terms, tprime=args.tprime,
                      recursive=args.recursive, workers=args.workers,
                      dtype=torch.float32 if args.fp32 else torch.float16,
                      force_cpu=args.cpu)
    else:
        if outp.is_dir():
            outp = (outp / inp.name).with_suffix(".mp4")
        video_convert_fast(inp, outp, terms=args.terms, tprime=args.tprime,
                           dtype=torch.float32 if args.fp32 else torch.float16,
                           device=get_device(args.cpu))

if __name__ == "__main__":
    main()
