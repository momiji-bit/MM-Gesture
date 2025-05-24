import os
import pickle
import numpy as np
import json
import torch
import zipfile
from os.path import basename

# ----------- Utility Functions -----------

def load_pkl(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def mmaction2numpy(file_path):
    """
    Load a mmaction2 result pkl file and stack all prediction scores into a numpy array.
    Each element should have a key 'pred_score' (torch.Tensor or np.array).
    """
    pkl_data = load_pkl(file_path)
    pred_scores = []
    for d in pkl_data:
        x = d['pred_score']
        # If already torch.Tensor, detach+cpu+clone; if numpy, convert to tensor
        if isinstance(x, torch.Tensor):
            pred_scores.append(x.detach().cpu().clone())
        else:
            pred_scores.append(torch.tensor(x))
    return torch.stack(pred_scores).numpy()


# ----------- Load Video List -----------

with open('test_list.json') as file_obj:
    video_names = json.load(file_obj)  # List of video ids/names

# ----------- Load Model Predictions -----------

# 1. Swin Transformer (Fine-tuned on iMiGUE)
R_25_all = mmaction2numpy('./2025/swin/swin_base/swin_last_base.pkl')
# 2. Swin Transformer (MA52 pretrain + iMiGUE fine-tune)
R_Trans_25_all = mmaction2numpy('./2025/swin/swin_small_trans/swin_last_small_trans.pkl')
# 3. Swin Transformer (Taylor Stream)
R_Tylor_25_all = mmaction2numpy('./2025/swin/swin_base_taylor/swin_best_base_taylor.pkl')
# 4. Swin Transformer (Depth)
D_25_all = mmaction2numpy('./2025/swin/swin_base_depth/swin_best_base_depth.pkl')
# 5. Swin Transformer (Optical Flow)
R_OF_25 = mmaction2numpy('./2025/swin/swin_base_of/swin_best_base_of.pkl')

# 6. PoseC3D Results (RGB+Joint, RGB+Limb)
def load_posec3d_pred(path, key1='rgb', key2='pose'):
    preds = load_pkl(path)
    rgb = np.array([x[key1] if isinstance(x, dict) else x['pred_score'][key1] for x in preds])
    pose = np.array([x[key2] if isinstance(x, dict) else x['pred_score'][key2] for x in preds])
    return rgb * 0.5 + pose * 0.5

p3d_2025_RJ = load_posec3d_pred('./2025/p3d/p3d_rgb+j/posec3d_best_rgb+joint_sorted.pkl', 'rgb', 'pose')
p3d_2025_RL = load_posec3d_pred('./2025/p3d/p3d_rgb+l_12/posec3d_latest_rgb+limb_sorted.pkl', 'rgb', 'pose')

# ----------- Ensemble Logic -----------
# Example: Uniform weighting for all Swin streams, higher weight for PoseC3D
ensemble_pred = (
    R_25_all * 0.2 +
    R_Trans_25_all * 0.2 +
    D_25_all * 0.2 +
    R_Tylor_25_all * 0.2 +
    R_OF_25 * 0.2 +
    p3d_2025_RJ * 0.5 +
    p3d_2025_RL * 0.5
)

# Take the class with the highest score for each video
out_rank = np.argmax(ensemble_pred, axis=1)

# ----------- Write Prediction to CSV and ZIP -----------

pred_dir = './'
os.makedirs(pred_dir, exist_ok=True)
csv_path = os.path.join(pred_dir, 'prediction.csv')

with open(csv_path, 'w') as out_file:
    out_file.write("Id,Target\n")
    for name, pred in zip(video_names, out_rank):
        out_file.write(f"{name},{pred}\n")

# Zip the CSV file for submission
with zipfile.ZipFile(os.path.join(pred_dir, 'prediction.zip'), "w") as zipf:
    zipf.write(csv_path, basename(csv_path))

