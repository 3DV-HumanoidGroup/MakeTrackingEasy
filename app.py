"""
Gradio Demo: Human → G1 Robot Motion Retargeting

上传 AMASS 格式 NPZ 文件，推理并展示 3D 骨骼动画，导出 bmimic 格式。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
# PyTorch 2.6 兼容 patch
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, weights_only=kwargs.pop('weights_only', False), **kwargs)

import numpy as np
import gradio as gr

from inference import load_all, infer_single
from visualize import create_skeleton_animation
from convert_bmimic import convert_to_bmimic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, 'assets', 'g1_mocap_29dof.xml')

# ---------- 启动时加载模型 ----------
print("Loading model...")
model, smplx_model, betas, smplx_mean, smplx_std, g1_mean, g1_std, device = load_all()
print(f"Model loaded on {device}")


def predict(input_file):
    """推理并可视化。"""
    if input_file is None:
        return None, None, "Please upload an AMASS NPZ file."

    file_path = input_file.name if hasattr(input_file, 'name') else input_file

    if not file_path.endswith('.npz'):
        return None, None, "Error: only .npz files are supported."

    result, timing = infer_single(
        file_path, model, smplx_model, betas,
        smplx_mean, smplx_std, g1_mean, g1_std,
        device, apply_filter=True,
    )

    if result is None:
        return None, None, "Error: sequence too short (< 4 frames)."

    # 转换为 bmimic npz 格式 (30fps → 50fps)
    bmimic_data = convert_to_bmimic(result, XML_PATH, device, tgt_fps=50.0, src_fps=30.0)
    output_path = os.path.join(tempfile.gettempdir(), 'g1_motion.npz')
    np.savez(output_path, **bmimic_data)

    # 生成 3D 骨骼动画
    fig = create_skeleton_animation(
        result['dof'], result['root_rot_quat'], result['root_trans'],
    )

    T = result['dof'].shape[0]
    T_bmimic = bmimic_data['joint_pos'].shape[0]
    info = (
        f"Frames: {T} ({T / 30:.1f}s @ 30 FPS) → bmimic: {T_bmimic} frames @ 50 FPS\n"
        f"Preprocess: {timing['preprocess']:.2f}s | "
        f"Inference: {timing['infer']:.2f}s | "
        f"Postprocess: {timing['postprocess']:.2f}s | "
        f"Total: {timing['total']:.2f}s\n"
        f"Output keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, "
        f"body_lin_vel_w, body_ang_vel_w"
    )

    return fig, output_path, info


# ---------- Gradio UI ----------
demo = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload AMASS NPZ file", file_types=[".npz"]),
    outputs=[
        gr.Plot(label="G1 Robot Skeleton Animation"),
        gr.File(label="Download Result (bmimic NPZ)"),
        gr.Textbox(label="Info", lines=3),
    ],
    title="Human → G1 Robot Motion Retargeting",
    description=(
        "Upload human motion capture data in AMASS format (.npz), "
        "and get Unitree G1 humanoid robot motion.\n\n"
        "**Input format**: AMASS NPZ with fields `trans/root_orient/pose_body` "
        "(or `transl/global_orient/body_pose`), optionally `mocap_frame_rate`.\n\n"
        "**Output**: bmimic NPZ (50 FPS) with `joint_pos`, `joint_vel`, "
        "`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`."
    ),
    examples=[["examples/sample_motion.npz"]] if os.path.exists("examples/sample_motion.npz") else None,
    cache_examples=False,
)

if __name__ == '__main__':
    demo.launch()
