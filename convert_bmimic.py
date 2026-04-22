"""Convert NMR inference output to bmimic npz format.

Output npz keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from src.utils.kinematics_model import KinematicsModel

G1_BMIMIC_BODY_NAMES = [
    "pelvis",               "left_hip_pitch_link",   "right_hip_pitch_link",
    "waist_yaw_link",       "left_hip_roll_link",    "right_hip_roll_link",
    "waist_roll_link",      "left_hip_yaw_link",     "right_hip_yaw_link",
    "torso_link",           "left_knee_link",         "right_knee_link",
    "left_shoulder_pitch_link",  "right_shoulder_pitch_link",
    "left_ankle_pitch_link",     "right_ankle_pitch_link",
    "left_shoulder_roll_link",   "right_shoulder_roll_link",
    "left_ankle_roll_link",      "right_ankle_roll_link",
    "left_shoulder_yaw_link",    "right_shoulder_yaw_link",
    "left_elbow_link",           "right_elbow_link",
    "left_wrist_roll_link",      "right_wrist_roll_link",
    "left_wrist_pitch_link",     "right_wrist_pitch_link",
    "left_wrist_yaw_link",       "right_wrist_yaw_link",
]

# inference 输出 DOF → MuJoCo XML joint 顺序
JOINT_MAPPING = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3, 9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
    18, 25,
    19, 26,
    20, 27,
    21, 28,
]


def resample_motion(root_pos, root_rot_wxyz, dof_pos, src_fps, tgt_fps=50.0):
    """Resample motion from src_fps to tgt_fps using linear/SLERP interpolation."""
    if abs(src_fps - tgt_fps) < 0.5:
        return root_pos, root_rot_wxyz, dof_pos
    N = root_pos.shape[0]
    t_src = np.arange(N) / src_fps
    duration = t_src[-1]
    M = int(round(duration * tgt_fps)) + 1
    t_tgt = np.clip(np.arange(M) / tgt_fps, 0.0, duration)

    root_pos_new = np.stack(
        [np.interp(t_tgt, t_src, root_pos[:, i]) for i in range(3)], axis=1
    ).astype(np.float32)
    dof_pos_new = np.stack(
        [np.interp(t_tgt, t_src, dof_pos[:, i]) for i in range(dof_pos.shape[1])], axis=1
    ).astype(np.float32)

    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    slerp = Slerp(t_src, Rotation.from_quat(root_rot_xyzw))
    root_rot_wxyz_new = slerp(t_tgt).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)

    return root_pos_new, root_rot_wxyz_new, dof_pos_new


def build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, fps, km):
    """Convert robot motion to bmimic npz format dict."""
    N = root_pos.shape[0]
    device = km._device

    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    root_pos_t = torch.from_numpy(root_pos).float().to(device)
    root_rot_t = torch.from_numpy(root_rot_xyzw).float().to(device)
    dof_pos_t_lab = torch.from_numpy(dof_pos).float().to(device)

    dof_pos_t = torch.zeros((N, 29), dtype=torch.float32, device=device)
    dof_pos_t[:, JOINT_MAPPING] = dof_pos_t_lab

    body_pos_all, body_rot_all = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    body_pos_all = body_pos_all.cpu().numpy()
    body_rot_all = body_rot_all.cpu().numpy()

    # XY origin at first frame
    root_pos = root_pos.copy()
    xy0 = root_pos[0, :2].copy()
    body_pos_all[:, :, :2] -= xy0
    root_pos[:, :2] -= xy0

    # Select bmimic bodies
    km_names = km.body_names
    body_idx = [km_names.index(n) for n in G1_BMIMIC_BODY_NAMES]
    B = len(body_idx)

    body_pos_w = body_pos_all[:, body_idx, :].astype(np.float32)
    body_rot_sel = body_rot_all[:, body_idx, :]             # xyzw
    body_quat_w = body_rot_sel[:, :, [3, 0, 1, 2]].astype(np.float32)  # → wxyz

    joint_pos = dof_pos.astype(np.float32)

    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) * fps
    joint_vel[-1] = joint_vel[-2]

    body_lin_vel_w = np.zeros_like(body_pos_w)
    body_lin_vel_w[:-1] = (body_pos_w[1:] - body_pos_w[:-1]) * fps
    body_lin_vel_w[-1] = body_lin_vel_w[-2]

    body_ang_vel_w = np.zeros((N, B, 3), dtype=np.float32)
    for b in range(B):
        rots = Rotation.from_quat(body_rot_sel[:, b, :])
        q_rel = rots[1:] * rots[:-1].inv()
        body_ang_vel_w[:-1, b] = (q_rel.as_rotvec() * fps).astype(np.float32)
    body_ang_vel_w[-1] = body_ang_vel_w[-2]

    return {
        "fps":            np.array([int(fps)], dtype=np.int64),
        "joint_pos":      joint_pos,
        "joint_vel":      joint_vel,
        "body_pos_w":     body_pos_w,
        "body_quat_w":    body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def convert_to_bmimic(result, xml_path, device, tgt_fps=50.0, src_fps=30.0):
    """Convert inference result dict to bmimic npz data dict.

    Args:
        result: dict with keys 'dof' (T,29), 'root_trans' (T,3), 'root_rot_quat' (T,4 wxyz)
        xml_path: path to MuJoCo XML for KinematicsModel
        device: torch device
        tgt_fps: target fps for bmimic output
        src_fps: source fps of inference output

    Returns:
        dict: bmimic format data ready for np.savez
    """
    root_pos = np.asarray(result['root_trans'], dtype=np.float32)
    root_rot_wxyz = np.asarray(result['root_rot_quat'], dtype=np.float32)
    dof_pos = np.asarray(result['dof'], dtype=np.float32)

    root_pos, root_rot_wxyz, dof_pos = resample_motion(
        root_pos, root_rot_wxyz, dof_pos, src_fps, tgt_fps
    )

    km = KinematicsModel(xml_path, device=str(device))
    return build_bmimic_data(root_pos, root_rot_wxyz, dof_pos, tgt_fps, km)
