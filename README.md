# Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control (NMR)


## News

- **2026.03.24**: Release NMR paper and website.
- **2026.03.26**: Release HuggingFace live demo.
- **2026.04**: Release deployable inference code and checkpoint.

## TODOs

- [x] 2026.03.24: Release NMR paper and website.
- [x] 2026.03.26: Release HuggingFace live demo: https://huggingface.co/spaces/RayZhao/NMR
- [x] Release deployable inference code.
- [ ] Release CEPR dataset (SMPL and robot).
- [ ] Release training code.

---

## Quick Start

### 1. Install Dependencies

We recommend using conda:

```bash
conda create -n nmr python=3.10
conda activate nmr
```

Install PyTorch (adjust CUDA version as needed):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```


Install remaining dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Gradio Web Demo

```bash
python app.py
```

On first run, the model checkpoint (~518 MB) and SMPL-X body model (~104 MB) will be **automatically downloaded** from HuggingFace Hub. Subsequent runs load from cache.

Upload any AMASS `.npz` file (or use the provided examples in `examples/`) to get:
- Interactive 3D skeleton animation
- Downloadable bmimic `.npz` result file

### 3. Command-line Inference

```bash
python inference.py --src examples/sample_motion.npz --output-dir output/
```

**Batch processing** (a directory of NPZ/PKL files):

```bash
python inference.py --src /path/to/motions/ --output-dir output/
```

**Disable low-pass filter** (raw network output):

```bash
python inference.py --src examples/sample_motion.npz --output-dir output/ --no-filter
```

#### Input formats

| Format | Fields | Coordinate |
|--------|--------|-----------|
| AMASS `.npz` | `trans`, `root_orient`, `pose_body` | Z-up (auto-converted) |
| Standard `.npz` | `transl`, `global_orient`, `body_pose` | Y-up |

High frame-rate sequences (>30 FPS) are automatically downsampled to 30 FPS.

#### Output format

A bmimic `.npz` file at 50 FPS:

```python
{
    'fps':            np.ndarray (1,),          # 50
    'joint_pos':      np.ndarray (T, 29),       # joint angles [rad]
    'joint_vel':      np.ndarray (T, 29),       # joint velocities [rad/s]
    'body_pos_w':     np.ndarray (T, 30, 3),    # body positions in world frame [m]
    'body_quat_w':    np.ndarray (T, 30, 4),    # body orientations wxyz in world frame
    'body_lin_vel_w': np.ndarray (T, 30, 3),    # body linear velocities [m/s]
    'body_ang_vel_w': np.ndarray (T, 30, 3),    # body angular velocities [rad/s]
}
```
---

## Model Architecture

NMR uses a two-stage pipeline:

```
SMPL-X motion (T, 140)
        ↓
   SMPL-X VQ-VAE Encoder
        ↓ (T/4, 512)
   LLaMA Transformer (forward, non-autoregressive)
        ↓ (T/4, 512)
   G1 VQ-VAE Decoder
        ↓
G1 robot motion (T, 217)
        ↓
post-processing (Butterworth low-pass filter)
        ↓
{dof (T,29), root_trans (T,3), root_rot_quat (T,4)}
```

**Stage 1 — VQ-VAE Tokenizer**: Encodes SMPL-X human motion into a compact latent space using FSQ quantization (codebook size 1920, temporal downsampling ×4).

**Stage 2 — Transformer**: A 70M-parameter LLaMA-style model that maps human motion embeddings to G1 robot motion embeddings in a one-to-one forward pass (non-autoregressive).

For full architecture details, see the paper.

---

## Checkpoint

Model weights are hosted on HuggingFace Hub at [`RayZhao/NMR`](https://huggingface.co/RayZhao/NMR) and will be downloaded automatically on first use.

If you prefer to download manually:

```bash
huggingface-cli download RayZhao/NMR weights/epoch_30.pth --local-dir .
huggingface-cli download RayZhao/NMR assets/SMPLX_NEUTRAL.npz --local-dir .
```

---

## Citation

```bibtex

@article{zhao2026make,
  title={Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control},
  author={Zhao, Qingrui and Yang, Kaiyue and Wang, Xiyu and Zhao, Shiqi and Lu, Yi and Zhang, Xinfang and Yin, Wei and Shen, Qiu and Long, Xiao-Xiao and Cao, Xun},
  journal={arXiv preprint arXiv:2603.22201},
  year={2026}
}
```
