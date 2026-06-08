"""Minimal SceneFactor inference script - direct text-to-semantic-to-geometry."""
import os
import sys
import json
import re
import argparse
import warnings
from collections import OrderedDict

import numpy as np
import torch
from scipy.special import softmax

# Set PYTHONPATH to SceneFactor root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Monkey-patch open3d before any model imports (they import it but we don't need it)
import unittest.mock
import types
fake_o3d = types.ModuleType('open3d')
fake_o3d.geometry = types.ModuleType('open3d.geometry')
fake_o3d.utility = types.ModuleType('open3d.utility')
class FakePointCloud: pass
class FakeVector3d: pass
fake_o3d.geometry.PointCloud = FakePointCloud
fake_o3d.utility.Vector3dVector = lambda x: x
sys.modules['open3d'] = fake_o3d
sys.modules['open3d.geometry'] = fake_o3d.geometry
sys.modules['open3d.utility'] = fake_o3d.utility

from models import *
from diff_utils.helpers import *
from models import BERTEmbedder, ContextEncoder


def strip_module_prefix(ckpt):
    """Remove 'module.' prefix from checkpoint keys if present."""
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in ckpt['model_state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            return ckpt['model_state_dict']
    return model_dict


def load_semantic_model(specs, device='cuda'):
    """Load semantic VQ-VAE + diffusion model + BERT context encoder."""
    print('[1/4] Loading semantic VQ-VAE...')
    model = CombinedModel3DVQOrig.load_from_checkpoint(
        specs["modulation_ckpt_path"], specs=specs, strict=False
    ).to(device).eval()

    print('[2/4] Loading semantic diffusion model (LDM)...')
    ckpt = torch.load(specs["diffusion_ckpt_path"], map_location=device)
    model.diffusion_model.load_state_dict(strip_module_prefix(ckpt))
    model = model.eval()

    print('[3/4] Loading BERT context encoder...')
    context_encoder = BERTEmbedder(n_embed=1280, n_layer=32, device=device).to(device).eval()
    context_ckpt = torch.load(specs["diffusion_ckpt_path_context"], map_location=device)
    context_encoder.load_state_dict(strip_module_prefix(context_ckpt))

    print('[4/4] Semantic model ready!')
    return model, context_encoder


def load_geometric_model(specs, device='cuda'):
    """Load geometric VQ-VAE + diffusion model + context encoder."""
    print('[1/4] Loading geometric VQ-VAE...')
    model = CombinedModel3DVQOrigGeo.load_from_checkpoint(
        specs["modulation_ckpt_path"], specs=specs, strict=False
    ).to(device).eval()

    print('[2/4] Loading geometric diffusion model (LDM)...')
    ckpt = torch.load(specs["diffusion_ckpt_path"], map_location=device)
    model.diffusion_model.load_state_dict(strip_module_prefix(ckpt))
    model = model.eval()

    print('[3/4] Loading geometric context encoder...')
    hidden_dims = [16, 32, 64, 128, 128]
    context_encoder = ContextEncoder(in_channels=1, hidden_dims=hidden_dims).to(device).eval()
    context_ckpt = torch.load(specs["diffusion_ckpt_path_context"], map_location=device)
    context_encoder.load_state_dict(strip_module_prefix(context_ckpt))

    print('[4/4] Geometric model ready!')
    return model, context_encoder


@torch.no_grad()
def generate_semantic_chunk(model, context_encoder, caption, device='cuda'):
    """Generate a single semantic chunk from text caption."""
    condition = context_encoder([caption]).to(device)

    samples, _ = model.diffusion_model.generate_conditional(1, cond=condition)
    samples = samples[:, :, :, :8, :]  # trim to valid latent size

    recon = model.vae_model.decode(samples)  # (1, 10, 32, 16, 32)
    return recon[0].cpu().numpy()


@torch.no_grad()
def generate_geometric_chunk(model, context_encoder, sem_onehot, device='cuda'):
    """Generate a single geometric SDF chunk from semantic one-hot."""
    sem_onehot = sem_onehot.to(device)
    condition = context_encoder(sem_onehot)

    samples, _ = model.diffusion_model.generate_conditional(1, cond=condition, mode='geometry')

    recon = model.vae_model.decode(samples)  # (1, 1, 128, 64, 128)
    return recon[0, 0].cpu().numpy()


def process_semantic_for_geo(sem_array):
    """Convert semantic output (10, 32, 16, 32) -> one-hot (10, 32, 16, 32)."""
    sem = softmax(sem_array, axis=0)
    sem = np.argmax(sem, axis=0)
    sem_vox = torch.LongTensor(sem)
    onehot = torch.nn.functional.one_hot(sem_vox, num_classes=10)
    onehot = torch.permute(onehot, (3, 0, 1, 2)).float()
    return onehot


def make_sem_specs(ckpt_dir):
    return {
        "modulation_ckpt_path": os.path.join(ckpt_dir, "sem_vqvae.ckpt"),
        "diffusion_ckpt_path": os.path.join(ckpt_dir, "sem_diff_main.ckpt"),
        "diffusion_ckpt_path_context": os.path.join(ckpt_dir, "sem_diff_encoder.ckpt"),
        "training_task": "combined",
        "Factor": "sem",
        "embed_dim": 1,
        "n_embed": 8192,
        "ddconfig": {
            "double_z": 0, "z_channels": 1, "resolution": 16,
            "in_channels": 10, "out_ch": 10, "ch": 16,
            "ch_mult": [2, 4], "num_res_blocks": 2,
            "attn_resolutions": [], "dropout": 0.0
        },
        "diffusion_specs": {
            "timesteps": 1000, "sampling_timesteps": 350,
            "objective": "pred_v", "loss_type": "l2",
            "beta_schedule": "cosine", "noise_scale": 1.0
        },
        "diffusion_model_specs": {
            "image_size": 16, "in_channels": 1, "out_channels": 1,
            "model_channels": 192, "num_res_blocks": 2,
            "attention_resolutions": [2, 4],
            "channel_mult": [1, 2, 4, 4], "downs": [1, 1, 1, 1],
            "num_heads": 8, "dims": 3, "context_dim": 1280,
            "use_spatial_transformer": 1, "dropout": 0.0, "mode": "sem"
        },
        "cond_mode": "text",
    }


def make_geo_specs(ckpt_dir):
    return {
        "modulation_ckpt_path": os.path.join(ckpt_dir, "geo_vqvae_onestage.ckpt"),
        "diffusion_ckpt_path": os.path.join(ckpt_dir, "geo_diff_main_onestage.ckpt"),
        "diffusion_ckpt_path_context": os.path.join(ckpt_dir, "geo_diff_encoder_onestage.ckpt"),
        "training_task": "combined",
        "Factor": "geo",
        "embed_dim": 1,
        "n_embed": 32768,
        "ddconfig": {
            "double_z": 0, "z_channels": 1, "resolution": 64,
            "in_channels": 1, "out_ch": 1, "ch": 16,
            "ch_mult": [1, 2, 4], "num_res_blocks": 1,
            "attn_resolutions": [], "dropout": 0.0
        },
        "diffusion_specs": {
            "timesteps": 1000, "sampling_timesteps": 350,
            "objective": "pred_v", "loss_type": "l2",
            "beta_schedule": "cosine", "noise_scale": 1.0
        },
        "diffusion_model_specs": {
            "image_size": 16, "in_channels": 1, "out_channels": 1,
            "model_channels": 96, "num_res_blocks": 1,
            "attention_resolutions": [4, 2, 1],
            "channel_mult": [1, 2, 4],
            "num_heads": 8, "dims": 3, "context_dim": 128,
            "use_spatial_transformer": 1, "dropout": 0.0, "mode": "geo"
        },
        "cond_mode": "sem",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', '-t', type=str, default='a living room with a sofa and a coffee table',
                        help='Text prompt for semantic generation')
    parser.add_argument('--out', '-o', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--sem-only', action='store_true',
                        help='Only run semantic stage')
    parser.add_argument('--ckpt_dir', type=str,
                        default='/home/lijiarui/Desktop/scene_factor/checkpoints',
                        help='Directory containing all 6 checkpoint files')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    os.makedirs(args.out, exist_ok=True)

    # ====== Semantic Stage ======
    print('\n' + '='*60)
    print('Stage 1: Semantic Generation')
    print('='*60)
    sem_specs = make_sem_specs(args.ckpt_dir)
    sem_model, context_encoder = load_semantic_model(sem_specs, device)

    print(f'\nGenerating semantic chunk from: "{args.text}"')
    sem_output = generate_semantic_chunk(sem_model, context_encoder, args.text, device)
    sem_path = os.path.join(args.out, 'semantic_chunk.npy')
    np.save(sem_path, sem_output)
    sem_classes = softmax(sem_output, axis=0).argmax(axis=0)
    unique_classes = np.unique(sem_classes)
    print(f'Saved: {sem_path}')
    print(f'Semantic shape: {sem_output.shape}')
    print(f'Detected object classes: {sorted(unique_classes.tolist())}')

    if args.sem_only:
        print('\nSemantic stage completed (--sem-only).')
        return

    # ====== Geometric Stage ======
    print('\n' + '='*60)
    print('Stage 2: Geometric Generation')
    print('='*60)
    geo_specs = make_geo_specs(args.ckpt_dir)
    geo_model, geo_encoder = load_geometric_model(geo_specs, device)

    print('\nConverting semantic to geometric...')
    sem_onehot = process_semantic_for_geo(sem_output)
    geo_output = generate_geometric_chunk(geo_model, geo_encoder, sem_onehot.unsqueeze(0), device)
    geo_path = os.path.join(args.out, 'geometric_chunk.npy')
    np.save(geo_path, geo_output)
    print(f'Saved: {geo_path}')
    print(f'Geometric shape: {geo_output.shape}')

    print('\n' + '='*60)
    print(f'Done! Results in: {args.out}')
    print('='*60)


if __name__ == '__main__':
    main()
