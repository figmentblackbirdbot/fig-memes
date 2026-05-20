#!/usr/bin/env python3
"""
Generate all three Figment Deep Dream variations:
  1. figment_deep_dream.png       — standard dreaming (VGG19, conv5_3)
  2. figment_deep_dream_heavy.png — heavy dreaming (more iters, deeper)
  3. figment_deep_dream_geometric.png — geometric patterns (earlier layer)
"""

import sys
import time
from pathlib import Path

import torch

# Add repo to path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from deep_dream import deep_dream, load_model, generate_figment_base, DEFAULT_LAYERS, EARLY_LAYERS
from PIL import Image

OUT = REPO

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Generate the base Figment image
print("=== Generating Figment base image ===")
base = generate_figment_base(800, 600)

# Save the base for reference
base.save(str(OUT / "figment_base.png"), quality=95)
print(f"Saved base: {OUT / 'figment_base.png'}")

# -------------------------------------------------------
# 1) Standard dream — VGG19, features.28 (conv5_3)
# -------------------------------------------------------
print("\n=== 1/3: Standard DeepDream (VGG19 conv5_3) ===")
model_vgg = load_model("vgg19")
t0 = time.time()
result_standard = deep_dream(
    model=model_vgg,
    input_img=base,
    layer_names=["features.28"],
    num_octaves=4,
    octave_scale=1.4,
    iterations_per_octave=15,
    lr=0.01,
    jitter=32,
    l2_reg=1e-3,
    device=device,
)
result_standard.save(str(OUT / "figment_deep_dream.png"), quality=95)
print(f"Saved: figment_deep_dream.png  ({time.time()-t0:.1f}s)")

# -------------------------------------------------------
# 2) Heavy dream — VGG19, conv5_1 + conv5_3, more iterations
# -------------------------------------------------------
print("\n=== 2/3: Heavy DeepDream (VGG19 conv5_1+conv5_3, 25 iters) ===")
model_vgg2 = load_model("vgg19")
t0 = time.time()
result_heavy = deep_dream(
    model=model_vgg2,
    input_img=base,
    layer_names=["features.24", "features.28"],  # conv5_1 + conv5_3
    num_octaves=4,
    octave_scale=1.4,
    iterations_per_octave=25,
    lr=0.012,
    jitter=40,
    l2_reg=5e-4,
    device=device,
)
result_heavy.save(str(OUT / "figment_deep_dream_heavy.png"), quality=95)
print(f"Saved: figment_deep_dream_heavy.png  ({time.time()-t0:.1f}s)")

# -------------------------------------------------------
# 3) Geometric — VGG19, features.10 (conv3_3) — early layer
# -------------------------------------------------------
print("\n=== 3/3: Geometric DeepDream (VGG19 conv3_3) ===")
model_vgg3 = load_model("vgg19")
t0 = time.time()
result_geometric = deep_dream(
    model=model_vgg3,
    input_img=base,
    layer_names=["features.10"],  # conv3_3 — geometric patterns
    num_octaves=4,
    octave_scale=1.5,
    iterations_per_octave=15,
    lr=0.01,
    jitter=24,
    l2_reg=1e-3,
    device=device,
)
result_geometric.save(str(OUT / "figment_deep_dream_geometric.png"), quality=95)
print(f"Saved: figment_deep_dream_geometric.png  ({time.time()-t0:.1f}s)")

print("\n=== All done! ===")
for f in ["figment_base.png", "figment_deep_dream.png", "figment_deep_dream_heavy.png", "figment_deep_dream_geometric.png"]:
    p = OUT / f
    size_kb = p.stat().st_size / 1024
    print(f"  {f}: {size_kb:.0f} KB")