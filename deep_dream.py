#!/usr/bin/env python3
"""
Deep Dream visualization tool using PyTorch + pretrained models.

Supports:
  - VGG19, ResNet50, and InceptionV3 models
  - Octave (multi-scale) processing
  - Guided dreaming with a target image
  - L2 regularization and jitter for stability
  - Configurable layers, octaves, iterations

Usage:
  python deep_dream.py --input INPUT --output OUTPUT [options]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

MODELS = {
    "vgg19": models.vgg19,
    "resnet50": models.resnet50,
    "inceptionv3": models.inception_v3,
}

# Sensible default layers per architecture
DEFAULT_LAYERS = {
    "vgg19": ["features.28"],        # conv5_3 — rich features
    "resnet50": ["layer4"],           # deepest residual block
    "inceptionv3": ["Mixed_7a"],      # deep inception module
}

# All candidate layers per architecture (for --list-layers)
LAYER_CATALOG = {
    "vgg19": (
        [f"features.{i}" for i in range(len(models.vgg19(pretrained=True).features))]
        + ["avgpool", "classifier.0", "classifier.3", "classifier.6"]
    ),
    "resnet50": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"],
    "inceptionv3": [
        "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1",
        "Conv2d_4a_3x3", "Mixed_5b", "Mixed_5c", "Mixed_5d",
        "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
        "Mixed_7a", "Mixed_7b", "Mixed_7c",
    ],
}

# Early (geometric) layer defaults per architecture
EARLY_LAYERS = {
    "vgg19": ["features.10"],   # conv3_3 — geometric
    "resnet50": ["layer1"],      # early residual
    "inceptionv3": ["Mixed_5d"], # early inception
}


def load_model(name: str) -> nn.Module:
    """Load a pretrained model in eval mode."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODELS.keys())}")
    fn = MODELS[name]
    if name == "inceptionv3":
        model = fn(pretrained=True, aux_logits=False)
    else:
        model = fn(pretrained=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_layer(model: nn.Module, name: str) -> nn.Module:
    """Walk dotted path to fetch a submodule."""
    cur = model
    for attr in name.split("."):
        if hasattr(cur, attr):
            cur = getattr(cur, attr)
        elif attr.isdigit():
            cur = cur[int(attr)]
        else:
            raise AttributeError(f"Layer '{name}' not found in model")
    return cur


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]
_normalize = transforms.Normalize(_mean, _std)
_denorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(_mean, _std)],
    std=[1 / s for s in _std],
)


def img_to_tensor(path_or_img, size=None) -> torch.Tensor:
    """Load an image → normalised float32 tensor [1, 3, H, W]."""
    if isinstance(path_or_img, (str, Path)):
        img = Image.open(path_or_img).convert("RGB")
    else:
        img = path_or_img
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    tfm = transforms.Compose([transforms.ToTensor(), _normalize])
    return tfm(img).unsqueeze(0)


def tensor_to_img(t: torch.Tensor) -> Image.Image:
    """Denormalise tensor → PIL Image."""
    t = _denorm(t.squeeze(0).cpu())
    t = t.clamp(0, 1)
    return transforms.ToPILImage()(t)


# ---------------------------------------------------------------------------
# Deep Dream core
# ---------------------------------------------------------------------------

class DreamLoss(nn.Module):
    """Activations from one or more layers, summed with L2."""
    def __init__(self, model: nn.Module, layer_names, guide_tensor=None):
        super().__init__()
        self.layer_names = layer_names
        self.activations = {}
        self.guide_acts = {}

        # Register hooks
        for name in layer_names:
            layer = get_layer(model, name)
            layer.register_forward_hook(self._hook(name))

        # Pre-compute guide activations if provided
        if guide_tensor is not None:
            with torch.no_grad():
                _ = model(guide_tensor)
            for name in layer_names:
                self.guide_acts[name] = self.activations[name].clone()
            self.activations.clear()

    def _hook(self, name):
        def _fn(_module, _input, output):
            self.activations[name] = output
        return _fn

    def forward(self, x):
        # activations filled via hooks during forward pass of the model
        loss = 0.0
        for name in self.layer_names:
            act = self.activations[name]
            if name in self.guide_acts:
                # Guided: maximise cosine-similarity with guide
                guide = self.guide_acts[name]
                loss -= torch.nn.functional.cosine_similarity(
                    act.flatten(1), guide.flatten(1)
                ).mean()
            else:
                # Standard: maximise L2 norm of activations
                loss -= (act ** 2).mean()
        return loss


def deep_dream(
    model: nn.Module,
    input_img: Image.Image,
    layer_names,
    num_octaves=4,
    octave_scale=1.4,
    iterations_per_octave=15,
    lr=0.01,
    jitter=32,
    l2_reg=1e-3,
    guide_img=None,
    device="cpu",
):
    """
    Run DeepDream with octave processing.

    Returns a PIL Image.
    """
    model = model.to(device)

    # If guide image provided, compute its activations
    guide_tensor = None
    if guide_img is not None:
        guide_tensor = img_to_tensor(guide_img, size=input_img.size).to(device)

    dream_loss_fn = DreamLoss(model, layer_names, guide_tensor)

    # Build octave sizes (smallest first)
    orig_w, orig_h = input_img.size
    octaves = []
    for o in range(num_octaves - 1, -1, -1):
        sz = (int(orig_w / (octave_scale ** o)), int(orig_h / (octave_scale ** o)))
        sz = (max(sz[0], 64), max(sz[1], 64))
        octaves.append(sz)

    img = input_img
    for oi, sz in enumerate(octaves):
        print(f"  Octave {oi+1}/{len(octaves)} — size {sz}")
        img_tensor = img_to_tensor(img, size=sz).to(device).detach().requires_grad_(True)

        for it in range(iterations_per_octave):
            # Random jitter
            ox, oy = np.random.randint(-jitter, jitter + 1, 2)
            shifted = torch.roll(img_tensor, (ox, oy), dims=(2, 3))

            # Forward + backward
            dream_loss_fn.activations.clear()
            model(shifted)
            loss = dream_loss_fn(shifted)
            loss.backward()

            grad = img_tensor.grad.data

            # L2 regularisation on the gradient
            grad = grad / (grad.norm() + 1e-8) * lr

            img_tensor.data += grad
            img_tensor.grad.zero_()

            # Un-jitter
            img_tensor.data = torch.roll(img_tensor.data, (-ox, -oy), dims=(2, 3))

            # Clip to valid range (in normalised space)
            img_tensor.data = torch.clip(img_tensor.data, -2.5, 2.5)

            if (it + 1) % 5 == 0:
                print(f"    iter {it+1}/{iterations_per_octave}  loss={loss.item():.4f}")

        img = tensor_to_img(img_tensor.detach())

        # Upscale to next octave size (or original)
        if oi < len(octaves) - 1:
            next_sz = octaves[oi + 1]
        else:
            next_sz = (orig_w, orig_h)
        img = img.resize(next_sz, Image.LANCZOS)

    return img


# ---------------------------------------------------------------------------
# Generate a base Figment image (purple gradient + dragon shapes)
# ---------------------------------------------------------------------------

def generate_figment_base(width=800, height=600):
    """Create a base purple/dragon-themed image for dreaming."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Purple gradient background
    for y in range(height):
        for x in range(width):
            r = int(80 + 40 * (x / width) + 20 * (y / height))
            g = int(20 + 30 * (1 - x / width))
            b = int(160 + 60 * (y / height) + 30 * (x / width))
            arr[y, x] = [min(r, 255), min(g, 255), min(b, 255)]

    # Add some "dragon" shapes — yellow eyes, swirls
    # Big yellow eyes
    for cx, cy, radius in [(280, 220, 35), (380, 210, 35)]:
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
        arr[mask] = [255, 220, 50]
        # Pupils
        mask2 = (xx - cx) ** 2 + (yy - cy) ** 2 < (radius * 0.4) ** 2
        arr[mask2] = [30, 10, 80]

    # Dragon body (rounded snout shape)
    for cx, cy, rx, ry in [(330, 340, 120, 80)]:  # body
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
        arr[mask] = [140, 50, 180]

    # Little horns
    for cx, cy, rx, ry in [(260, 150, 20, 50), (400, 145, 20, 50)]:
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
        arr[mask] = [200, 160, 50]

    # Yellow belly
    for cx, cy, rx, ry in [(330, 380, 80, 40)]:
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
        arr[mask] = [255, 230, 80]

    # Add some noise/texture
    noise = np.random.randint(0, 15, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Dream visualisation")
    parser.add_argument("--input", "-i", help="Input image path (generates a Figment base if omitted)")
    parser.add_argument("--output", "-o", default="dream_output.png", help="Output image path")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="vgg19")
    parser.add_argument("--layers", nargs="+", default=None, help="Layer names to dream at")
    parser.add_argument("--octaves", type=int, default=4)
    parser.add_argument("--octave-scale", type=float, default=1.4)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--jitter", type=int, default=32)
    parser.add_argument("--l2-reg", type=float, default=1e-3)
    parser.add_argument("--guide", help="Guide image for guided dreaming")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list-layers", action="store_true", help="Print available layers and exit")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    if args.list_layers:
        print(f"\nAvailable layers for {args.model}:")
        for l in LAYER_CATALOG.get(args.model, []):
            print(f"  {l}")
        return

    # Determine layers
    if args.layers:
        layers = args.layers
    else:
        layers = DEFAULT_LAYERS[args.model]

    print(f"Layers: {layers}")

    # Load or generate base image
    if args.input:
        input_img = Image.open(args.input).convert("RGB")
        print(f"Loaded input: {args.input} ({input_img.size})")
    else:
        print("Generating Figment-themed base image…")
        input_img = generate_figment_base(args.width, args.height)

    # Load guide image if provided
    guide_img = None
    if args.guide:
        guide_img = Image.open(args.guide).convert("RGB")
        print(f"Loaded guide: {args.guide}")

    print(f"Running DeepDream — {args.octaves} octaves, {args.iterations} iters/octave, lr={args.lr}")
    result = deep_dream(
        model=model,
        input_img=input_img,
        layer_names=layers,
        num_octaves=args.octaves,
        octave_scale=args.octave_scale,
        iterations_per_octave=args.iterations,
        lr=args.lr,
        jitter=args.jitter,
        l2_reg=args.l2_reg,
        guide_img=guide_img,
        device=args.device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out_path), quality=95)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()