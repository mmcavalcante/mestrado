"""
data.py — Etapas 1 e 2 do pipeline: IPMet Radar Dataset + Image Preprocessing.

Decodificação de refletividade
-------------------------------
As imagens do IPMet Radar Dataset são PNGs RGBA (README declara "8-bit
grayscale", mas os arquivos reais são compostos por um colormap meteorológico
+ canal alfa de cobertura, confirmado por inspeção). Cada pixel colorido
representa uma faixa de refletividade (dBZ) segundo uma paleta de cores; o
canal alfa=0 indica ausência de eco (fora do alcance/sem retorno).

Para este pipeline, decodificamos a refletividade relativa como:
  1. mask   = alpha > 0                      (presença de eco)
  2. intens = luminância ponderada de R,G,B, normalizada em [0,1],
              multiplicada pela mask (fundo = 0)

Essa é uma proxy documentada da intensidade de refletividade (não a
reconstrução exata da escala dBZ da paleta IPMet, que exigiria a tabela de
cores oficial do produto). É registrada explicitamente como premissa
metodológica no Capítulo 7 (Limitações) do artigo.
"""

from __future__ import annotations
import os
import re
import glob
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
from PIL import Image

FILENAME_RE = re.compile(r",-(\d{8})-(\d{4})_")


@dataclass
class RadarFrame:
    path: str
    acq_dt: datetime


def parse_acquisition_time(path: str) -> datetime:
    m = FILENAME_RE.search(os.path.basename(path))
    if not m:
        return datetime.min
    date_s, time_s = m.groups()
    return datetime.strptime(date_s + time_s, "%Y%m%d%H%M")


def list_dataset(images_dir: str) -> List[RadarFrame]:
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    frames = [RadarFrame(p, parse_acquisition_time(p)) for p in paths]
    frames.sort(key=lambda f: f.acq_dt)
    return frames


def load_reflectivity(path: str, size: int | None = None) -> np.ndarray:
    """Carrega um PNG RGBA e retorna intensidade normalizada float32 [0,1], shape (H,W)."""
    im = Image.open(path).convert("RGBA")
    if size is not None:
        im = im.resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    mask = (a > 0).astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum = lum / 255.0
    intens = lum * mask
    return intens.astype(np.float32)


def chronological_split(frames: List[RadarFrame], train=0.7, val=0.15):
    """Split cronológico (não aleatório) — evita vazamento temporal entre
    quadros de radar adjacentes (altamente correlacionados)."""
    n = len(frames)
    n_train = int(n * train)
    n_val = int(n * val)
    return (
        frames[:n_train],
        frames[n_train:n_train + n_val],
        frames[n_train + n_val:],
    )


# --------------------------------------------------------------------------- #
# Etapa 2: Image Preprocessing
# --------------------------------------------------------------------------- #

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma, mode="reflect")


def add_gaussian_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noisy = img + rng.normal(0, std, size=img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def downsample(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape
    return img.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))


def degrade(hr: np.ndarray, scale: int, blur_sigma: float, noise_std: float,
            rng: np.random.Generator) -> np.ndarray:
    """Gera entrada degradada (LR): blur -> downsample -> ruído gaussiano.
    Simula perda óptica/eletrônica + subamostragem do radar."""
    blurred = gaussian_blur(hr, blur_sigma)
    lr = downsample(blurred, scale)
    lr = add_gaussian_noise(lr, noise_std, rng)
    return lr.astype(np.float32)


def extract_patches(img: np.ndarray, patch_size: int, stride: int, max_patches: int | None = None,
                     rng: np.random.Generator | None = None) -> List[np.ndarray]:
    h, w = img.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(img[y:y + patch_size, x:x + patch_size])
    if max_patches is not None and len(patches) > max_patches:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(patches), size=max_patches, replace=False)
        patches = [patches[i] for i in idx]
    return patches


def build_lr_hr_pairs(hr_patch: np.ndarray, scale: int, blur_sigma: float, noise_std: float,
                       rng: np.random.Generator):
    """A partir de um patch HR, gera o par (LR, HR) para treino/avaliação de SR."""
    lr = degrade(hr_patch, scale, blur_sigma, noise_std, rng)
    return lr[None, None, ...], hr_patch[None, None, ...]  # (N=1,C=1,H,W)
