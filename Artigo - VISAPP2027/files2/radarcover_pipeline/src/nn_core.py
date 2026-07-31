"""
nn_core.py — Motor mínimo de CNN em NumPy puro (im2col + backprop manual).

Motivação de design
--------------------
O ambiente de execução é CPU-only com recursos limitados (sem GPU, disco
restrito para wheels CUDA de frameworks como PyTorch/TensorFlow). Para manter
o pipeline RadarCover 100% executável e reprodutível neste ambiente, os
modelos candidatos (Etapa 3 do pipeline) são implementados sobre este motor
de autograd manual, e não sobre um framework de deep learning completo.

Isso é documentado explicitamente como limitação de escala (ver Capítulo 7 —
Limitações do artigo): os blocos aqui definidos preservam a *topologia* e o
papel arquitetural de cada modelo candidato (profundidade, uso de
skip-connections, sub-pixel convolution, mistura convolucional-recorrente),
mas usam larguras/profundidades reduzidas. Para experimentos em escala de
publicação, a mesma interface (`BaseSRModel.forward/backward/train_step`)
deve ser portada 1:1 para PyTorch em GPU — a arquitetura do pipeline
(regimes, cobertura, multicover, ensemble podado) é agnóstica ao framework.
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Utilitários de convolução via im2col
# --------------------------------------------------------------------------- #

def _pad(x: np.ndarray, p: int) -> np.ndarray:
    if p == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="reflect")


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0):
    """x: (N, C, H, W) -> cols: (N, C*kh*kw, out_h*out_w)"""
    x = _pad(x, pad)
    N, C, H, W = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            cols[:, :, y, xk, :, :] = x[:, :, y:y_max:stride, xk:x_max:stride]
    cols = cols.reshape(N, C * kh * kw, out_h * out_w)
    return cols, out_h, out_w


def col2im(cols_grad, x_shape, kh, kw, stride=1, pad=0):
    N, C, H, W = x_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    out_h = (Hp - kh) // stride + 1
    out_w = (Wp - kw) // stride + 1
    cols_grad = cols_grad.reshape(N, C, kh, kw, out_h, out_w)
    dx_pad = np.zeros((N, C, Hp, Wp), dtype=cols_grad.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            dx_pad[:, :, y:y_max:stride, xk:x_max:stride] += cols_grad[:, :, y, xk, :, :]
    if pad == 0:
        return dx_pad
    return dx_pad[:, :, pad:-pad, pad:-pad]


# --------------------------------------------------------------------------- #
# Camadas
# --------------------------------------------------------------------------- #

class Conv2D:
    """Convolução 2D 'same' (padding reflect) com Adam embutido."""

    def __init__(self, in_ch, out_ch, k=3, stride=1, bias=True, seed=0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = rng.normal(0, scale, size=(out_ch, in_ch, k, k)).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32) if bias else None
        self.k, self.stride = k, stride
        self.pad = k // 2
        self._cache = None
        # Adam state
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        if bias:
            self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t = 0

    @property
    def out_ch(self):
        return self.W.shape[0]

    def n_params(self):
        return self.W.size + (self.b.size if self.b is not None else 0)

    def forward(self, x):
        N, C, H, W = x.shape
        cols, out_h, out_w = im2col(x, self.k, self.k, self.stride, self.pad)
        Wm = self.W.reshape(self.out_ch, -1)
        out = np.einsum("oc,ncp->nop", Wm, cols)
        if self.b is not None:
            out += self.b[None, :, None]
        out = out.reshape(N, self.out_ch, out_h, out_w)
        self._cache = (x.shape, cols)
        return out

    def backward(self, dout):
        x_shape, cols = self._cache
        N = x_shape[0]
        dout_flat = dout.reshape(N, self.out_ch, -1)
        Wm = self.W.reshape(self.out_ch, -1)
        dW = np.einsum("nop,ncp->oc", dout_flat, cols).reshape(self.W.shape)
        dcols = np.einsum("oc,nop->ncp", Wm, dout_flat)
        dx = col2im(dcols, x_shape, self.k, self.k, self.stride, self.pad)
        db = dout_flat.sum(axis=(0, 2)) if self.b is not None else None
        return dx, dW, db

    def step(self, dW, db, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * (dW ** 2)
        mW_hat = self.mW / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        if self.b is not None and db is not None:
            self.mb = beta1 * self.mb + (1 - beta1) * db
            self.vb = beta2 * self.vb + (1 - beta2) * (db ** 2)
            mb_hat = self.mb / (1 - beta1 ** self.t)
            vb_hat = self.vb / (1 - beta2 ** self.t)
            self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class ReLU:
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class PixelShuffle:
    """Sub-pixel convolution upsampling (usado por ESPCN e afins)."""

    def __init__(self, scale):
        self.scale = scale

    def forward(self, x):
        N, C, H, W = x.shape
        r = self.scale
        Co = C // (r * r)
        self._shape = x.shape
        x = x.reshape(N, Co, r, r, H, W)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        return x.reshape(N, Co, H * r, W * r)

    def backward(self, dout):
        N, C, H, W = self._shape
        r = self.scale
        Co = C // (r * r)
        d = dout.reshape(N, Co, H, r, W, r)
        d = d.transpose(0, 1, 3, 5, 2, 4)
        return d.reshape(N, C, H, W)


def bicubic_upsample(x, scale):
    """Upsample simples (replicação bilinear) usado como base residual."""
    N, C, H, W = x.shape
    out = np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)
    return out


class BaseSRModel:
    """Interface comum a todos os modelos candidatos do pool (Etapa 3)."""

    name = "base"

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout, lr):
        raise NotImplementedError

    def n_params(self):
        raise NotImplementedError

    def train_step(self, lr_batch, hr_batch, lr_rate=1e-3):
        pred = self.forward(lr_batch)
        diff = pred - hr_batch
        loss = float(np.mean(diff ** 2))
        dout = (2.0 / diff.size) * diff
        self.backward(dout, lr_rate)
        return loss

    def predict(self, lr_batch):
        return self.forward(lr_batch)
