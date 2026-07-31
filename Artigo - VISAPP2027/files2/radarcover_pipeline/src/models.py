"""
models.py — Etapa 3: Candidate Model Pool.

Seis arquiteturas de restauração/super-resolução, cada uma implementada sobre
o motor `nn_core` preservando o traço arquitetural que a diferencia na
literatura:

  - SRCNN        : 3 convs simples, sem skip-connections, upsample bicúbico
                    prévio (arquitetura original de Dong et al.).
  - ESPCN        : extrai features em baixa resolução e faz upsample via
                    sub-pixel convolution (PixelShuffle) no final — mais
                    barato que SRCNN pois convolui em LR.
  - EDSR-lite     : blocos residuais empilhados (sem batchnorm, como no EDSR
                    original), upsample bicúbico + refinamento residual.
  - RLFN-lite     : blocos "residual local feature" com convoluções 1x1 de
                    gargalo (bottleneck) intercaladas com 3x3 — leve, focado
                    em eficiência (few-params, poucos FLOPs).
  - SwinIR-Light  : aproximação leve sem atenção real (custo de implementar
                    self-attention em NumPy puro é proibitivo); usa blocos
                    conv residuais com normalização de escala para emular o
                    comportamento de "deep feature extraction + shallow
                    residual" do SwinIR. Documentado como simplificação.
  - CRMN-lite     : Convolutional Recurrent Mixer Network — bloco conv
                    seguido de uma mistura recorrente ao longo de "passos"
                    (estado oculto reaproveitado entre iterações), similar ao
                    modelo do próprio dataset IPMet (Pires et al., ICASSP
                    2025).

Cada modelo tem custo (profundidade x largura x fusão) crescente, o que gera
o trade-off qualidade x latência que a Etapa 6 (Weighted Set Multicover)
explora.

Todos os modelos operam em imagens de 1 canal (intensidade de refletividade,
ver `data.py`) e recebem entrada em baixa resolução (LR) já do tamanho da
imagem de saída dividido pelo fator de escala.
"""

from __future__ import annotations
import time
import numpy as np
from nn_core import Conv2D, ReLU, PixelShuffle, bicubic_upsample, BaseSRModel


class SRCNN(BaseSRModel):
    """Dong et al. 2014 — 3 camadas conv sobre entrada já upsampled."""
    name = "SRCNN"

    def __init__(self, scale, seed=0):
        self.scale = scale
        self.c1 = Conv2D(1, 16, k=9, seed=seed + 1)
        self.r1 = ReLU()
        self.c2 = Conv2D(16, 8, k=5, seed=seed + 2)
        self.r2 = ReLU()
        self.c3 = Conv2D(8, 1, k=5, seed=seed + 3)

    def forward(self, x):
        xu = bicubic_upsample(x, self.scale)
        h = self.r1.forward(self.c1.forward(xu))
        h = self.r2.forward(self.c2.forward(h))
        out = self.c3.forward(h)
        return out

    def backward(self, dout, lr):
        d = self.c3.backward(dout)
        dh, dW3, db3 = d
        dh = self.r2.backward(dh)
        dh, dW2, db2 = self.c2.backward(dh)
        dh = self.r1.backward(dh)
        _, dW1, db1 = self.c1.backward(dh)
        self.c3.step(dW3, db3, lr); self.c2.step(dW2, db2, lr); self.c1.step(dW1, db1, lr)

    def n_params(self):
        return self.c1.n_params() + self.c2.n_params() + self.c3.n_params()


class ESPCN(BaseSRModel):
    """Shi et al. 2016 — features em LR + sub-pixel convolution."""
    name = "ESPCN"

    def __init__(self, scale, seed=0):
        self.scale = scale
        self.c1 = Conv2D(1, 16, k=5, seed=seed + 1)
        self.r1 = ReLU()
        self.c2 = Conv2D(16, 16, k=3, seed=seed + 2)
        self.r2 = ReLU()
        self.c3 = Conv2D(16, scale * scale, k=3, seed=seed + 3)
        self.ps = PixelShuffle(scale)

    def forward(self, x):
        h = self.r1.forward(self.c1.forward(x))
        h = self.r2.forward(self.c2.forward(h))
        h = self.c3.forward(h)
        out = self.ps.forward(h)
        return out

    def backward(self, dout, lr):
        dh = self.ps.backward(dout)
        dh, dW3, db3 = self.c3.backward(dh)
        dh = self.r2.backward(dh)
        dh, dW2, db2 = self.c2.backward(dh)
        dh = self.r1.backward(dh)
        _, dW1, db1 = self.c1.backward(dh)
        self.c3.step(dW3, db3, lr); self.c2.step(dW2, db2, lr); self.c1.step(dW1, db1, lr)

    def n_params(self):
        return self.c1.n_params() + self.c2.n_params() + self.c3.n_params()


class ResBlock:
    """Bloco residual simples (conv-relu-conv + skip), sem batchnorm."""

    def __init__(self, ch, seed=0):
        self.c1 = Conv2D(ch, ch, k=3, seed=seed + 1)
        self.r1 = ReLU()
        self.c2 = Conv2D(ch, ch, k=3, seed=seed + 2)

    def forward(self, x):
        self._x = x
        h = self.r1.forward(self.c1.forward(x))
        h = self.c2.forward(h)
        return x + h

    def backward(self, dout, lr):
        dh, dW2, db2 = self.c2.backward(dout)
        dh = self.r1.backward(dh)
        dh2, dW1, db1 = self.c1.backward(dh)
        self.c2.step(dW2, db2, lr); self.c1.step(dW1, db1, lr)
        return dout + dh2  # gradiente da skip-connection soma diretamente

    def n_params(self):
        return self.c1.n_params() + self.c2.n_params()


class EDSRLite(BaseSRModel):
    """EDSR simplificado: N blocos residuais + upsample bicúbico + refino."""
    name = "EDSR-Lite"

    def __init__(self, scale, n_blocks=3, ch=12, seed=0):
        self.scale = scale
        self.head = Conv2D(1, ch, k=3, seed=seed + 1)
        self.blocks = [ResBlock(ch, seed=seed + 10 + i) for i in range(n_blocks)]
        self.tail = Conv2D(ch, 1, k=3, seed=seed + 99)

    def forward(self, x):
        xu = bicubic_upsample(x, self.scale)
        h = self.head.forward(xu)
        self._acts = [h]
        for b in self.blocks:
            h = b.forward(h)
            self._acts.append(h)
        out = self.tail.forward(h) + xu  # residual global
        self._xu = xu
        return out

    def backward(self, dout, lr):
        dh, dWt, dbt = self.tail.backward(dout)
        self.tail.step(dWt, dbt, lr)
        for b in reversed(self.blocks):
            dh = b.backward(dh, lr)
        _, dWh, dbh = self.head.backward(dh)
        self.head.step(dWh, dbh, lr)

    def n_params(self):
        return self.head.n_params() + sum(b.n_params() for b in self.blocks) + self.tail.n_params()


class RLFBlock:
    """Residual Local Feature block: bottleneck 1x1 -> 3x3 -> 1x1 + skip."""

    def __init__(self, ch, bottleneck, seed=0):
        self.c1 = Conv2D(ch, bottleneck, k=1, seed=seed + 1)
        self.r1 = ReLU()
        self.c2 = Conv2D(bottleneck, bottleneck, k=3, seed=seed + 2)
        self.r2 = ReLU()
        self.c3 = Conv2D(bottleneck, ch, k=1, seed=seed + 3)

    def forward(self, x):
        h = self.r1.forward(self.c1.forward(x))
        h = self.r2.forward(self.c2.forward(h))
        h = self.c3.forward(h)
        return x + h

    def backward(self, dout, lr):
        dh, dW3, db3 = self.c3.backward(dout)
        dh = self.r2.backward(dh)
        dh, dW2, db2 = self.c2.backward(dh)
        dh = self.r1.backward(dh)
        dh2, dW1, db1 = self.c1.backward(dh)
        self.c3.step(dW3, db3, lr); self.c2.step(dW2, db2, lr); self.c1.step(dW1, db1, lr)
        return dout + dh2

    def n_params(self):
        return self.c1.n_params() + self.c2.n_params() + self.c3.n_params()


class RLFNLite(BaseSRModel):
    """RLFN simplificado — foco em poucos parâmetros/FLOPs."""
    name = "RLFN-Lite"

    def __init__(self, scale, n_blocks=2, ch=10, bottleneck=6, seed=0):
        self.scale = scale
        self.head = Conv2D(1, ch, k=3, seed=seed + 1)
        self.blocks = [RLFBlock(ch, bottleneck, seed=seed + 10 + i) for i in range(n_blocks)]
        self.tail = Conv2D(ch, scale * scale, k=3, seed=seed + 99)
        self.ps = PixelShuffle(scale)

    def forward(self, x):
        h = self.head.forward(x)
        for b in self.blocks:
            h = b.forward(h)
        h = self.tail.forward(h)
        return self.ps.forward(h)

    def backward(self, dout, lr):
        dh = self.ps.backward(dout)
        dh, dWt, dbt = self.tail.backward(dh)
        self.tail.step(dWt, dbt, lr)
        for b in reversed(self.blocks):
            dh = b.backward(dh, lr)
        _, dWh, dbh = self.head.backward(dh)
        self.head.step(dWh, dbh, lr)

    def n_params(self):
        return self.head.n_params() + sum(b.n_params() for b in self.blocks) + self.tail.n_params()


class SwinIRLightLite(BaseSRModel):
    """
    Aproximação leve do SwinIR-Light SEM atenção real (custo proibitivo em
    NumPy puro por imagem completa). Emula "deep feature extraction em
    janelas locais + conexão residual global rasa" via blocos conv mais
    profundos que o EDSR-lite mas com canais mais estreitos, refletindo o
    trade-off qualidade/custo do transformer original. Simplificação
    registrada no Capítulo 7 (Limitações).
    """
    name = "SwinIR-Light-Lite"

    def __init__(self, scale, n_blocks=4, ch=8, seed=0):
        self.scale = scale
        self.head = Conv2D(1, ch, k=3, seed=seed + 1)
        self.blocks = [ResBlock(ch, seed=seed + 20 + i) for i in range(n_blocks)]
        self.tail = Conv2D(ch, 1, k=3, seed=seed + 88)

    def forward(self, x):
        xu = bicubic_upsample(x, self.scale)
        h = self.head.forward(xu)
        for b in self.blocks:
            h = b.forward(h)
        out = self.tail.forward(h) + xu
        return out

    def backward(self, dout, lr):
        dh, dWt, dbt = self.tail.backward(dout)
        self.tail.step(dWt, dbt, lr)
        for b in reversed(self.blocks):
            dh = b.backward(dh, lr)
        _, dWh, dbh = self.head.backward(dh)
        self.head.step(dWh, dbh, lr)

    def n_params(self):
        return self.head.n_params() + sum(b.n_params() for b in self.blocks) + self.tail.n_params()


class CRMNLite(BaseSRModel):
    """
    Convolutional Recurrent Mixer Network (leve) — inspirado em Pires et al.
    (ICASSP 2025), autores do próprio dataset IPMet. Um bloco conv é
    reaplicado T vezes reaproveitando um estado oculto (mistura recorrente),
    seguido de upsample via sub-pixel convolution.
    """
    name = "CRMN-Lite"

    def __init__(self, scale, ch=10, steps=3, seed=0):
        self.scale = scale
        self.steps = steps
        self.head = Conv2D(1, ch, k=3, seed=seed + 1)
        self.mix = Conv2D(ch, ch, k=3, seed=seed + 2)  # pesos compartilhados entre passos
        self.r = ReLU()
        self.tail = Conv2D(ch, scale * scale, k=3, seed=seed + 3)
        self.ps = PixelShuffle(scale)

    def forward(self, x):
        h = self.head.forward(x)
        self._states = [h]
        for _ in range(self.steps):
            h = self.r.forward(self.mix.forward(h)) + h  # mistura recorrente com skip
            self._states.append(h)
        out = self.ps.forward(self.tail.forward(h))
        return out

    def backward(self, dout, lr):
        dh = self.ps.backward(dout)
        dh, dWt, dbt = self.tail.backward(dh)
        self.tail.step(dWt, dbt, lr)
        # backprop através dos T passos recorrentes (pesos compartilhados: acumula grad)
        dW_acc = np.zeros_like(self.mix.W)
        db_acc = np.zeros_like(self.mix.b)
        for _ in range(self.steps):
            dmix = self.r.backward(dh)
            dprev, dW, db = self.mix.backward(dmix)
            dW_acc += dW; db_acc += db
            dh = dh + dprev  # soma pois havia skip h_t = f(h_{t-1}) + h_{t-1}
        self.mix.step(dW_acc, db_acc, lr)
        _, dWh, dbh = self.head.backward(dh)
        self.head.step(dWh, dbh, lr)

    def n_params(self):
        return self.head.n_params() + self.mix.n_params() + self.tail.n_params()


# --------------------------------------------------------------------------- #
# Registro do pool de modelos candidatos
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "SRCNN": SRCNN,
    "ESPCN": ESPCN,
    "EDSR-Lite": EDSRLite,
    "RLFN-Lite": RLFNLite,
    "SwinIR-Light-Lite": SwinIRLightLite,
    "CRMN-Lite": CRMNLite,
}


def build_candidate_pool(scale: int, seeds=(0, 1)):
    """
    Constrói o pool de modelos candidatos (Etapa 3), incluindo múltiplas
    seeds aleatórias por arquitetura (conforme diagrama: 'Multiple random
    seeds'), retornando uma lista de instâncias nomeadas de forma única
    (ex.: 'SRCNN#0', 'SRCNN#1', ...).
    """
    pool = {}
    for name, cls in MODEL_REGISTRY.items():
        for s in seeds:
            key = f"{name}#{s}"
            pool[key] = cls(scale=scale, seed=s * 100)
    return pool


def measure_latency(model, lr_patch, n_reps=5):
    """Latência média de inferência (ms) para um patch LR único."""
    # warm-up
    model.predict(lr_patch)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        model.predict(lr_patch)
    t1 = time.perf_counter()
    return (t1 - t0) / n_reps * 1000.0
