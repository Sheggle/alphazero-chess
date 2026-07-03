"""ResNet policy/value network for chess + an evaluator for the search.

Architecture follows AlphaZero in miniature: a conv stem, a stack of residual
blocks, then a policy head (4672 logits) and a value head (tanh scalar, from the
side-to-move's perspective). Sized modestly so CPU self-play is tractable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chess_encode import INPUT_PLANES, encode_state
from .chess_env import ACTION_SIZE


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head: 1x1 conv to a small width, then linear to the move space.
        self.p_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 64, ACTION_SIZE)

        # Value head.
        self.v_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(8)
        self.v_fc1 = nn.Linear(8 * 64, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.tower(self.stem(x))
        p = F.relu(self.p_bn(self.p_conv(x))).flatten(1)
        policy_logits = self.p_fc(p)
        v = F.relu(self.v_bn(self.v_conv(x))).flatten(1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return policy_logits, value


class ChessEvaluator:
    """Maps a ChessGame to (policy over ACTION_SIZE, value) for the search.

    Only legal moves get probability mass (softmax over legal logits).
    """

    def __init__(self, net: ChessNet, device: str = "cpu"):
        self.net = net.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, state):
        self.net.eval()
        x = torch.from_numpy(encode_state(state)[None]).to(self.device)
        logits, value = self.net(x)
        logits = logits[0].cpu().numpy()
        legal = state.legal_moves()
        probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        if legal:
            ll = logits[legal]
            ll = ll - ll.max()
            ex = np.exp(ll)
            probs[legal] = ex / ex.sum()
        return probs, float(value.item())

    @torch.no_grad()
    def predict_batch(self, states):
        self.net.eval()
        x = torch.from_numpy(np.stack([encode_state(s) for s in states])).to(self.device)
        logits, values = self.net(x)
        logits = logits.cpu().numpy()
        out = []
        for i, s in enumerate(states):
            legal = s.legal_moves()
            probs = np.zeros(ACTION_SIZE, dtype=np.float32)
            if legal:
                ll = logits[i][legal]; ll = ll - ll.max(); ex = np.exp(ll)
                probs[legal] = ex / ex.sum()
            out.append(probs)
        return np.stack(out), values.cpu().numpy()
