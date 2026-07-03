"""Tiny policy + value network for tic-tac-toe.

A small conv trunk over the 2x3x3 input, splitting into:
  - policy head: 9 logits (one per cell),
  - value head:  1 scalar in [-1, 1] (tanh), from the side-to-move's view.

TTT is trivial, so this is deliberately small. The same interface (a `predict`
that maps a state to (policy_probs, value)) is what the AlphaZero MCTS consumes,
so we can later swap in a bigger ResNet for bigger games without touching MCTS.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import BOARD, INPUT_PLANES, encode


class TicTacToeNet(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

        self.policy_fc = nn.Linear(channels * BOARD, BOARD)
        self.value_fc1 = nn.Linear(channels * BOARD, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        policy_logits = self.policy_fc(x)
        v = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy_logits, value


class NetEvaluator:
    """Wraps a net to give MCTS (policy_probs over legal moves, value) for a state.

    Masks illegal moves before softmax so priors only cover legal cells.
    """

    def __init__(self, net: TicTacToeNet, device: str = "cpu"):
        self.net = net.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, state) -> tuple[np.ndarray, float]:
        self.net.eval()
        x = torch.from_numpy(encode(state)[None]).to(self.device)
        logits, value = self.net(x)
        logits = logits[0].cpu().numpy()
        mask = np.asarray(state.legal_mask(), dtype=bool)
        logits = np.where(mask, logits, -1e9)
        logits -= logits.max()
        exp = np.exp(logits) * mask
        probs = exp / exp.sum()
        return probs, float(value.item())

    @torch.no_grad()
    def predict_batch(self, states) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        x = torch.from_numpy(np.stack([encode(s) for s in states])).to(self.device)
        logits, values = self.net(x)
        logits = logits.cpu().numpy()
        masks = np.stack([np.asarray(s.legal_mask(), dtype=bool) for s in states])
        logits = np.where(masks, logits, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits) * masks
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs, values.cpu().numpy()
