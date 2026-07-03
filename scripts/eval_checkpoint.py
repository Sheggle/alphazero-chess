"""Evaluate an AlphaZero-chess checkpoint and print all metrics as one JSON line.

Usage:
    PYTHONPATH=. uv run python scripts/eval_checkpoint.py <path-to-ckpt.pt>
    PYTHONPATH=. uv run python scripts/eval_checkpoint.py fresh [channels] [blocks]

The checkpoint dict is expected to hold {state_dict, channels, blocks}. The
special arg `fresh` builds a random-init net (default 32 channels, 4 blocks) for
a from-scratch baseline. Built to run in well under a minute (1 torch thread).
"""

from __future__ import annotations

import json
import sys
import time

import torch

from alphazero.chess_eval import evaluate
from alphazero.chess_net import ChessEvaluator, ChessNet

torch.set_num_threads(1)


def load_net(arg: str, extra: list[str]) -> tuple[ChessNet, dict]:
    if arg == "fresh":
        channels = int(extra[0]) if len(extra) > 0 else 32
        blocks = int(extra[1]) if len(extra) > 1 else 4
        torch.manual_seed(0)
        net = ChessNet(channels=channels, blocks=blocks)
        meta = {"source": "fresh", "channels": channels, "blocks": blocks}
        return net, meta
    ckpt = torch.load(arg, map_location="cpu", weights_only=False)
    channels = ckpt.get("channels", 64)
    blocks = ckpt.get("blocks", 6)
    net = ChessNet(channels=channels, blocks=blocks)
    net.load_state_dict(ckpt["state_dict"])
    meta = {"source": arg, "channels": channels, "blocks": blocks,
            "iter": ckpt.get("iter")}
    return net, meta


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: eval_checkpoint.py <ckpt.pt | fresh [ch] [blocks]>",
              file=sys.stderr)
        sys.exit(1)
    net, meta = load_net(sys.argv[1], sys.argv[2:])
    net.eval()
    ev = ChessEvaluator(net)

    t0 = time.time()
    metrics = evaluate(ev)
    metrics["meta"] = meta
    metrics["eval_secs"] = round(time.time() - t0, 1)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
