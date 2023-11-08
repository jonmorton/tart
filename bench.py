import argparse

import torch
from torch.utils.benchmark import Timer

from tart import vocab
from tart.config import cfg, cfg_util
from tart.model.builder import build_model

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def main(args):
    if args.cfg:
        cfg_util.merge_yaml(args.cfg)
    if args.opts:
        cfg_util.merge_dotlist(args.opts)

    device = "cuda"
    model = build_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    if cfg.compile:
        model = torch.compile(model)
    x = torch.randint(
        0, vocab.size - 1, (cfg.optim.batch_size, cfg.model.seq_len), dtype=torch.long
    ).to(device)

    res = Timer(
        """
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y, (loss, _) = model(x, x)
        sum(loss.values()).backward()
        """,
        globals={"x": x, "model": model},
    ).blocked_autorange(min_run_time=15.0)

    print(repr(res))


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", "-c", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument("--name", "-n", default="", help="Experiment name")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    main(args)
