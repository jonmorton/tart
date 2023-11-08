import argparse
import math
import os
import time

import nnt
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tart import vocab
from tart.config import cfg, cfg_util
from tart.data import create_dataset
from tart.generation import generate_text
from tart.model.builder import build_model
from tart.model.weights import get_wd_params

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def data_iterable(dataloader):
    while True:
        yield from dataloader


class LRSchedule(LRScheduler):
    def __init__(self, optimizer, warmup_iter, max_iter, warmup_frac, final_frac):
        self.warmup_iter = warmup_iter
        self.warmup_frac = warmup_frac
        self.final_frac = final_frac
        self.max_iter = max_iter
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        i = self.last_epoch
        if i < self.warmup_iter:
            mult = (i / self.warmup_iter) * (1.0 - self.warmup_frac) + self.warmup_frac
        else:
            mult = (i - self.warmup_iter) / (self.max_iter - self.warmup_iter) * (
                self.final_frac - 1.0
            ) + 1.0
        return [lr * mult for lr in self.base_lrs]


def compute_loss(model, input, dtype=torch.bfloat16):
    x = input[:, :-1].contiguous()
    target = input[:, 1:].contiguous()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        pred, (losses, correct) = model(x, target)

    acc = correct.sum() / target.numel()
    nnt.log({f"accuracy/{'train' if model.training else 'eval'}": acc.item()})

    return pred, losses


def do_eval(model, val_loader, device, dtype):
    val_it = 0
    val_loss = 0
    t = time.perf_counter()
    max_val_iter = cfg.val_examples // cfg.optim.batch_size
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            _, losses = compute_loss(model, batch, dtype)
            val_loss += sum(l.item() for l in losses.values())
            val_it += 1
            if val_it >= max_val_iter:
                break

    nt = time.perf_counter()
    val_time = nt - t
    nnt.log(
        {
            "loss/val": val_loss / val_it,
            "time/validation": val_time,
        }
    )
    t = nt
    model.train()


def train():
    device = "cuda"
    dtype = torch.bfloat16
    max_iter = cfg.optim.total_steps
    start_iter = 0

    nnt.seed_all(2)
    nnt.path_mgr.mkdirs(cfg.out_dir)
    cfg_util.save_yaml(os.path.join(cfg.out_dir, "config.yaml"))

    model = build_model().to(device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    all_params = set(model.parameters())
    wd_params = set(get_wd_params(model))

    param_groups = [
        {"params": list(wd_params), "weight_decay": cfg.optim.weight_decay},
        {"params": list(all_params - wd_params), "weight_decay": 0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
    )
    lr_scheduler = LRSchedule(
        optimizer,
        warmup_iter=cfg.optim.warmup_steps,
        max_iter=cfg.optim.total_steps,
        warmup_frac=0.05,
        final_frac=0.1,
    )

    train_ds = create_dataset(cfg.data.root, "train", cfg.model.seq_len + 1)
    train_nbatch = len(train_ds)
    val_ds = create_dataset(cfg.data.root, "validation", cfg.model.seq_len + 1)

    train_loader = DataLoader(
        train_ds,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        num_workers=cfg.data.num_workers,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    nnt.log.add_writers(
        nnt.TensorboardWriter(os.path.join(cfg.out_dir, "tb")),
        nnt.MetricPrinter(cfg.log_period, max_iter),
    )

    checkpointer = nnt.Checkpointer(
        model,
        cfg.out_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    if checkpointer.has_checkpoint():
        ckpt = checkpointer.load(checkpointer.get_checkpoint_file())
        start_iter = ckpt["iteration"] + 1
    checkpointer = nnt.PeriodicCheckpointer(
        checkpointer,
        period=cfg.checkpoint_period,
        max_iter=max_iter,
    )

    if cfg.compile:
        model = torch.compile(model)

    nnt.log.iter = start_iter
    t = time.perf_counter()
    data_iter = iter(data_iterable(train_loader))
    tokens_per_batch = cfg.model.seq_len * cfg.optim.batch_size
    ntok = 0
    nbatch = 0

    for iteration in range(start_iter, max_iter):
        nt = time.perf_counter()
        data_time = nt - t
        grad_accum_steps = cfg.optim.grad_accum_steps
        optimizer.zero_grad()
        loss_accum = {}
        for _ in range(grad_accum_steps):
            nbatch += 1
            batch = next(data_iter).to(device)
            pred, losses = compute_loss(model, batch, dtype)
            loss = sum(losses.values())
            loss.backward()
            for k, v in losses.items():
                loss_accum[k] = v / grad_accum_steps + loss_accum.get(k, 0)

        if len(loss_accum) > 1:
            for k, v in loss_accum.items():
                nnt.log({f"loss/{k}": v.item()})

        ntok += tokens_per_batch * grad_accum_steps
        nnt.log(
            {
                "loss/total": sum(loss_accum.values()).item(),
                "Mtoken": ntok / 1e6,
                "epoch": nbatch // train_nbatch,
            }
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_norm_clip)
        optimizer.step()

        if iteration > 0 and iteration % cfg.val_period == 0:
            plen = 128
            vp = pred[0, :plen, :].argmax(-1).cpu()
            vt = batch[0, 1 : plen + 1].cpu()

            print("pred>", vocab.decode(vp.tolist()))
            print("targ>", vocab.decode(vt.tolist()))
            model.eval()
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = generate_text(
                    model,
                    prepend_eod=True,
                    max_new_tokens=256,
                    temperature=1.0,
                    top_p=0.0,
                )
            print(" gen>", out)
            model.train()

        nt = time.perf_counter()
        iter_time = nt - t
        t = nt
        nnt.log(lr=lr_scheduler.get_last_lr()[0])

        if iteration - start_iter > 2:
            nnt.log(
                {
                    "time/data": data_time,
                    "time/iter": iter_time,
                }
            )

        if iteration > 0 and iteration % cfg.val_period == 0:
            do_eval(model, val_loader, device, dtype)

        if iteration > 0 and iteration % cfg.log_hists_period == 0:
            for name, p in model.named_parameters():
                nnt.log({f"parameters/{name}": p.data.to("cpu")})

        lr_scheduler.step()
        nnt.log.step()
        checkpointer.step(iteration)


def main(args):
    if args.cfg:
        cfg_util.merge_yaml(args.cfg)
    if args.opts:
        cfg_util.merge_dotlist(args.opts)

    print(cfg_util.to_yaml())

    train()


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
