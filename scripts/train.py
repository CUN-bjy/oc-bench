import math
import os.path
import argparse

import torch
import torchvision.utils as vutils
from torchvision import transforms

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.nn import DataParallel as DP
from datetime import datetime

from ocbench.slot_attention.model import SlotAttentionModel
from ocbench.dataset import ClevrWithMasks, DATASET_PATH
from tqdm import tqdm

from ocbench.evaluator import adjusted_rand_index


def configure_optimizers(args, model, train_size):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps_pct = args.warmup_steps_pct
    decay_steps_pct = args.decay_steps_pct
    total_steps = args.max_epochs * train_size

    def warm_and_decay_lr_scheduler(step: int):
        warmup_steps = warmup_steps_pct * total_steps
        decay_steps = decay_steps_pct * total_steps
        assert step < total_steps
        if step < warmup_steps:
            factor = step / warmup_steps
        else:
            factor = 1
        factor *= args.scheduler_gamma ** (step / decay_steps)
        return factor

    scheduler = lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler
    )

    return (
        [optimizer],
        [
            {
                "scheduler": scheduler,
                "interval": "step",
            }
        ],
    )


def visualize(input, recon_combined, recons, N=8):
    B, C, H, W = input.size()

    input = input.unsqueeze(1)
    recon_combined = recon_combined.unsqueeze(1)
    
    frames = []
    
    # tile
    tiles = torch.cat((input, recon_combined, recons), dim=1).flatten(end_dim=1)

    # grid
    frame = vutils.make_grid(tiles, nrow=(args.num_slots + 2), pad_value=0.8)
    frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


def train(args, model, writer, train_loader, val_loader):
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]
        model.load_state_dict(checkpoint["model"])
    else:
        checkpoint = None
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0

    train_epoch_size = len(train_loader)
    val_epoch_size = len(val_loader)

    log_interval = train_epoch_size // 5

    # define optimizers
    optimizer, optim_info = configure_optimizers(args, model, train_epoch_size)
    optimizer = optimizer[0]
    scheduler = optim_info[0]["scheduler"]

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch, args.max_epochs if not args.debug else 5):
        # Train Phase!
        model.train()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )
        for i, batch_data in train_iterator:
            global_step = epoch * train_epoch_size + i

            images = batch_data['image'].cuda()
            B, C, H, W = images.size()

            optimizer.zero_grad()

            recon_combined, recons, _, _ = model(images)
            mse = ((recon_combined - images) ** 2).sum() / B

            if args.use_dp:
                mse = mse.mean()

            loss = mse

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if i % log_interval == 0:
                    print(
                        "Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}".format(
                            epoch + 1, i, train_epoch_size, loss.item(), loss.item()
                        )
                    )

                    writer.add_scalar("TRAIN/loss", loss.item(), global_step)
                    writer.add_scalar("TRAIN/mse", loss.item(), global_step)
                    writer.add_scalar(
                        "TRAIN/lr", scheduler.get_last_lr()[0], global_step
                    )

            if args.debug:
                break

        with torch.no_grad():
            frames = visualize(images, recon_combined, recons, N=8)
            writer.add_video("TRAIN_recons/epoch={:03}".format(epoch + 1), frames)

        with torch.no_grad():
            model.eval()
            val_mse = 0.0
            ari_log = []
            fgari_log = []
            
            # Eval Phase!
            val_iterator = tqdm(
                enumerate(val_loader), total=len(val_loader), desc="testing"
            )
            for i, batch_data in val_iterator:
                images = batch_data['image'].cuda()
                gt_masks = batch_data['masks'].cuda()
                
                B, C, H, W = images.size()

                recon_combined, recons, pred_masks, _ = model(images)
                mse = ((recon_combined - images) ** 2).sum() / B

                ari = adjusted_rand_index(gt_masks, pred_masks)
                fgari = adjusted_rand_index(gt_masks[:, 1:], pred_masks)
                ari_log.append(torch.mean(ari))
                fgari_log.append(torch.mean(fgari))

                if args.use_dp:
                    mse = mse.mean()

                val_mse += loss.item()

                if args.debug:
                    break

            val_mse /= val_epoch_size

            val_loss = val_mse
            val_ari = torch.mean(torch.stack(ari_log))
            val_fgari = torch.mean(torch.stack(fgari_log))

            writer.add_scalar("VAL/loss", val_loss, epoch + 1)
            writer.add_scalar("VAL/mse", val_mse, epoch + 1)
            writer.add_scalar("VAL/ARI", val_ari, epoch + 1)
            writer.add_scalar("VAL/FG-ARI", val_fgari, epoch + 1)

            print("====> Epoch: {:3} \t Loss = {:F} \t ARI = {:F} \t FG-ARI = {:F}".format(epoch + 1, val_loss, val_ari, val_fgari))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(
                    model.module.state_dict() if args.use_dp else model.state_dict(),
                    os.path.join(log_dir, "best_model.pt"),
                )

                if global_step < args.steps:
                    torch.save(
                        model.module.state_dict()
                        if args.use_dp
                        else model.state_dict(),
                        os.path.join(
                            log_dir, f"best_model_until_{args.steps}_steps.pt"
                        ),
                    )

                if 10 <= epoch:
                    frames = visualize(images, recon_combined, recons, N=8)
                    writer.add_video("VAL_recons/epoch={:03}".format(epoch + 1), frames)

            checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "model": model.module.state_dict()
                if args.use_dp
                else model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, os.path.join(log_dir, "checkpoint.pt.tar"))
            print("====> Best Loss = {:F} @ Epoch {}".format(best_val_loss, best_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--checkpoint_path", default="checkpoint.pt.tar")
    parser.add_argument("--log_path", default="logs/")

    parser.add_argument("--dataset", type=str, default="clevr", choices=["clevr"])
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=128)

    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_slots", type=int, default=5)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--empty_cache", type=bool, default=True)
    parser.add_argument("--is_logger_enabled", type=bool, default=True)
    parser.add_argument("--is_verbose", type=bool, default=True)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--warmup_steps_pct", type=float, default=0.02)
    parser.add_argument("--decay_steps_pct", type=float, default=0.2)
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=(64, 64, 64, 64))
    parser.add_argument("--slot_size", type=int, default=64)

    parser.add_argument("--use_dp", default=True, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    arg_str_list = ["{}={}".format(k, v) for k, v in vars(args).items()]
    arg_str = "__".join(arg_str_list)
    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)
    writer.add_text("hparams", arg_str)

    training_transforms  = {
        'image': transforms.Compose([
            transforms.CenterCrop(192),
            transforms.Resize(128, transforms.InterpolationMode.NEAREST)
        ]),     
        'mask': transforms.Compose([
            transforms.CenterCrop(192),
            transforms.Resize(128, transforms.InterpolationMode.NEAREST)
        ])
    }

    train_dataset = ClevrWithMasks(
        root=DATASET_PATH, 
        split='Train', 
        ttv=[90000, 5000, 5000], 
        transforms=training_transforms,
        download=True, 
        convert=True
    )
    val_dataset = ClevrWithMasks(
        root=DATASET_PATH, 
        split='Val', 
        ttv=[90000, 5000, 5000], 
        transforms=training_transforms,
        download=True, 
        convert=True
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }

    train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)

    model = SlotAttentionModel(
        resolution=(args.image_size, args.image_size),
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        empty_cache=args.empty_cache,
        hidden_dims=tuple(args.hidden_dims),
        slot_size=args.slot_size,
    )
    model = model.cuda()
    if args.use_dp:
        model = DP(model)

    train(args, model, writer, train_loader, val_loader)
