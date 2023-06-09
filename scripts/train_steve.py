# sourced by https://github.com/singhgautam/steve

import math
import os.path
import argparse

import wandb
import torch
import torchvision.utils as vutils
from torchvision import transforms

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from ocbench.steve.steve import STEVE
from ocbench.dataset.GlobVideoDataset import GlobVideoDataset

from ocbench.dataset import ClevrWithMasks, DATASET_PATH
from ocbench.steve.utils import cosine_anneal, linear_warmup
from ocbench.evaluator import adjusted_rand_index
from tqdm import tqdm


def visualize(video, recon_dvae, recon_tf, attns, N=8):
    B, T, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :]

        # tile
        tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(
            end_dim=1
        )

        # grid
        frame = vutils.make_grid(tiles, nrow=(args.num_slots + 3), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


def train(args, model, optimizer, writer, train_loader, val_loader):
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

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    train_epoch_size = len(train_loader)
    val_epoch_size = len(val_loader)

    log_interval = train_epoch_size // 5

    for epoch in range(start_epoch, args.epochs if not args.debug else 5):
        model.train()

        # Train Phase!
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )
        for i, batch_data in train_iterator:
            global_step = epoch * train_epoch_size + i

            tau = cosine_anneal(
                global_step, args.tau_start, args.tau_final, 0, args.tau_steps
            )

            lr_warmup_factor_enc = linear_warmup(
                global_step, 0.0, 1.0, 0.0, args.lr_warmup_steps
            )

            lr_warmup_factor_dec = linear_warmup(
                global_step, 0.0, 1.0, 0, args.lr_warmup_steps
            )

            lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

            optimizer.param_groups[0]["lr"] = args.lr_dvae
            optimizer.param_groups[1]["lr"] = (
                lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
            )
            optimizer.param_groups[2]["lr"] = (
                lr_decay_factor * lr_warmup_factor_dec * args.lr_dec
            )

            images = batch_data["image"].cuda()
            video = images.unsqueeze(1)
            B, T, C, H, W = video.size()

            optimizer.zero_grad()

            (recon, cross_entropy, mse, recons, masks) = model(video, tau, args.hard)

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            loss = mse + cross_entropy

            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip, "inf")
            optimizer.step()

            with torch.no_grad():
                if i % log_interval == 0:
                    print(
                        "Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}".format(
                            epoch + 1, i, train_epoch_size, loss.item(), mse.item()
                        )
                    )

                    writer.add_scalar("TRAIN/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "TRAIN/cross_entropy", cross_entropy.item(), global_step
                    )
                    writer.add_scalar("TRAIN/mse", mse.item(), global_step)

                    writer.add_scalar("TRAIN/tau", tau, global_step)
                    writer.add_scalar(
                        "TRAIN/lr_dvae", optimizer.param_groups[0]["lr"], global_step
                    )
                    writer.add_scalar(
                        "TRAIN/lr_enc", optimizer.param_groups[1]["lr"], global_step
                    )
                    writer.add_scalar(
                        "TRAIN/lr_dec", optimizer.param_groups[2]["lr"], global_step
                    )

            if args.debug:
                break

        with torch.no_grad():
            gen_video = (
                model.module if args.use_dp else model
            ).reconstruct_autoregressive(video[:8])
            frames = visualize(video, recon, gen_video, recons, N=8)
            writer.add_video("TRAIN_recons/epoch={:03}".format(epoch + 1), frames)

        with torch.no_grad():
            model.eval()
            # Eval Phase!
            val_cross_entropy = 0.0
            val_mse = 0.0
            ari_log = []
            fgari_log = []

            val_iterator = tqdm(
                enumerate(val_loader), total=len(val_loader), desc="testing"
            )
            for i, batch_data in val_iterator:
                images = batch_data["image"].cuda()
                gt_masks = batch_data["mask"].cuda()

                video = images.unsqueeze(1)
                B, T, C, H, W = video.size()

                (recon, cross_entropy, mse, recons, masks) = model(
                    video, tau, args.hard
                )

                ari = adjusted_rand_index(
                    gt_masks, masks.squeeze(1), exclude_background=False
                )
                fgari = adjusted_rand_index(gt_masks, masks.squeeze(1))
                ari_log.append(torch.mean(ari))
                fgari_log.append(torch.mean(fgari))

                if args.use_dp:
                    mse = mse.mean()
                    cross_entropy = cross_entropy.mean()

                val_cross_entropy += cross_entropy.item()
                val_mse += mse.item()

                if args.debug:
                    break

            val_cross_entropy /= val_epoch_size
            val_mse /= val_epoch_size

            val_loss = val_mse + val_cross_entropy
            val_ari = torch.mean(torch.stack(ari_log))
            val_fgari = torch.mean(torch.stack(fgari_log))

            writer.add_scalar("VAL/loss", val_loss, epoch + 1)
            writer.add_scalar("VAL/cross_entropy", val_cross_entropy, epoch + 1)
            writer.add_scalar("VAL/mse", val_mse, epoch + 1)
            writer.add_scalar("VAL/ARI", val_ari, epoch + 1)
            writer.add_scalar("VAL/FG-ARI", val_fgari, epoch + 1)

            print(
                "====> Epoch: {:3} \t Loss = {:F} \t ARI = {:F} \t FG-ARI = {:F}".format(
                    epoch + 1, val_loss, val_ari, val_fgari
                )
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(
                    model.module.state_dict() if args.use_dp else model.state_dict(),
                    os.path.join(args.log_path, "best_model.pt"),
                )

                if global_step < args.steps:
                    torch.save(
                        model.module.state_dict()
                        if args.use_dp
                        else model.state_dict(),
                        os.path.join(
                            args.log_path, f"best_model_until_{args.steps}_steps.pt"
                        ),
                    )

                if 50 <= epoch:
                    gen_video = (
                        model.module if args.use_dp else model
                    ).reconstruct_autoregressive(video[:8])
                    frames = visualize(video, recon, gen_video, recons, N=8)
                    writer.add_video("VAL_recons/epoch={:03}".format(epoch + 1), frames)

            writer.add_scalar("VAL/best_loss", best_val_loss, epoch + 1)

            checkpoint = {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "model": model.module.state_dict()
                if args.use_dp
                else model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, os.path.join(args.log_path, "checkpoint.pt.tar"))

            print("====> Best Loss = {:F} @ Epoch {}".format(best_val_loss, best_epoch))


def main(args):
    torch.manual_seed(args.seed)

    arg_str_list = ["{}={}".format(k, v) for k, v in vars(args).items()]
    arg_str = "__".join(arg_str_list)
    args.log_path = os.path.join(args.log_path, f"{datetime.today()}-{args.prefix}-{args.seed}")
    writer = SummaryWriter(args.log_path)
    writer.add_text("hparams", arg_str)

    training_transforms = {
        "image": transforms.Compose(
            [
                transforms.CenterCrop(192),
                transforms.Resize(128, transforms.InterpolationMode.NEAREST),
                transforms.ConvertImageDtype(torch.float32),
            ]
        ),
        "mask": transforms.Compose(
            [
                transforms.CenterCrop(192),
                transforms.Resize(128, transforms.InterpolationMode.NEAREST),
                transforms.ConvertImageDtype(torch.float32),
            ]
        ),
    }

    train_dataset = ClevrWithMasks(
        root=DATASET_PATH,
        split="Train",
        ttv=[90000, 5000, 5000],
        transforms=training_transforms,
        download=True,
        convert=True,
    )
    val_dataset = ClevrWithMasks(
        root=DATASET_PATH,
        split="Val",
        ttv=[90000, 5000, 5000],
        transforms=training_transforms,
        download=True,
        convert=True,
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

    model = STEVE(args)

    model = model.cuda()
    if args.use_dp:
        model = DP(model)

    optimizer = Adam(
        [
            {
                "params": (x[1] for x in model.named_parameters() if "dvae" in x[0]),
                "lr": args.lr_dvae,
            },
            {
                "params": (
                    x[1] for x in model.named_parameters() if "steve_encoder" in x[0]
                ),
                "lr": 0.0,
            },
            {
                "params": (
                    x[1] for x in model.named_parameters() if "steve_decoder" in x[0]
                ),
                "lr": 0.0,
            },
        ]
    )

    # start to train STEVE
    train(args, model, optimizer, writer, train_loader, val_loader)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", type=str, default="sa-clevr-n5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument(
    #     "--level", type=str, default="A", choices=["A", "B", "C", "D", "E"]
    # )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--ep_len", type=int, default=1)

    parser.add_argument("--checkpoint_path", default="checkpoint.pt.tar")
    parser.add_argument("--log_path", default="logs/")

    parser.add_argument("--lr_dvae", type=float, default=3e-4)
    parser.add_argument("--lr_enc", type=float, default=1e-4)
    parser.add_argument("--lr_dec", type=float, default=3e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=30000)
    parser.add_argument("--lr_half_life", type=int, default=250000)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps", type=int, default=200000)

    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--num_slots", type=int, default=15)
    parser.add_argument("--cnn_hidden_size", type=int, default=64)
    parser.add_argument("--slot_size", type=int, default=192)
    parser.add_argument("--mlp_hidden_size", type=int, default=192)
    parser.add_argument("--num_predictor_blocks", type=int, default=1)
    parser.add_argument("--num_predictor_heads", type=int, default=4)
    parser.add_argument("--predictor_dropout", type=int, default=0.0)

    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--num_decoder_blocks", type=int, default=8)
    parser.add_argument("--num_decoder_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--dropout", type=int, default=0.1)

    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_final", type=float, default=0.1)
    parser.add_argument("--tau_steps", type=int, default=30000)

    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--use_dp", default=True, action="store_true")

    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()

    def _create_prefix(args: dict):
        assert (
            args["prefix"] is not None and args["prefix"] != ""
        ), "Must specify a prefix to use W&B"
        d = datetime.today()
        date_id = f"{d.month}{d.day}{d.hour}{d.minute}{d.second}"
        before = f"{date_id}-{args['seed']}-"

        if args["prefix"] != "debug" and args["prefix"] != "NONE":
            prefix = before + args["prefix"]
            print("Assigning full prefix %s" % prefix)
        else:
            prefix = args["prefix"]

        return prefix

    prefix_args = {
        "prefix": args.prefix,
        "seed": args.seed,
    }

    wandb.init(
        name=_create_prefix(prefix_args),
        project="sa-flex",
        entity="cun_bjy",
        sync_tensorboard=True,
    )

    wandb.config.update(args)

    main(args)
