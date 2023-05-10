# modified from https://github.com/untitled-ai/slot_attention

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from ocbench.slot_attention.utils import Tensor
from ocbench.slot_attention.utils import assert_shape
from ocbench.slot_attention.utils import build_grid
from ocbench.slot_attention.utils import conv_transpose_out_shape


class SlotAttention(nn.Module):
    def __init__(self, in_features, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.slot_dim = dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(in_features, dim)
        self.to_v = nn.Linear(in_features, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(in_features)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs: Tensor, num_slots = None):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # Initialize the slots. Shape: [batch_size, num_slots, slot_dim].
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)
        
        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        assert_shape(k.size(), (b, n, self.slot_dim))
        assert_shape(v.size(), (b, n, self.slot_dim))
        
        # Iterative update slots
        for _ in range(self.iters):
            slots_prev = slots
            
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            assert_shape(q.size(), (b, n_s, self.slot_dim))

            # Attention
            scaled_dots = self.scale * torch.matmul(k, q.transpose(2, 1))
            attn = scaled_dots.softmax(dim=1) + self.eps
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (b, n, n_s))

            # Weighted Mean.
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_dim].
            assert_shape(updates.size(), (b, n_s, self.slot_dim))

            # Slot Update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(b * n_s, self.slot_dim),
                slots_prev.view(b * n_s, self.slot_dim),
            )
            slots = slots.view(b, n_s, self.slot_dim)
            assert_shape(slots.size(), (b, n_s, self.slot_dim))
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            assert_shape(slots.size(), (b, n_s, self.slot_dim))

        return slots




class SlotAttentionModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (8, 8),
        empty_cache=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]

        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(
            self.in_channels, self.out_features, resolution
        )
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features,
                    self.out_features,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(
                    self.out_features,
                    4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=0,
                ),
            )
        )

        assert_shape(resolution, (out_size, out_size), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(
            self.in_channels, self.out_features, self.decoder_resolution
        )

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_slots=self.num_slots,
            iters=self.num_iterations,
            dim=self.slot_size,
            hidden_dim=128,
        )

    def forward(self, x):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        slots = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape

        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(
            1, 1, self.decoder_resolution[0], self.decoder_resolution[1]
        )

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(
            out.size(), (batch_size * num_slots, num_channels + 1, height, width)
        )

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def loss_function(self, input):
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input)
        return {
            "loss": loss,
        }


class SoftPositionEmbed(nn.Module):
    def __init__(
        self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]
    ):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
