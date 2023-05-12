# modified from https://github.com/singhgautam/sysbinder

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ocbench.sysbinder.utils import Tensor, Conv2dBlock, conv2d, linear
from ocbench.sysbinder.utils import assert_shape, gumbel_softmax
from ocbench.sysbinder.block_utils import BlockLinear, BlockGRU, BlockLayerNorm
from ocbench.sysbinder.transformer import TransformerEncoder, TransformerDecoder, CartesianPositionalEmbedding, LearnedPositionalEmbedding1D
from ocbench.sysbinder.dvae import dVAE, OneHotDictionary

class SysBinder(nn.Module):
    def __init__(self, in_features, num_slots, num_blocks, num_prototypes, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots # N
        self.num_blocks = num_blocks # M
        self.slot_dim = dim # Md
        self.block_dim = dim // num_blocks # d
        self.num_prototypes = num_prototypes # k
        assert dim % num_blocks == 0, "slot_dim(Md) must be dividible by num_blocks(M)"
        
        self.iters = iters
        self.eps = eps

        # parameters for slots
        self.slots_mu = nn.Parameter(torch.Tensor(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_logsigma)

        # linear function for attention module
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(in_features, dim)
        self.to_v = nn.Linear(in_features, dim)

        # GRU & MLP to update block-slots
        self.gru = BlockGRU(dim, dim, num_blocks)

        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(       
            BlockLinear(dim, hidden_dim, num_blocks), 
            nn.ReLU(inplace = True),
            BlockLinear(hidden_dim, dim, num_blocks)
        )

        # Concept Memory
        self.concept_vector = nn.Parameter(torch.Tensor(1, num_prototypes, num_blocks, self.block_dim))
        nn.init.trunc_normal_(self.concept_vector)
        self.concept_memory = nn.Sequential(       
            nn.Linear(self.block_dim, self.block_dim*4), 
            nn.ReLU(),
            nn.Linear(self.block_dim*4, self.block_dim*4),
            nn.ReLU(),
            nn.Linear(self.block_dim*4, self.block_dim*4),
            nn.ReLU(),
            nn.Linear(self.block_dim*4, self.block_dim)
        )

        # norms
        self.norm_input  = nn.LayerNorm(in_features)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = BlockLayerNorm(dim, num_blocks)

        self.norm_concept = BlockLayerNorm(dim, num_blocks)
        self.norm_bq = BlockLayerNorm(dim, num_blocks) # query normalization for block-wise attention

    def block_wise_attention(self, q, k, v):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """

        B, T, _ = q.shape
        _, S, _ = k.shape

        q = q.view(B, T, self.num_blocks, -1).transpose(1, 2)
        k = k.view(B, S, self.num_blocks, -1).transpose(1, 2)
        v = v.view(B, S, self.num_blocks, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)

        return output

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
            scaled_dots = (self.slot_dim ** -0.5) * torch.matmul(k, q.transpose(2, 1))
            attn = scaled_dots.softmax(dim=1) + self.eps
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (b, n, n_s))

            # Weighted Mean.
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_dim].
            assert_shape(updates.size(), (b, n_s, self.slot_dim))

            # Block-Slot Update in parallel
            slots = self.gru(                         
                updates.view(b * n_s, self.slot_dim),
                slots_prev.view(b * n_s, self.slot_dim),
            )
            slots = slots.view(b, n_s, self.slot_dim)
            assert_shape(slots.size(), (b, n_s, self.slot_dim))
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            assert_shape(slots.size(), (b, n_s, self.slot_dim))
            
            concept_mem = self.concept_memory(self.concept_vector).reshape(self.num_prototypes, -1).unsqueeze(0)
            concept_mem = self.norm_concept(concept_mem).expand(slots.size(0), -1, -1)
            slots = self.block_wise_attention(self.norm_bq(slots), concept_mem, concept_mem)            
            
        return slots, attn


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.cnn = nn.Sequential(
            Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(args.d_model, args.image_size if args.image_size == 64 else args.image_size // 2)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_model, args.d_model))

        self.sysbinder = SysBinder(
            in_features=args.d_model,  
            num_slots=args.num_slots, num_blocks=args.num_blocks, num_prototypes=args.num_prototypes, 
            dim=args.slot_size, iters=args.num_iterations, hidden_dim=args.mlp_hidden_size, )


class ImageDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_proj = BlockLinear(args.slot_size, args.d_model * args.num_blocks, args.num_blocks)

        self.block_pos = nn.Parameter(torch.zeros(1, 1, args.d_model * args.num_blocks), requires_grad=True)
        self.block_pos_proj = nn.Sequential(
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks),
            nn.ReLU(),
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks)
        )

        self.block_coupler = TransformerEncoder(num_blocks=1, d_model=args.d_model, num_heads=4)

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)

        self.decoder_pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)

        self.tf = TransformerDecoder(
            args.num_decoder_layers, (args.image_size // 4) ** 2, args.d_model, args.num_decoder_heads, args.dropout)

        self.head = linear(args.d_model, args.vocab_size, bias=False)


class SysBinderImageAutoEncoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.num_prototypes = args.num_prototypes
        self.image_channels = args.img_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.num_blocks = args.num_blocks

        # dvae
        self.dvae = dVAE(args.vocab_size, args.img_channels)

        # encoder networks
        self.image_encoder = ImageEncoder(args)

        # decoder networks
        self.image_decoder = ImageDecoder(args)

    def forward(self, image, tau):
        """
        image: B, C, H, W
        tau: float
        """
        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)  # B, vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(z_logits, tau, False, dim=1)  # B, vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # B, vocab_size, H_enc, W_enc
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H_enc * W_enc, vocab_size
        z_emb = self.image_decoder.dict(z_hard)  # B, H_enc * W_enc, d_model
        z_emb = torch.cat([self.image_decoder.bos.expand(B, -1, -1), z_emb], dim=1)  # B, 1 + H_enc * W_enc, d_model
        z_emb = self.image_decoder.decoder_pos(z_emb)  # B, 1 + H_enc * W_enc, d_model

        # dvae recon
        dvae_recon = self.dvae.decoder(z_soft).reshape(B, C, H, W)  # B, C, H, W
        dvae_mse = ((image - dvae_recon) ** 2).sum() / B  # 1

        # sysbinder
        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

        slots, attns = self.image_encoder.sysbinder(emb_set)  # slots: B, num_slots, slot_size
                                                              # attns: B, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
        attns = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, self.num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, self.num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        # decode
        pred = self.image_decoder.tf(z_emb[:, :-1], slots)   # B, H_enc * W_enc, d_model
        pred = self.image_decoder.head(pred)  # B, H_enc * W_enc, vocab_size
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / B  # 1

        return (dvae_recon.clamp(0., 1.),
                cross_entropy,
                dvae_mse,
                attns)

    def encode(self, image):
        """
        image: B, C, H, W
        """
        B, C, H, W = image.size()

        # sysbinder
        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

        slots, attns = self.image_encoder.sysbinder(emb_set)  # slots: B, num_slots, slot_size
                                                              # attns: B, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
        attns_vis = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W
        
        return slots, attns_vis, attns

    def decode(self, slots):
        """
        slots: B, N, slot_size
        """
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.image_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.image_decoder.tf(
                self.image_decoder.decoder_pos(input),
                slots
            )
            z_next = F.one_hot(self.image_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.image_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, image):
        """
        image: batch_size x image_channels x H x W
        """
        B, C, H, W = image.size()
        slots, attns, _ = self.encode(image)
        recon_transformer = self.decode(slots)
        recon_transformer = recon_transformer.reshape(B, C, H, W)

        return recon_transformer
