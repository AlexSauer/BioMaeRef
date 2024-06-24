# This code is partially adopted from the repo https://github.com/facebookresearch/mae

from functools import partial
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from einops.layers.torch import Rearrange
import lightning as L

from timm.models.vision_transformer import PatchEmbed, Block

def save_checkpoint(path, state_dict, name):
    filename = os.path.join(path, name)
    torch.save(state_dict, filename)
    print("Saving checkpoint:", filename)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    code is adopted from https://github.com/facebookresearch/mae
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        pos_encode_w=1.0,
        entropy_loss_threshold = 0.4,
        entropy_patch_threshold = 0.4,
        patch_perc_keep = 1.0,
        weights=None,
    ):
        super().__init__()
        self.n_classes = 5
        self.closest_patch_perc_keep = patch_perc_keep

        self.entropy_loss_threshold = entropy_loss_threshold
        # MAE encoder specifics

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pos_encode_w = pos_encode_w
        self.weights = torch.Tensor(weights) if weights is not None else None
        print("img_size:", img_size)
        print("patch_size:", patch_size)
    
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Sequential(
            nn.Linear( decoder_embed_dim, patch_size**2 * self.n_classes, bias=True),  # decoder to patch
            Rearrange('b l (p c) -> b l p c', p=patch_size**2, c=self.n_classes)
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, dist_patches=None, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * self.closest_patch_perc_keep * (1 - mask_ratio))  # After removing the distant patches, we keep the (1-mask_ratio)% of the patches
        noise_provided = noise is not None

        if noise is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        if dist_patches is not None:
            id_dist_sort = torch.argsort(dist_patches, dim=1)
            id_dist_remove = id_dist_sort[:, int(id_dist_sort.shape[1] * self.closest_patch_perc_keep):]  # Only keep the the closest_patch_perc_keep% closest patches
            noise = noise.scatter(dim=1, index=id_dist_remove, src=torch.rand(noise.shape, device=x.device)+1)

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # If the noise is provided, we set the noise of the patches that we kept in this round to 1
        # So that they will be removed in the next round because it puts them 'at the end of the queue'
        if noise_provided:
            noise = noise.scatter(dim=1, index=ids_keep, src=torch.ones_like(noise))

        return x_masked, mask, ids_restore, noise

    def forward_encoder(self, x, dist_patches, mask_ratio, noise=None):
        # embed patches
        x = self.patch_embed(x)  # [N, L, D] where D is the embedding dimensions

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] * self.pos_encode_w

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, noise = self.random_masking(x, mask_ratio, dist_patches, noise=noise)  
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, noise

    def forward_decoder(self, x, ids_restore, dist_patches=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed  # Shape: [N, 1+L, D_decoder]

        # Delete the background patches
        if dist_patches is not None:    

            # Append True to the dist_patches to account for the cls token
            org_number_patches = dist_patches.shape[1]
            # dist_patches = torch.cat([torch.zeros(dist_patches.shape[0], 1, device=dist_patches.device), dist_patches], dim=1)

            # Remove the x% patches with the highest distance to any mitochondria
            # (I need to do it this way because I need to have the same number of patches for each image in the batch)
            id_dist_sort = torch.argsort(dist_patches, dim=1)
            id_dist_keep = id_dist_sort[:, :int(id_dist_sort.shape[1] * self.closest_patch_perc_keep)]

            # Adjust for the class token
            id_dist_keep += 1
            id_dist_keep = torch.cat([torch.zeros(id_dist_keep.shape[0], 1, device=id_dist_keep.device, dtype=id_dist_keep.dtype), 
                                      id_dist_keep], dim=1)

            x = torch.gather(x, dim=1, index=id_dist_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]  # torch.Size([N, (N_patches), patch_size**2, n_classes])

        if dist_patches is not None:
            # Add the background patches back
            n_classes = x.shape[-1]
            final_shape = (x.shape[0], org_number_patches, self.patch_size**2, n_classes)
            back_pred = torch.zeros(final_shape, device=x.device)
            back_pred[..., 0] = 1e3

            # Adjust the size
            id_dist_keep = id_dist_keep[:, 1:] - 1  # Adjust for the class token
            id = id_dist_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
            x = back_pred.scatter(dim=1, index=id, src=x)
        else:
            id_dist_keep =  None
        
        return x, id_dist_keep

    def forward_loss(self, imgs, pred, mask, entropy, id_dist_keep=None):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1, n_classes]
        mask: [N, L], 0 is keep, 1 is remove,
        entropy: [N, 1, H, W]
        """
        target = self.patchify(imgs)
        entropy = self.patchify(entropy)

        if id_dist_keep is not None:
            kept = torch.zeros_like(target)
            masked_out = kept.scatter(dim=1, index=id_dist_keep.unsqueeze(-1).repeat(1,1,target.shape[-1]), src=torch.ones_like(target)) == 0
            target[masked_out] = -1
        
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.flatten()
        target[entropy.flatten() > self.entropy_loss_threshold] = -1 
        loss = cross_entropy(pred, target.long(), weight=self.weights, ignore_index=-1, reduction="mean")
        return loss

    def forward(self, imgs, entropy, dist=None, mask_ratio=0.75):
        if dist is not None:
            dist = F.unfold(dist, kernel_size=self.patch_size, stride=self.patch_size, padding=0) # N, patch_size**2, L
            dist = dist.mean(dim=1)
        latent, mask, ids_restore, _ = self.forward_encoder(imgs, dist_patches=dist, mask_ratio=mask_ratio)
        pred, id_dist_keep = self.forward_decoder(latent, ids_restore, dist_patches=dist)  # [N, L, p*p*1, n_classes]
        loss = self.forward_loss(imgs, pred, mask, entropy, id_dist_keep)
        return loss, pred, mask

    @torch.no_grad()
    def forward_full_pred_with_mask(self, img, mask=None):
        """
        Boolean Mask: True is masked/removed, False is kept
        """
        # Use the other function by creating a fake entropy which 
        fake_entropy = torch.randn_like(img)
        fake_entropy[mask==1] += 3
        preds = []
        for ind in range(img.shape[0]):
            perc_masked = (mask[ind]==1).float().mean()
            cur_pred, masks = self.forward_full_pred(img[ind][None], dist=None, entropy=fake_entropy[ind][None], perc_masked=perc_masked) 
            preds.append(cur_pred[0])
        return preds

    @torch.no_grad()
    def forward_full_pred(self, img, entropy=None, dist=None, perc_masked=0.5):
        preds, masks = [], []        

        if dist is not None:
            dist = F.unfold(dist, kernel_size=self.patch_size, stride=self.patch_size, padding=0) # N, patch_size**2, L
            dist = dist.mean(dim=1)
        else:
            noise = None

        # I want to make sure that the low entropy patches are in the first partition, therefore, I will increase the noise 
        # for the high entropy patches so that they will not be considered in the first partition
        if entropy is not None:
            entropy = F.unfold(entropy, kernel_size=self.patch_size, stride=self.patch_size, padding=0) # indN, patch_size**2, L
            entropy = entropy.mean(dim=1)

            N, L = entropy.shape  # batch, length, dim
            noise = torch.rand(N, L, device=img.device)  # noise in [0, 1]
            
            # Find the patches which will never be considered because their distance is too high
            if dist is not None:
                # noise_mod = noise.clone()
                id_dist_sort = torch.argsort(dist, dim=1)
                id_dist_remove = id_dist_sort[:, int(id_dist_sort.shape[1] * self.closest_patch_perc_keep):]  # Only keep the the patch_perc_keep% closest patches
                entropy = entropy.scatter(dim=1, index=id_dist_remove, src=torch.ones_like(entropy)*2)
                # Now, the very far patches will be be at the very end of argsort because they are considered 'highest entropy'
                # and it is enough for me to reorder the first closest_patch_perc_keep% of the patches

            # Make sure that perc_masked% of the patches are removed (masked) in the first partition.
            # Therefore, I leave the noise level of the (1-perc_masked)% smallest entropy patches untouched
            # but increase the noise of the (perc_masked)% highest entropy patches (and always need to keep in touch that we don't work with all patches but only cloest_patch_perc_keep%)
            id_dist_sort = torch.argsort(entropy, dim=1)
            id_dist_remove = id_dist_sort[:, int(id_dist_sort.shape[1] * self.closest_patch_perc_keep * (1-perc_masked)):\
                                                int(id_dist_sort.shape[1] * self.closest_patch_perc_keep)]
            
            noise_add = torch.zeros_like(noise)
            noise_add = noise_add.scatter(dim=1, index=id_dist_remove, src=torch.ones_like(noise)*2) # For debugging
            noise = noise + noise_add 

        latent, mask, ids_restore, noise = self.forward_encoder(img, dist_patches=dist, mask_ratio=perc_masked, noise=noise)
        pred, id_dist_keep = self.forward_decoder(latent, ids_restore, dist_patches=dist)  # [N, L, p*p*1, n_classes]
        
        pred = self.unpatchify(pred.argmax(-1))
        preds.append(pred)
        
        mask_img = self.unpatchify(mask[..., None].expand(mask.shape[0], #cfg['DATASET']['batch_size'], 
                                            mask.shape[1],
                                            self.patch_size**2)).cpu()
        masks.append(mask_img)
        
        return preds, masks

    @torch.no_grad()
    def forward_full_pred_mult_part(self, img, entropy=None, dist=None, n_partitions=4):
        if dist is not None:
            dist = F.unfold(dist, kernel_size=self.patch_size, stride=self.patch_size, padding=0) # N, patch_size**2, L
            dist = dist.mean(dim=1)

        preds, masks = [], []        

        N, L = dist.shape  # batch, length, dim
        noise = torch.rand(N, L, device=img.device)  # noise in [0, 1]

        # I want to make sure that the low entropy patches are in the first partition, therefore, I will increase the noise 
        # for the high entropy patches so that they will not be considered in the first partition
        if entropy is not None:
            entropy = F.unfold(entropy, kernel_size=self.patch_size, stride=self.patch_size, padding=0) # N, patch_size**2, L
            entropy = entropy.mean(dim=1)
            
            # Find the patches which will never be considered because their distance is too high
            if dist is not None:
                # noise_mod = noise.clone()
                id_dist_sort = torch.argsort(dist, dim=1)
                id_dist_remove = id_dist_sort[:, int(id_dist_sort.shape[1] * (1-self.closest_patch_perc_keep)):]  # Only keep the the patch_perc_keep% closest patches
                entropy = entropy.scatter(dim=1, index=id_dist_remove, src=torch.ones_like(entropy)*10)

                # Now, the very far patches will be be at the very end of argsort


            id_dist_sort = torch.argsort(entropy, dim=1)

            for quantile in range(1, n_partitions):
                id_dist_remove = id_dist_sort[:, int(id_dist_sort.shape[1] * self.closest_patch_perc_keep * quantile/n_partitions):int(id_dist_sort.shape[1] * self.closest_patch_perc_keep * (quantile+1)/n_partitions)]
                noise_add = torch.zeros_like(noise)
                noise_add = noise_add.scatter(dim=1, index=id_dist_remove, src=torch.ones_like(noise))
                # noise = torch.clip(noise + noise_add * (quantile/n_partitions)*0.5, 0, 0.95)
                noise = torch.clip(noise + noise_add, 0, 0.95)  # Extreme for debugging

        for _ in range(n_partitions):
            latent, mask, ids_restore, noise = self.forward_encoder(img, dist_patches=dist, mask_ratio=1-1/n_partitions, noise=noise)
            pred, id_dist_keep = self.forward_decoder(latent, ids_restore, dist_patches=dist)  # [N, L, p*p*1, n_classes]
            
            pred = self.unpatchify(pred.argmax(-1))
            preds.append(pred)
            
            mask_img = self.unpatchify(mask[..., None].expand(mask.shape[0], #cfg['DATASET']['batch_size'], 
                                                mask.shape[1],
                                                self.patch_size**2)).cpu()
            masks.append(mask_img)
        
        return preds, masks

class MAE_Module(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        cfg_model = cfg['MODEL']
        self.model = MAE(
            img_size=cfg_model['img_size'], 
            in_chans=1,
            patch_size=cfg_model['patch_size'], 
            embed_dim=cfg_model["embed_dim"], 
            depth=cfg_model["depth"], 
            num_heads=cfg_model["num_heads"],
            decoder_embed_dim=cfg_model["decoder_embed_dim"], 
            decoder_depth=cfg_model["decoder_depth"],
            decoder_num_heads=cfg_model["decoder_num_heads"],
            mlp_ratio=cfg_model["mlp_ratio"],
            pos_encode_w = cfg_model["pos_encode_w"],
            entropy_loss_threshold = cfg_model["entropy_loss_threshold"], 
            entropy_patch_threshold = cfg_model["entropy_pixel_threshold"],
            patch_perc_keep = cfg_model.get("patch_perc_keep", 1.),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            weights=cfg_model.get("weights", None),
        )

    def training_step(self, batch, batch_idx):
        mito, entropy, dist = batch
        self.model.weights = self.model.weights.to(self.device)
        mask_ratio = np.random.uniform(0.1, 0.9)
        loss, _, _ = self.model(mito, entropy, dist=dist, mask_ratio=mask_ratio)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.5e-4, weight_decay=1e-5)
        return optimizer
