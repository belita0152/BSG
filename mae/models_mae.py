# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone for EEG """
    def __init__(self, img_size=(50, 1000), patch_size=(10,10), in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        # 이미지를 ViT에서 사용할 patch 단위로 나누어 embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.img_size = img_size
        self.patch_size = patch_size

        # class 예측을 위한 토큰 (Encoder에만 사용)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 각 patch 위치 정보를 주기 위해 sin-cos 기반 고정 positional encoding 생성
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        # transformer encoder block들 (ViT 구조)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])

        # 최종 encoder 출력 정규화
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        # encoder의 출력(latent)을 decoder 차원으로 변환
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # 마스킹된 위치에 삽입될 학습 가능한 벡터
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # decoder에서도 positional encoding 사용
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # transformer decoder block들 (얕고 작게 구성됨)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # decoder 출력 → 픽셀값 예측
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # 손실 계산 시 patch를 픽셀 단위로 정규화할지를 결정
        self.norm_pix_loss = norm_pix_loss

        # pos embed 고정값 설정, 학습 가능한 값 초기화
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed, decoder_pos_embed → sin-cos positional encoding 으로 고정값 채움
        H, W = self.patch_embed.grid_size
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (H, W), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (H, W), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # patch embedding weight → Xavier 초기화
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # mask token, cls token → 정규 분포 초기화 (std=0.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # 전체 모델 내 Linear/LayerNorm → _init_weights() 함수로 순회하며 초기화
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


    # patchify(imgs) → 이미지 → (N, L, p*p*1) 텐서로 변환
    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        - 이미지 → patch 단위로 쪼갬
        - 각 patch를 1D 벡터로 flatten
        x: (N, L, patch_size**2 *1)
        """
        p1, p2 = self.patch_size
        H, W = self.patch_embed.grid_size
        assert imgs.shape[2] == H * p1 and imgs.shape[3] == W * p2
        x = imgs.reshape(shape=(imgs.shape[0], 1, H, p1, W, p2))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], H * W, p1 * p2 * 1))

        return x


    # unpatchify(x) → 복원된 patch 벡터 → 이미지로 재조립
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p1, p2 = self.patch_size
        H, W = self.patch_embed.grid_size
        x = x.reshape(shape=(x.shape[0], H, W, p1, p2, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, H * p1, W * p2))

        return imgs


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # 각 샘플에 대해 패치 순서를 랜덤하게 섞기
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # noise = torch.rand(N, L) → argsort → ids_shuffle
        # 각 샘플에 대한 노이즈 정렬
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 상위 mask_ratio만큼의 patch를 제거
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 이진 마스크 생성: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # 이진 마스크를 get 하기 위해 unshuffle
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_latent(self, x):
        # 이미지를 patch로 나눈 뒤 embedding
        x = self.patch_embed(x)

        # cls token 없이 positional encoding 추가
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]   # CLS token을 처음 위치에 붙임
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)       # 배치 크기만큼 복제해서 확장
        x = torch.cat((cls_tokens, x), dim=1)

        # transformer block을 거쳐 전체 latent representation 계산
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # CLS token만 추출
        x = x[:, :1, :].squeeze()
        return x


    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # patch_embed: 이미지 패치를 embedding
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # pos_embed 추가 (단, cls token 제외)
        x = x + self.pos_embed[:, 1:, :]

        # masking: length → length * mask_ratio
        # random_masking 수행 → 일부 패치만 유지
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token을 첫 위치에 붙임
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # transformer encoder 통과
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)    # norm 정규화 후 latent 반환

        return x, mask, ids_restore


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # latent를 decoder embedding으로 차원 변환
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask token 생성 후 마스킹된 위치에 삽입
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # decoder_pos_embed 더함
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        # decoder transformer 통과
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        # decoder_pred로 픽셀값 예측
        x = self.decoder_pred(x)

        # remove cls token
        # cls token 제거 후 예측 결과 반환
        x = x[:, 1:, :]

        return x


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        # 입력 이미지를 patchify → target 생성
        target = self.patchify(imgs)    # patchify(imgs): 입력 이미지를 patch 단위로 쪼갬
        # (선택적 정규화) 평균/표준편차로 정규화
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        # pred와 target 간 MSE 계산
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # 마스킹된 패치에 대해서만 손실 평균 계산
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, imgs, mask_ratio=0.75):
        # encoder, decoder, loss를 순차적으로 호출하여 최종 학습 출력 변환
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=8, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks