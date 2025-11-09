# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_upsample_layer, xavier_init
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16

from ..builder import NECKS
# NOTA: NO importamos nada de .fpn (evita ImportError)

# --------------------------
# FPN_CARAFE (base, fiel al original)
# --------------------------
@NECKS.register_module()
class FPN_CARAFE(BaseModule):
    """FPN con soporte de upsample flexible (nearest/bilinear/deconv/pixel_shuffle/CARAFE)."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(
                     type='carafe',
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1),
                 init_cfg=None):
        assert init_cfg is None, (
            'Para evitar inicialización anómala, init_cfg no se permite aquí.')
        super(FPN_CARAFE, self).__init__(init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None

        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert 'upsample_kernel' in self.upsample_cfg and self.upsample_cfg['upsample_kernel'] > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.upsample_modules = ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # 1x1 lateral
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)

            # 3x3 salida por nivel
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)

            # upsample para camino top-down (no en el tope)
            if i != self.backbone_end_level - 1:
                up_cfg = self.upsample_cfg.copy()
                if self.upsample == 'deconv':
                    up_cfg.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    up_cfg.update(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe':
                    up_cfg.update(channels=out_channels, scale_factor=2)
                else:
                    align_corners = (None if self.upsample == 'nearest' else False)
                    up_cfg.update(scale_factor=2, mode=self.upsample, align_corners=align_corners)
                self.upsample_modules.append(build_upsample_layer(up_cfg))

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # niveles extra (e.g., para P6, P7)
        extra_out_levels = (num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_ch = self.in_channels[self.backbone_end_level - 1] if i == 0 else out_channels
                extra_l_conv = ConvModule(
                    in_ch, out_channels, 3, stride=2, padding=1,
                    norm_cfg=norm_cfg, bias=self.with_bias, act_cfg=act_cfg,
                    inplace=False, order=self.order)
                extra_fpn_conv = ConvModule(
                    out_channels, out_channels, 3, padding=1,
                    norm_cfg=self.norm_cfg, bias=self.with_bias, act_cfg=act_cfg,
                    inplace=False, order=self.order)
                # mantenemos la lista simétrica para el hijo
                self.lateral_convs.append(extra_l_conv)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        super().init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    @staticmethod
    def _slice_as(src, dst):
        if src.size()[2:] == dst.size()[2:]:
            return src
        return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        return a + (b if a.size() == b.size() else self._slice_as(b, a))

    def forward(self, inputs):
        """FPN estándar con upsample (se deja por compatibilidad)."""
        assert len(inputs) == len(self.in_channels)

        laterals = []
        for i, l_conv in enumerate(self.lateral_convs[:self.backbone_end_level - self.start_level]):
            inp = inputs[i + self.start_level]
            laterals.append(l_conv(inp))

        # top-down
        for i in range(len(laterals) - 1, 0, -1):
            up_feat = self.upsample_modules[i - 1](laterals[i])
            laterals[i - 1] = self.tensor_add(laterals[i - 1], up_feat)

        # salidas base
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # niveles extra vía max-pool (si hacen falta)
        while len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return tuple(outs)


# --------------------------
# FreqFusionCARAFEFPN (sustituye la fusión top-down pero conserva #salidas)
# --------------------------
from .FreqFusion import FreqFusion

@NECKS.register_module()
class FreqFusionCARAFEFPN(FPN_CARAFE):
    """Como FPN_CARAFE, pero la suma top-down se reemplaza por FreqFusion."""

    def __init__(self,
                 use_high_pass=True,
                 use_low_pass=True,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 compress_ratio=8,
                 semi_conv=True,
                 feature_resample=False,
                 feature_resample_group=8,
                 comp_feat_upsample=True,
                 feature_resample_norm=True,
                 **kwargs):
        super().__init__(**kwargs)
        # no usamos upsample_modules del padre
        if hasattr(self, 'upsample_modules'):
            del self.upsample_modules

        self.alignment = nn.ModuleList()
        # un módulo por “salto” top-down
        n_levels = self.backbone_end_level - self.start_level
        for _ in range(n_levels):
            self.alignment.append(FreqFusion(
                hr_channels=self.out_channels,
                lr_channels=self.out_channels,
                scale_factor=1,
                lowpass_kernel=lowpass_kernel,
                highpass_kernel=highpass_kernel,
                up_group=1,
                upsample_mode='nearest',
                align_corners=False,
                feature_resample=feature_resample,
                feature_resample_group=feature_resample_group,
                hr_residual=True,
                comp_feat_upsample=comp_feat_upsample,
                compressed_channels=(2 * self.out_channels) // compress_ratio,
                use_high_pass=use_high_pass,
                use_low_pass=use_low_pass,
                semi_conv=semi_conv,
                feature_resample_norm=feature_resample_norm))

    def init_weights(self):
        super().init_weights()
        for m in self.modules():
            if isinstance(m, FreqFusion):
                m.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # laterales 1x1
        laterals = []
        for i in range(self.backbone_end_level - self.start_level):
            laterals.append(self.lateral_convs[i](inputs[i + self.start_level]))

        # top-down con FreqFusion (alinea y suma)
        for i in range(len(laterals) - 1, 0, -1):
            _, hi, lo = self.alignment[i - 1](
                hr_feat=laterals[i - 1], lr_feat=laterals[i], use_checkpoint=False)
            laterals[i - 1] = self.tensor_add(hi, lo)

        # salidas base
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # niveles extra si num_outs > niveles del backbone
        while len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return tuple(outs)
