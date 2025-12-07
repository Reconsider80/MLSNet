import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# ==================== FRD (Feature Recalibration Decoder) Block ====================
class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(x_cat)
        return self.sigmoid(attention)

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class DualAttention(nn.Module):
    """双注意力机制（空间+通道）"""
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa
        
        # 残差连接
        out = x + x_sa
        return out

class MultiScaleFeatureAggregation(nn.Module):
    """多尺度特征聚合"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 不同尺度的卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4)
        
        # 自适应权重学习
        self.weight_net = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 4, 1),
            nn.Softmax(dim=1)
        )
        
        # 融合卷积
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, 1)
        
    def forward(self, x):
        # 提取多尺度特征
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        f4 = self.conv4(x)
        
        # 拼接特征
        features = torch.cat([f1, f2, f3, f4], dim=1)
        
        # 学习自适应权重
        weights = self.weight_net(features)
        weights = weights.chunk(4, dim=1)
        
        # 加权融合
        weighted_f1 = f1 * weights[0]
        weighted_f2 = f2 * weights[1]
        weighted_f3 = f3 * weights[2]
        weighted_f4 = f4 * weights[3]
        
        # 最终融合
        fused = self.fusion(torch.cat([weighted_f1, weighted_f2, weighted_f3, weighted_f4], dim=1))
        return fused

class FeatureRecalibrationUnit(nn.Module):
    """特征重校准单元"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 全局上下文信息提取
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 局部特征增强
        self.local_enhance = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 校准门控
        self.calibration_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 全局上下文
        global_context = self.global_context(x)
        
        # 局部特征增强
        local_features = self.local_enhance(x)
        
        # 上下文指导的特征校准
        context_guided = local_features * global_context
        
        # 门控融合
        gate_input = torch.cat([local_features, context_guided], dim=1)
        gate = self.calibration_gate(gate_input)
        
        # 重校准输出
        recalibrated = local_features * gate + context_guided * (1 - gate)
        
        return recalibrated

class CrossScaleFeatureFusion(nn.Module):
    """跨尺度特征融合"""
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        
        # 低层特征处理（细节丰富）
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 高层特征处理（语义丰富）
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 跨尺度注意力
        self.cross_attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, low_feat, high_feat):
        # 处理特征
        low_proc = self.low_conv(low_feat)
        high_proc = self.high_conv(high_feat)
        
        # 确保尺寸匹配
        if high_proc.shape[2:] != low_proc.shape[2:]:
            high_proc = F.interpolate(high_proc, size=low_proc.shape[2:], mode='bilinear', align_corners=True)
        
        # 计算跨尺度注意力
        combined = torch.cat([low_proc, high_proc], dim=1)
        attention = self.cross_attention(combined)
        
        # 注意力指导的融合
        attended_low = low_proc * attention
        attended_high = high_proc * (1 - attention)
        
        # 最终融合
        fused = self.fusion(torch.cat([attended_low, attended_high], dim=1))
        
        return fused

class FRDBlock(nn.Module):
    """特征重校准解码器块 (FRD Block)"""
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # 上采样层
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 跳跃连接处理
        if skip_channels > 0:
            self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)
            self.skip_norm = nn.BatchNorm2d(out_channels)
            
            # 跳跃连接融合
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # 多尺度特征聚合
        self.multi_scale = MultiScaleFeatureAggregation(out_channels, out_channels)
        
        # 双注意力机制
        if use_attention:
            self.dual_attention = DualAttention(out_channels)
        
        # 特征重校准单元
        self.recalibration = FeatureRecalibrationUnit(out_channels, out_channels)
        
        # 残差连接
        self.residual_conv = nn.Conv2d(out_channels, out_channels, 1)
        
        # 输出处理
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x, skip=None):
        # 上采样
        x_up = self.upsample(x)
        
        # 跳跃连接处理
        if skip is not None and hasattr(self, 'skip_conv'):
            # 处理跳跃连接
            skip_proc = self.skip_norm(self.skip_conv(skip))
            
            # 确保尺寸匹配
            if skip_proc.shape[2:] != x_up.shape[2:]:
                skip_proc = F.interpolate(skip_proc, size=x_up.shape[2:], mode='bilinear', align_corners=True)
            
            # 融合跳跃连接
            x_fused = self.skip_fusion(torch.cat([x_up, skip_proc], dim=1))
        else:
            x_fused = x_up
        
        # 多尺度特征聚合
        x_ms = self.multi_scale(x_fused)
        
        # 双注意力机制
        if self.use_attention:
            x_att = self.dual_attention(x_ms)
        else:
            x_att = x_ms
        
        # 特征重校准
        x_rc = self.recalibration(x_att)
        
        # 残差连接
        residual = self.residual_conv(x_fused)
        x_res = x_rc + residual
        
        # 输出处理
        output = self.output(x_res)
        
        return output

# ==================== 重新定义必要的组件 ====================
class SAM2AdapterLayer(nn.Module):
    """SAM2适配器层"""
    def __init__(self, in_channels, out_channels, embed_dim=256, depth=2, scale_factor=2):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # 输入处理
        if scale_factor > 1:
            self.input_conv = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, 3, stride=scale_factor, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(inplace=True)
            )
        else:
            self.input_conv = nn.Conv2d(in_channels, embed_dim // 2, 3, padding=1)
        
        # 简化的注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 输出转换
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 如果需要上采样
        if scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Identity()
    
    def forward(self, x):
        # 输入处理
        x_in = self.input_conv(x)
        
        # 注意力机制
        attn = self.attention(x_in)
        x_att = x_in * attn + x_in
        
        # 输出转换
        x_out = self.output_conv(x_att)
        
        # 上采样
        x_out = self.upsample(x_out)
        
        return x_out

class VKANBlock(nn.Module):
    """简化的VKAN块"""
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.conv3 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.drop = nn.Dropout2d(drop)
        
    def forward(self, x):
        # 第一层
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = x + identity
        
        # 第二层
        identity = x
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = x + identity
        
        return x

class VKANEncoder(nn.Module):
    """VKAN编码器"""
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(*[
            VKANBlock(out_channels)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x

class VKANDecoder(nn.Module):
    """VKAN解码器"""
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=2):
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if skip_channels > 0:
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.skip_fusion = None
        
        self.blocks = nn.Sequential(*[
            VKANBlock(out_channels)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None and self.skip_fusion is not None:
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fusion(x)
        
        x = self.blocks(x)
        return x

class LearnableSkipConnection(nn.Module):
    """可学习跳跃连接"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, skip_feature, decoder_feature):
        transformed_skip = self.transform(skip_feature)
        attention = self.attention(skip_feature)
        return transformed_skip * attention + decoder_feature

# ==================== 完整的MLSN_LSC_SAM2_VKAN_FRD网络 ====================
class MLSN_LSC_SAM2_VKAN_FRD(nn.Module):
    """完整的MLSN_LSC_SAM2_VKAN_FRD网络"""
    def __init__(self, n_classes=1, in_channels=3, base_channels=64, img_size=256):
        super().__init__()
        
        self.n_classes = n_classes
        self.base_channels = base_channels
        
        # ========== 编码器部分 ==========
        # 第1级：SAM2增强编码器
        self.encoder1 = SAM2AdapterLayer(
            in_channels=in_channels,
            out_channels=base_channels,
            embed_dim=128,
            scale_factor=1
        )
        
        # 第2级：SAM2增强编码器
        self.encoder2 = SAM2AdapterLayer(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            embed_dim=256,
            scale_factor=2
        )
        
        # 第3级：VKAN编码器
        self.encoder3 = VKANEncoder(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            num_blocks=2
        )
        
        # 第4级：VKAN编码器
        self.encoder4 = VKANEncoder(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            num_blocks=2
        )
        
        # ========== 瓶颈层 ==========
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            VKANBlock(base_channels * 8),
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True)
        )
        
        # ========== 可学习跳跃连接 ==========
        self.skip4 = LearnableSkipConnection(base_channels * 8, base_channels * 8)
        self.skip3 = LearnableSkipConnection(base_channels * 4, base_channels * 4)
        self.skip2 = LearnableSkipConnection(base_channels * 2, base_channels * 2)
        self.skip1 = LearnableSkipConnection(base_channels, base_channels)
        
        # ========== 解码器部分 ==========
        # 第4级解码器：VKAN解码器
        self.decoder4 = VKANDecoder(
            in_channels=base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
            num_blocks=2
        )
        
        # 第3级解码器：VKAN解码器
        self.decoder3 = VKANDecoder(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
            num_blocks=2
        )
        
        # 第2级解码器：FRD解码器
        self.decoder2 = FRDBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            use_attention=True
        )
        
        # 第1级解码器：FRD解码器
        self.decoder1 = FRDBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            use_attention=True
        )
        
        # ========== 输出层 ==========
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, n_classes, 1)
        )
        
        # ========== 深度监督 ==========
        self.ds4 = nn.Conv2d(base_channels * 8, n_classes, 1)
        self.ds3 = nn.Conv2d(base_channels * 4, n_classes, 1)
        self.ds2 = nn.Conv2d(base_channels * 2, n_classes, 1)
        self.ds1 = nn.Conv2d(base_channels, n_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ========== 编码器前向传播 ==========
        enc1 = self.encoder1(x)        # [B, 64, H, W]
        enc2 = self.encoder2(enc1)     # [B, 128, H/2, W/2]
        enc3 = self.encoder3(enc2)     # [B, 256, H/4, W/4]
        enc4 = self.encoder4(enc3)     # [B, 512, H/8, W/8]
        
        # ========== 瓶颈层 ==========
        bottleneck = self.bottleneck(enc4)  # [B, 1024, H/16, W/16]
        
        # ========== 解码器前向传播 ==========
        # 第4级解码
        dec4 = self.decoder4(bottleneck, enc4)
        dec4_input = self.skip4(enc4, dec4)
        ds4 = self.ds4(dec4_input)
        
        # 第3级解码
        dec3 = self.decoder3(dec4_input, enc3)
        dec3_input = self.skip3(enc3, dec3)
        ds3 = self.ds3(dec3_input)
        
        # 第2级解码 - FRD解码器
        dec2_input = self.skip2(enc2, dec3_input)
        dec2 = self.decoder2(dec2_input, enc2)
        ds2 = self.ds2(dec2)
        
        # 第1级解码 - FRD解码器
        dec1_input = self.skip1(enc1, dec2)
        dec1 = self.decoder1(dec1_input, enc1)
        ds1 = self.ds1(dec1)
        
        # ========== 最终输出 ==========
        output = self.output(dec1)
        
        # ========== 深度监督输出 ==========
        ds_outputs = {
            'level4': F.interpolate(ds4, size=x.shape[2:], mode='bilinear', align_corners=True),
            'level3': F.interpolate(ds3, size=x.shape[2:], mode='bilinear', align_corners=True),
            'level2': F.interpolate(ds2, size=x.shape[2:], mode='bilinear', align_corners=True),
            'level1': F.interpolate(ds1, size=x.shape[2:], mode='bilinear', align_corners=True),
            'final': output
        }
        
        return ds_outputs

# ==================== 增强的FRD解码器版本 ====================
class EnhancedFRDBlock(nn.Module):
    """增强版FRD解码器块"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # 上采样层
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 跨尺度特征融合
        if skip_channels > 0:
            self.cross_fusion = CrossScaleFeatureFusion(
                low_channels=skip_channels,
                high_channels=out_channels,
                out_channels=out_channels
            )
        
        # 特征金字塔处理
        self.pyramid_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(3)
        ])
        
        # 多尺度上下文聚合
        self.context_aggregation = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力引导的特征精炼
        self.refinement = nn.Sequential(
            ChannelAttention(out_channels),
            SpatialAttention(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        # 上采样
        x_up = self.upsample(x)
        
        # 跨尺度特征融合
        if skip is not None and hasattr(self, 'cross_fusion'):
            x_fused = self.cross_fusion(skip, x_up)
        else:
            x_fused = x_up
        
        # 特征金字塔处理
        pyramid_features = []
        current_feat = x_fused
        for block in self.pyramid_blocks:
            current_feat = block(current_feat)
            pyramid_features.append(current_feat)
        
        # 多尺度上下文聚合
        pyramid_combined = torch.cat(pyramid_features, dim=1)
        context_agg = self.context_aggregation(pyramid_combined)
        
        # 注意力引导的精炼
        refined = self.refinement(context_agg)
        
        return refined

# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试基础版本
    print("=" * 60)
    print("测试 MLSN_LSC_SAM2_VKAN_FRD 网络")
    print("=" * 60)
    
    # 创建模型
    model = MLSN_LSC_SAM2_VKAN_FRD(
        n_classes=1,
        in_channels=3,
        base_channels=64,
        img_size=256
    )
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # 前向传播
    outputs = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"最终输出形状: {outputs['final'].shape}")
    
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024**2:.2f} MB")
    
    # 各层输出形状
    print("\n各层编码器输出:")
    enc1 = model.encoder1(x)
    enc2 = model.encoder2(enc1)
    enc3 = model.encoder3(enc2)
    enc4 = model.encoder4(enc3)
    print(f"Encoder1 (SAM2): {enc1.shape}")
    print(f"Encoder2 (SAM2): {enc2.shape}")
    print(f"Encoder3 (VKAN): {enc3.shape}")
    print(f"Encoder4 (VKAN): {enc4.shape}")
    
    print("\n深度监督输出:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # 测试增强版本
    print("\n" + "=" * 60)
    print("测试增强版FRD解码器")
    print("=" * 60)
    
    enhanced_frd = EnhancedFRDBlock(
        in_channels=256,
        skip_channels=128,
        out_channels=128
    )
    
    test_input = torch.randn(2, 256, 16, 16)
    test_skip = torch.randn(2, 128, 32, 32)
    output = enhanced_frd(test_input, test_skip)
    print(f"输入: {test_input.shape}, 跳跃连接: {test_skip.shape}")
    print(f"增强FRD输出: {output.shape}")
    
    # 内存占用测试
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU内存测试")
        print("=" * 60)
        
        torch.cuda.empty_cache()
        model = model.cuda()
        x = x.cuda()
        
        with torch.no_grad():
            outputs = model(x)
        
        print(f"GPU内存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
