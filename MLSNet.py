import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# ==================== FRD (Feature Recalibration Decoder) Block ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class MambaResidualBlock(nn.Module):
    """Mamba残差块"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = dim * expand
        
        # 前向和后向投影
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        
        # 1D卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.inner_dim,
            bias=False
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.inner_dim, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dim, self.inner_dim, bias=True)
        
        # 状态参数A
        self.A_log = nn.Parameter(torch.randn(self.inner_dim, d_state))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        
        # 输出投影
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
        
        # 激活函数
        self.act = nn.SiLU()
        
    def selective_scan(self, x, dt, A, B, C):
        """选择性扫描算法"""
        batch, seq_len, dim = x.shape
        d_state = A.shape[-1]
        
        # 离散化参数
        A = -torch.exp(A.float())
        dt = F.softplus(dt.float())
        
        # 离散化A和B
        dA = torch.exp(einsum(dt, A, 'b n d, d s -> b n d s'))
        dB = einsum(dt, B, 'b n d, b n s -> b n d s')
        
        # 初始化状态
        states = torch.zeros(batch, dim, d_state, device=x.device)
        outputs = []
        
        # 序列扫描
        for i in range(seq_len):
            states = states * dA[:, i] + dB[:, i].unsqueeze(2)
            output = einsum(states, C[:, i], 'b d s, b s -> b d')
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def forward(self, x):
        """
        输入形状: (batch, seq_len, dim)
        输出形状: (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        
        # 前向投影
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        
        # 1D卷积
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = self.act(x)
        x = rearrange(x, 'b d l -> b l d')
        
        # 准备SSM参数
        dt = self.dt_proj(x)
        B_C = self.x_proj(x)
        B, C = B_C.chunk(2, dim=-1)
        
        # 选择性扫描
        y = self.selective_scan(x, dt, self.A_log, B, C)
        y = y + self.D * x
        
        # 门控和输出
        y = y * self.act(z)
        y = self.out_proj(y)
        
        return y + residual


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, dim, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        输入形状: (batch, dim, seq_len)
        输出形状: (batch, dim, seq_len)
        """
        # 全局平均和最大池化
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        
        # 注意力权重
        attention = self.sigmoid(avg_out + max_out)
        
        # 应用注意力
        return x * attention.unsqueeze(-1)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(
            2, 1, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        输入形状: (batch, dim, seq_len)
        输出形状: (batch, dim, seq_len)
        """
        # 通道维度的平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接和卷积
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        # 应用注意力
        return x * attention


class MambaFeatureRecalibration(nn.Module):
    """基于Mamba的特征重校准模块"""
    def __init__(
        self, 
        dim, 
        d_state=16,
        d_conv=4,
        expand=2,
        use_channel_attention=True,
        use_spatial_attention=True,
        reduction_ratio=16
    ):
        super().__init__()
        self.dim = dim
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        
        # Mamba块
        self.mamba_block = MambaResidualBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 通道注意力
        if use_channel_attention:
            self.channel_attention = ChannelAttention(
                dim=dim,
                reduction_ratio=reduction_ratio
            )
        
        # 空间注意力
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # 自适应权重学习
        self.weight_gamma = nn.Parameter(torch.zeros(1))
        self.weight_beta = nn.Parameter(torch.zeros(1))
        
        # 最终归一化
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, return_attention=False):
        """
        输入形状: (batch, seq_len, dim) 或 (batch, dim, seq_len)
        输出形状: (batch, seq_len, dim)
        """
        original_shape = x.shape
        
        # 确保输入形状为 (batch, seq_len, dim)
        if len(original_shape) == 3 and original_shape[1] == self.dim:
            # 如果是 (batch, dim, seq_len)，转置为 (batch, seq_len, dim)
            x = rearrange(x, 'b d l -> b l d')
        
        # 1. Mamba特征变换
        mamba_out = self.mamba_block(x)
        
        # 转置以应用注意力
        if self.use_channel_attention or self.use_spatial_attention:
            x_transposed = rearrange(mamba_out, 'b l d -> b d l')
            
            # 2. 通道注意力
            if self.use_channel_attention:
                ca_out = self.channel_attention(x_transposed)
            else:
                ca_out = x_transposed
            
            # 3. 空间注意力
            if self.use_spatial_attention:
                sa_out = self.spatial_attention(ca_out)
            else:
                sa_out = ca_out
            
            # 转置回来
            attention_out = rearrange(sa_out, 'b d l -> b l d')
        else:
            attention_out = mamba_out
        
        # 4. 自适应融合
        recalibrated = mamba_out + self.weight_gamma * attention_out
        
        # 残差连接
        output = x + self.weight_beta * recalibrated
        
        # 归一化
        output = self.norm(output)
        
        # 如果需要恢复原始形状
        if len(original_shape) == 3 and original_shape[1] == self.dim:
            output = rearrange(output, 'b l d -> b d l')
        
        if return_attention:
            attention_weights = {
                'mamba_output': mamba_out,
                'attention_output': attention_out if self.use_channel_attention or self.use_spatial_attention else None
            }
            return output, attention_weights
        
        return output


class MultiScaleMambaRecalibration(nn.Module):
    """多尺度Mamba重校准模块"""
    def __init__(
        self, 
        dim, 
        num_scales=3,
        d_state=16,
        reduction_ratio=16
    ):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # 多尺度卷积
        self.conv_scales = nn.ModuleList([
            nn.Conv1d(
                dim, dim, 
                kernel_size=3 + 2*i, 
                padding=1 + i,
                groups=dim
            )
            for i in range(num_scales)
        ])
        
        # 多尺度Mamba模块
        self.mamba_scales = nn.ModuleList([
            MambaFeatureRecalibration(
                dim=dim,
                d_state=d_state,
                use_channel_attention=True,
                use_spatial_attention=False,
                reduction_ratio=reduction_ratio
            )
            for _ in range(num_scales)
        ])
        
        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        输入形状: (batch, seq_len, dim)
        输出形状: (batch, seq_len, dim)
        """
        # 转置以进行卷积
        x_transposed = rearrange(x, 'b l d -> b d l')
        
        scale_outputs = []
        for conv, mamba in zip(self.conv_scales, self.mamba_scales):
            # 多尺度卷积
            conv_out = conv(x_transposed)
            conv_out = rearrange(conv_out, 'b d l -> b l d')
            
            # Mamba重校准
            mamba_out = mamba(conv_out)
            scale_outputs.append(mamba_out)
        
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, scale_outputs))
        
        # 残差连接和输出
        output = x + self.out_proj(fused)
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

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
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

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)
        return self.sigmoid(attention)

class CBAM(nn.Module):
    """结合通道和空间注意力的CBAM模块"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x
class LearnableSkipConnection(nn.Module):
    """
    可学习跳跃连接模块
    结合通道注意力、空间注意力和可学习权重
    """
    def __init__(self, encoder_channels, decoder_channels, use_cbam=True):
        super(LearnableSkipConnection, self).__init__()
        
        # 对齐通道数
        self.conv_align = nn.Conv2d(encoder_channels, decoder_channels, 1)
        self.bn = nn.BatchNorm2d(decoder_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.use_cbam = use_cbam
        if use_cbam:
            self.attention = CBAM(decoder_channels)
        
        # 可学习权重参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 编码器特征权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # 解码器特征权重
        
    def forward(self, encoder_feat, decoder_feat):
        """
        Args:
            encoder_feat: 编码器特征 [B, C_enc, H, W]
            decoder_feat: 解码器特征 [B, C_dec, H, W]
        Returns:
            enhanced_feat: 增强后的特征 [B, C_dec, H, W]
        """
        # 1. 对齐编码器特征的通道数
        aligned_encoder = self.relu(self.bn(self.conv_align(encoder_feat)))
        
        # 2. 应用注意力机制
        if self.use_cbam:
            aligned_encoder = self.attention(aligned_encoder)
        
        # 3. 调整大小（如果空间维度不一致）
        if aligned_encoder.shape[-2:] != decoder_feat.shape[-2:]:
            aligned_encoder = F.interpolate(
                aligned_encoder, 
                size=decoder_feat.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # 4. 可学习加权融合（使用sigmoid确保权重在0-1之间）
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        # 归一化权重
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total
        
        # 融合特征
        enhanced_feat = alpha * aligned_encoder + beta * decoder_feat
        
        return enhanced_feat      

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
