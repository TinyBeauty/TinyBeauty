import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PosEnSine(nn.Module):

    def __init__(self, num_pos_feats):
        super(PosEnSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = True
        self.scale = 2 * math.pi
        self.temperature = 10000

    def forward(self, x):
        b, c, h, w = x.shape
        not_mask = torch.ones(1, h, w, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.repeat(b, 1, 1, 1)
        return pos


def softmax_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)  # b x n x hw x d
    k = k.flatten(-2)   # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)

    N = k.shape[-1]     # ?????? maybe change to k.shape[-2]????
    attn = torch.matmul(q / N ** 0.5, k)
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn

class OurMultiheadAttention(nn.Module):
    def __init__(self, feat_dim, n_head, d_k=None, d_v=None):
        super(OurMultiheadAttention, self).__init__()
        if d_k is None:
            d_k = feat_dim // n_head
        if d_v is None:
            d_v = feat_dim // n_head

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)

    def forward(self, q, k, v, attn_type='softmax', **kwargs):
        # input: b x d x h x w
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection: b x (nhead*dk) x h x w
        # Separate different heads: b x nhead x dk x h x w
        q = self.w_qs(q).view(q.shape[0], n_head, d_k, q.shape[2], q.shape[3])
        k = self.w_ks(k).view(k.shape[0], n_head, d_k, k.shape[2], k.shape[3])
        v = self.w_vs(v).view(v.shape[0], n_head, d_v, v.shape[2], v.shape[3])

        # -------------- Attention -----------------
        if attn_type == 'softmax':
            q, attn = softmax_attention(q, k, v)  # b x n x dk x h x w --> b x n x dv x h x w
        elif attn_type == 'dotproduct':
            q, attn = dotproduct_attention(q, k, v)
        elif attn_type == 'patch':
            q, attn = patch_attention(q, k, v, P=kwargs['P'])
        elif attn_type == 'sparse_long':
            q, attn = long_range_attention(q, k, v, P_h=kwargs['ah'], P_w=kwargs['aw'])
        elif attn_type == 'sparse_short':
            q, attn = short_range_attention(q, k, v, Q_h=kwargs['ah'], Q_w=kwargs['aw'])
        else:
            raise NotImplementedError(f'Unknown attention type {attn_type}')
        # ------------ end Attention ---------------

        # Concatenate all the heads together: b x (n*dv) x h x w
        q = q.reshape(q.shape[0], -1, q.shape[3], q.shape[4])
        q = self.fc(q)   # b x d x h x w

        return q, attn

class TransformerDecoderUnit(nn.Module):
    def __init__(self, feat_dim, n_head=8, pos_en_flag=True, attn_type='softmax', P=None):
        super(TransformerDecoderUnit, self).__init__()
        self.feat_dim = feat_dim
        self.attn_type = attn_type
        self.pos_en_flag = pos_en_flag
        self.P = P

        self.pos_en = PosEnSine(self.feat_dim // 2)
        self.attn = OurMultiheadAttention(feat_dim, n_head)   # cross-attention
        
        self.linear1 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.linear2 = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.activation = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm2d(self.feat_dim)

    def forward(self, q, k, v):
        if self.pos_en_flag:
            q_pos_embed = self.pos_en(q)
            k_pos_embed = self.pos_en(k)
        else:
            q_pos_embed = 0
            k_pos_embed = 0
        
        # cross-multi-head attention
        out = self.attn(q=q+q_pos_embed, k=k+k_pos_embed, v=v, attn_type=self.attn_type, P=self.P)[0]

        # feed forward
        out2 = self.linear2(self.activation(self.linear1(out)))
        out = out + out2
        out = self.norm(out)

        return out

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.conv_first = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        self.down1 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down2 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.style = nn.Conv2d(dim, style_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_first(x)
        style1 = self.down1(x)
        style2 = self.down2(style1)
        style = self.style(style2)
        return style

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()


        self.conv_first = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        self.down1 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down2 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.resblocks = ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)

    def forward(self, x):

        x = self.conv_first(x)
        content1 = self.down1(x)
        content2 = self.down2(content1)
        content3 = self.resblocks(content2)
        return content1, content2, content3

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)



# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, content):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return content * y.expand_as(content)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, content):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        
        return content * scale

class BeautyREC(nn.Module):
    def __init__(self, params):
        super(BeautyREC, self).__init__()
        
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        self.content_encoder = ContentEncoder(n_downsample, n_res, 3, dim, 'in', activ, pad_type=pad_type)

        # self.attention1 = TransformerDecoderUnit(feat_dim = dim)
        # reconstruc decoder
        self.res = ResBlocks(n_res, dim, norm='in', activation=activ, pad_type=pad_type)
        self.up = nn.Upsample(scale_factor=2)
        self.up1 = nn.Upsample(scale_factor=2)
        self.upconv1 = Conv2dBlock(dim*2, dim, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)
        self.up2 = nn.Upsample(scale_factor=2)
        self.upconv2 = Conv2dBlock(dim*2, dim, 3, 1, 1, norm='ln', activation=activ, pad_type=pad_type)
        self.out = Conv2dBlock(dim, 3, 3, 1, 1, norm='none', activation='tanh', pad_type=pad_type)

    def forward(self, x):
        content1, content2, content3= self.content_encoder(x)

        recon3 = self.res(content3)
        recon23 = torch.cat([content2, recon3], dim=1)
        recon23 = self.upconv1(self.up1(recon23))
        recon12 = torch.cat([ content1, recon23], dim=1)
        recon12 = self.upconv2(self.up2(recon12))
        images_recon = self.out(recon12)
        
        return images_recon

    
    def infercontent(self, x):
        content, _, _ = self.content_encoder(x)
        return content
    
    def inferstyle(self,x,xmask):
        for i in range(0, xmask.size(1)):
            xi = xmask[:, i, :, :]
            xi = torch.unsqueeze(xi, 1).repeat(1, x.size(1), 1, 1)
            xi = x.mul(xi)
            if i==0:
                style = xi
            else:
                style = torch.cat([style, xi], dim=1)
        return style

