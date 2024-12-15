import torch.nn as nn
import torch

from models.rdm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param mid_channels: the number of middle channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    """

    def __init__(
        self,
        channels,
        mid_channels,
        emb_channels,
        dropout,
        use_context=False,
        context_channels=512
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, mid_channels, bias=True),
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Linear(mid_channels, channels, bias=True)
            ),
        )

        self.use_context = use_context
        if use_context:
            self.context_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_channels, mid_channels, bias=True),
        )

    def forward(self, x, emb, context):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        if self.use_context:
            context_out = self.context_layers(context)
            if not context_out.shape == h.shape == emb_out.shape:
                assert context_out.flatten().shape == h.flatten().shape == emb_out.flatten().shape
                assert context_out.shape[0] == 1 and len(context_out.shape) == 2
                h = h.reshape(context_out.shape)
            assert context_out.shape == h.shape == emb_out.shape
            h = h + emb_out + context_out
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return x + h


class SimpleMLP(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        time_embed_dim,
        model_channels,
        bottleneck_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        use_context=False,
        context_channels=512
    ):
        super().__init__()

        self.image_size = 1
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                bottleneck_channels,
                time_embed_dim,
                dropout,
                use_context=use_context,
                context_channels=context_channels
            ))

        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(model_channels, eps=1e-6),
            nn.SiLU(),
            zero_module(nn.Linear(model_channels, out_channels, bias=True)),
        )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = x.squeeze()
        # context = context.squeeze() # NOTE: We do not use conditioning (temporarily).
        x = self.input_proj(x)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        for block in self.res_blocks:
            x = block(x, emb, context)

        return self.out(x).unsqueeze(-1).unsqueeze(-1)




class SimpleEncoder(nn.Module):
    
    def __init__(
        self,
        dim_sequence,
        use_layer_norm=True
    ):
        super().__init__()
        
        res_block_num = len(dim_sequence) - 1
        
        self.out_dim = dim_sequence[-1]
        self.blocks = nn.ModuleList()
        for i in range(res_block_num):
            in_c = dim_sequence[i]
            out_c = dim_sequence[i + 1]
            if i == res_block_num - 1:
                out_c = out_c * 2
            layer = nn.Sequential(
                nn.LayerNorm(in_c) if use_layer_norm else nn.Identity(),
                nn.SiLU(),
                nn.Linear(in_c, out_c, bias=True),
            )
            self.blocks.append(layer)
        self.out_dim = dim_sequence[-1]

        
        
    def forward(
        self,
        x
    ):
        for block in self.blocks:
            x_update = block(x)
            if x_update.shape[-1] == x.shape[-1]:
                x_update = x_update + x
            x = x_update
        
        x_mean = x[:, :self.out_dim]
        x_log_std = x[:, self.out_dim:]
        assert x_mean.shape == x_log_std.shape
        x_std = torch.exp(0.5 * x_log_std)
        
        
        return x_mean, x_std
    
class SimpleDecoder(nn.Module):
    
    def __init__(
        self,
        dim_sequence,
        use_layer_norm=True
    ):
        super().__init__()
        
        res_block_num = len(dim_sequence) - 1
        
        self.blocks = nn.ModuleList()
        for i in range(res_block_num):
            in_c = dim_sequence[i]
            out_c = dim_sequence[i + 1]
            layer = nn.Sequential(
            nn.LayerNorm(in_c) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(in_c, out_c, bias=True),
        )
            self.blocks.append(layer)
    def forward(
        self,
        x
    ):
        for block in self.blocks:
            x_update = block(x)
            if x_update.shape[-1] == x.shape[-1]:
                x_update = x_update + x
            x = x_update

        return x