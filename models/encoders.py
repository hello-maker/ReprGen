import sys
sys.path.append(".")

from models.torchmdnet.models.model import load_model

import torch
import numpy as np
import omegaconf
import os
from omegaconf import OmegaConf



def initialize_encoder(encoder_type, device, encoder_ckpt_path):
    assert os.path.exists(encoder_ckpt_path), f"Please ensure the existence of {encoder_ckpt_path}"
    if encoder_type == "frad":
        encoder_args_path = "./hydra_configs/ET-QM9-FT_dw_0.2_long.yaml"
        encoder_args = OmegaConf.load(encoder_args_path).encoder_args
        
        encoder = load_model(filepath=encoder_ckpt_path, device=device, args=encoder_args, only_representation=True)
        encoder = encoder.to(device)

    elif encoder_type == "unimol":
        try:
            import unicore
        except ImportError as e:
            raise ImportError("The 'unicore' package is not installed. Please install it in https://github.com/dptech-corp/Uni-Core for the use of unimol encoder.")
        from unicore import (
            tasks,
            utils,
        )
        encoder_args_path = "./hydra_configs/unimol_encoder.yaml"
        unimol_args = OmegaConf.load(encoder_args_path).unimol_args
        unimol_args.finetune_mol_model = encoder_ckpt_path
        sys.path.append("models/unimol")
        
        unimol_args = omegaconf.OmegaConf.create(unimol_args)
        utils.import_user_module(unimol_args)
        utils.set_jit_fusion_options()

        task = tasks.setup_task(unimol_args)

        # Build model and loss
        encoder = task.build_model(unimol_args)
        encoder = encoder.to(device)
    else:
        raise ValueError(f"No encoder type of {encoder_type}")
    return encoder
        


import torch
from torch_scatter import scatter
import numpy as np
from collections import OrderedDict
def flatten_x_h(x, h, node_mask):
    '''
        Input structured dense x, h and output flattened sparse x, h.
    '''
    assert len(x.shape) == 3
    batch_size, max_node_num = x.shape[0], x.shape[1]
    device = x.device
    
    
    assert node_mask.shape == (batch_size, max_node_num) or node_mask.shape == (batch_size, max_node_num, 1)
    
    
    assert x.shape == (batch_size, max_node_num, 3) and h.shape == (batch_size, max_node_num, 1)
    
    node_mask_flatten = node_mask.flatten() # (batch_size * max_node_num)
    
    node_mask_flatten = node_mask_flatten.to(torch.bool)

    batch = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, max_node_num).flatten() # (batch_size * max_node_num)
    batch = batch[node_mask_flatten]
    
    # x shape (batch_size, max_node_num, 3)
    x_flatten = x.flatten(0, 1) # (batch_size * max_node_num, 3)
    x_flatten = x_flatten[node_mask_flatten] # (flattened_node_num, 3)
    
    # h shape (batch_size, max_node_num, 1)
    h_integer_flatten = h.flatten(0, 2) # (batch_size * max_node_num)
    h_integer_flatten = h_integer_flatten[node_mask_flatten] # (flattened_node_num)
    
    assert len(batch.shape) == 1 and len(x_flatten.shape) == 2 and len(h_integer_flatten.shape) == 1

    h_integer_flatten = h_integer_flatten.to(torch.long)
    assert h_integer_flatten.unique()[0] != 0
    
    return batch, x_flatten, h_integer_flatten


def get_global_representation(node_mask, rep_encoder, x, h, training_encoder=False, noise_sigma=0., device=None):
    # assert atom_decoder is not None
    if not training_encoder:
        rep_encoder.eval()
    else:
        assert rep_encoder.training # We set training in loop codes
        
    if device is not None:
        rep_encoder.to(device)
    
    batch_size, max_node_num = x.shape[0], x.shape[1]
    if "UniMol" in rep_encoder.__class__.__name__:
        input_dict = format_input_to_unimol(x, h, node_mask)
        node_context, pair_context = rep_encoder(**input_dict, features_only=True)
        # Flatten node_context
        node_mask_bool = node_mask.to(torch.bool).reshape(batch_size, max_node_num) # (b, max_n)
        # remove pretend
        # node_context = [node_context_piece[:, 1:, :] for i, node_context_piece in enumerate(node_context)]
        node_context = node_context[:, 1:-1, :]
        rep = [torch.sum(node_context_piece[node_mask_bool[i]], dim=0, keepdim=True) for i, node_context_piece in enumerate(node_context)]
        rep = torch.cat(rep, dim=0)
    else:
        
        if type(h) is dict:
            h_integer_feature = h["integer"] if len(h["integer"]) > 0 else h["rep_integer"] # NOTE: CAREFUL WITH THIS. When not include_charges, h["integer"] will be empty. But we ensure integer through rep_integer.
            batch, x_flatten, h_integer_flatten = flatten_x_h(x, h_integer_feature, node_mask)
        else:
            batch, x_flatten, h_integer_flatten = flatten_x_h(x, h, node_mask)
        
        assert h_integer_flatten.min() > 0
        node_context, vec, z, pos, batch = rep_encoder(h_integer_flatten, x_flatten, batch)
    
        assert len(node_context.shape) == 2 # (flattened_node_num, context_dim)
        rep = scatter(node_context, batch, dim=0, reduce="sum") # NOTE: We use sum here, but we can use other aggregation methods.
    assert rep.shape == (batch_size, node_context.shape[-1])
        
    rep_std = torch.std(rep, dim=1, keepdim=True)
    rep_mean = torch.mean(rep, dim=1, keepdim=True) # NOTE: Layer Norm.
    rep = (rep - rep_mean) / rep_std
    
    if noise_sigma > 0.:
        noise = torch.randn(rep.shape, device=rep.device) * noise_sigma
        rep = rep + noise
        

    return rep


def format_input_to_unimol(x, h, node_mask):
    
    
    def collate_tokens(
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
        return res

    def collate_tokens_coords(
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        res = values[0].new(len(values), size, 3).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
        return res

    def collate_tokens_2d(
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        res = values[0].new(len(values), size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):, size - len(v):] if left_pad else res[i][:len(v), :len(v)])
        return res
        
    
    if type(h) is dict:
        h = h["integer"] if len(h["integer"]) > 0 else h["rep_integer"] # NOTE: CAREFUL WITH THIS. When not include_charges != False, h["integer"] can be empty. But we ensure integer through rep_integer.
    
    # obtain data list
    node_mask = node_mask.to(torch.bool)
    atom_num = [h_piece[node_mask[i]].squeeze() for i, h_piece in enumerate(h)] # [(num_atom_i) for i]
    coord = [x_piece[node_mask[i].squeeze()] for i, x_piece in enumerate(x)] # [(num_atom_i, 3) for i]

    
    '''
        process atoms
    '''
    # tokenization
    atom_num2unimol_token_idx = {
        1: 8,
        5: 15,
        6: 4,
        7: 5,
        8: 6,
        9: 10,
        13: 18,
        14: 13,
        15: 14,
        16: 7,
        17: 9,
        33: 21,
        35: 11,
        53: 12,
        80: 22,
        83: 30 
    }
    atom_token_idx = [np.vectorize(lambda x: atom_num2unimol_token_idx[x])(atom_num_piece.cpu().numpy()) for atom_num_piece in atom_num]
    atom_token_idx = [torch.tensor(atom_num_piece, dtype=torch.long, device=coord[0].device) for atom_num_piece in atom_token_idx]
    # prepend and append
    atom_token_idx = [torch.cat([torch.tensor([1], dtype=torch.long, device=atom_token_idx_piece.device), atom_token_idx_piece], dim=0) for atom_token_idx_piece in atom_token_idx]
    atom_token_idx = [torch.cat([atom_token_idx_piece, torch.tensor([2], dtype=torch.long, device=atom_token_idx_piece.device)], dim=0) for atom_token_idx_piece in atom_token_idx]
    
    '''
        process coord
    '''
    # normalize
    coord = [coord_piece - torch.mean(coord_piece, dim=0, keepdim=True) for coord_piece in coord]
    
    # prepend and append
    coord = [torch.cat([torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=coord_piece.device).unsqueeze(0), coord_piece], dim=0) for coord_piece in coord]
    coord = [torch.cat([coord_piece, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=coord_piece.device).unsqueeze(0)], dim=0) for coord_piece in coord]
    
    
    '''
        process edge_type
    '''
    unimol_token_num = 32 # NOTE: Carefully check again
    edge_type = [atom_token_idx_piece.view(-1, 1) * unimol_token_num + atom_token_idx_piece.view(1, -1) for atom_token_idx_piece in atom_token_idx] # NOTE: Check the padded positions
    
    '''
        process distance matrix
    '''
    dist_matrix = [torch.norm((coord_piece.unsqueeze(0) - coord_piece.unsqueeze(1)), dim=-1, p=2) for coord_piece in coord]
    
    '''
        collate them.
    '''
    atom_token_idx = collate_tokens(atom_token_idx, pad_idx=0, left_pad=False, pad_to_multiple=1)
    coord = collate_tokens_coords(coord, pad_idx=0, left_pad=False, pad_to_multiple=1)
    dist_matrix = collate_tokens_2d(dist_matrix, pad_idx=0, left_pad=False, pad_to_multiple=1)
    edge_type = collate_tokens_2d(edge_type, pad_idx=0, left_pad=False, pad_to_multiple=1)
    
    
    input_dict = OrderedDict()
    input_dict["src_tokens"] = atom_token_idx
    input_dict["src_coord"] = coord
    input_dict["src_distance"] = dist_matrix
    input_dict["src_edge_type"] = edge_type
    
    return input_dict


    