from models.rdm.util import instantiate_from_config
import os
from models.util.misc import NativeScalerWithGradNormCount as NativeScaler
import models.util.misc as misc
import torch
from models.encoders import initialize_encoder


def initialize_RDM(rdm_args, model_args, device):
    rank = rdm_args.rank
    # Set up for RDM encoder
    assert rdm_args.encoder_path is None or rdm_args.encoder_path == "" or os.path.exists(rdm_args.encoder_path), 'Encoder path does not exist.'
    
    
    encoder = initialize_encoder(encoder_type=rdm_args.encoder_type, device=device, encoder_ckpt_path=rdm_args.encoder_path)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    encoder = encoder.to(device)

    
    # Set up for RDM_model
    RDM_model = instantiate_from_config(model_args)
    RDM_model.pretrained_encoder = encoder
    
    from models.rdm.models.diffusion.ddpm import LatentRDM
    if type(RDM_model) is LatentRDM:
        if rdm_args.train_diffusion:
            assert not (rdm_args.vae_ckpt is None and not rdm_args.trainable_ae), "You are not loading vae, and not training it. It is random."
            # Load trained vae parameters
            if rdm_args.vae_ckpt is not None:
                ckpt = torch.load(rdm_args.vae_ckpt, map_location=device)
                missing_keys, unexpected_keys = RDM_model.load_state_dict(ckpt["model"], strict=False) # This will also load other parameters, but it is fine since other parameters are random initialized in the first stage and not trained.
                assert len(missing_keys) == 0
                assert len([i for i in unexpected_keys if "ema" not in i]) == 0
                
            # Freeze vae model if not allowing training
            if not rdm_args.trainable_ae:
                for param in RDM_model.vae.parameters():
                    param.requires_grad = False
        else:
            for name, param in RDM_model.named_parameters(): # NOTE: We are not using any conditioning in VAE, therefore no gradient in conditioning stage model here.
                if "vae" not in name:
                    param.requires_grad = False
                else:
                    pass
    
    
    RDM_model.to(device)
    RDM_model_without_ddp = RDM_model
    if rdm_args.dp:
        RDM_model = torch.nn.parallel.DistributedDataParallel(RDM_model, device_ids=[rank], find_unused_parameters=True)
        RDM_model_without_ddp = RDM_model.module
    
    # Set up for optimizer.
    params = RDM_model.parameters()
    
    optimizer = torch.optim.AdamW(params, lr=rdm_args.lr, weight_decay=rdm_args.weight_decay)
    
    loss_scaler = NativeScaler()
    
    if rdm_args.rdm_ckpt is not None:
        misc.load_model(args=rdm_args, model_without_ddp=RDM_model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    return RDM_model, RDM_model_without_ddp, loss_scaler, optimizer