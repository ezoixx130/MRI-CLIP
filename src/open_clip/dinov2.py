import os
import sys
import timm
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torchvision import transforms


def make_dinov2(dinov2_config, ckpt_path=''):
    ### build the dinov2 model ###
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(this_file_dir))
    from dinov2_modules.eval.setup import setup_and_build_model, get_args_parser
    dinov2_args = get_args_parser()
    # setup the config file
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = dinov2_config
    dinov2_args.opts = []
    dinov2_args.config_file = config_file
    dinov2_args.output_dir = os.path.dirname(ckpt_path)
    dinov2_args.pretrained_weights = ckpt_path
    dinov2_args.skip_distributed = True
    # load the dinov2 model
    model, autocast_dtype = setup_and_build_model(dinov2_args)

    return model, autocast_dtype
    

class DINOv2(nn.Module):
    def __init__(self,
                 config_path: str,
                 image_size: int=256,
                 pretrained: str=None,
                 embed_dim: int=768,
                 **kwargs):
        '''
        Implement the DINOv2 model with attention mechanism for ECHO classification task.

        Arguments:
        ----------
        config_path (str): path to the DINOv2 config file
        num_classes (int): number of classes
        pretrained (str): path to the pretrained model
        hidden_dim (int): hidden dimension
        '''
        
        super(DINOv2, self).__init__()
        self.image_size = image_size
        self.vit, _ = make_dinov2(config_path, ckpt_path=pretrained)
        self.transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "vit.cls_token",
            "vit.pos_embed",
            "vit.pos_embed_spatial",
            "vit.pos_embed_temporal",
            "vit.pos_embed_class",
        }
    
    def make_3ch(self, x):
        '''Make 3 channel input'''
        if x.shape[1] == 1:
            return torch.cat([x, x, x], dim=1)
        else:
            return x
    
    def forward(self, x):

        x, masks = x # x: [B, N, C, H, W], masks: [B, N]
        assert len(x.shape) == 5
        B, N, C, H, W = x.shape
        
        # assert N <= 40, "Each patient can have at most 40 images."

        # reshape input [B, N, C, H, W] -> [B * N, C, H, W]
        x = x.reshape(B * N, C, H, W)
        x = self.make_3ch(x)
        
        # normalize the input
        # if x.max() > 1:
            # x = x / 255.0
        
        # x = self.transform(x)
        
        # run through the backbone
        # print("😊", x[0][0])
        embedding_output = self.vit(x) # [B * N, hidden_dim]
        embedding_output = embedding_output.view(B, N, -1) # [B, N, hidden_dim]
        output = embedding_output * masks.unsqueeze(-1) # [B, N, hidden_dim]
        # print("😊",output.sum(dim=-1))
        output = output.sum(dim=1) # [B, hidden_dim]
        output = output / masks.sum(dim=1, keepdim=True) # masks.sum() is [B, 1] embedding_output: [B, hidden_dim]
        # print("😊",output)
        return output
        

def dinov2_large_classifier(**kwargs):
    model = DINOv2(config_path='open_clip/dinov2_modules/configs/train/vitl16_lbsz48_short.yaml', 
                               embed_dim=1024, 
                               **kwargs)
    return model
