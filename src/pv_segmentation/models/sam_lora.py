import torch 
from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch.nn.functional as F

class SAMLoRABinary(nn.Module):
    def __init__(self, checkpoint="checkpoints/sam_vit_b_01ec64.pth"):
        super().__init__()
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        encoder = sam.image_encoder

        # Freeze everything in the encoder
        for p in encoder.parameters():
            p.requires_grad = False

        # LoRA on fused qkv (SAM has no separate q_proj/v_proj)
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["qkv"],   
            lora_dropout=0.0,
            bias="none",
        )
        self.image_encoder = get_peft_model(encoder, lora_cfg)
        self.prompt_encoder = sam.prompt_encoder 

        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        
        self.mask_decoder = sam.mask_decoder

        
        # Decoder parameters are trainable
        for p in self.mask_decoder.parameters():
            p.requires_grad = True

    def forward(self, x):
        x_up = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        features = self.image_encoder(x_up)
        B = x.shape[0]
        sparse = torch.zeros(B, 0, 256, device=x.device)
        dense = torch.zeros(B, 256, 64, 64, device=x.device)
        low_res, _ = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return F.interpolate(low_res, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_sam_lora():
    return SAMLoRABinary()