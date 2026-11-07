import torch 
from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch.nn.functional as F



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
        # 1. Upscale input to SAM's expected 1024x1024 resolution
        # The image encoder handles batched inputs efficiently
        x_up = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        features = self.image_encoder(x_up)  # (B, 256, 64, 64)
        
        B = x.shape[0]
        outs = []
        
        # 2. Loop over the batch dimension for the mask decoder.
        # This bypasses SAM's internal repeat_interleave bug that expands 
        # batched image_embeddings to B*B when tokens.shape[0] == B.
        for i in range(B):
            feat_i = features[i:i+1]  # (1, 256, 64, 64)
            
            # Empty prompts for a single image (batch size 1)
            sparse = torch.zeros(1, 0, 256, device=x.device, dtype=feat_i.dtype)
            dense = torch.zeros(1, 256, 64, 64, device=x.device, dtype=feat_i.dtype)

            low_res, _ = self.mask_decoder(
                image_embeddings=feat_i,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            outs.append(low_res)

        # 3. Concatenate results and downscale back to original resolution
        logits = torch.cat(outs, dim=0)  # (B, 1, 256, 256)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_sam_lora():
    return SAMLoRABinary()







# class SAMLoRABinary(nn.Module):
#     def __init__(self, checkpoint="checkpoints/sam_vit_b_01ec64.pth"):
#         super().__init__()
#         sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
#         encoder = sam.image_encoder

#         # Freeze everything in the encoder
#         for p in encoder.parameters():
#             p.requires_grad = False

#         # LoRA on fused qkv (SAM has no separate q_proj/v_proj)
#         lora_cfg = LoraConfig(
#             r=8, lora_alpha=16,
#             target_modules=["qkv"],   
#             lora_dropout=0.0,
#             bias="none",
#         )
#         self.image_encoder = get_peft_model(encoder, lora_cfg)
#         self.prompt_encoder = sam.prompt_encoder 

#         for p in self.prompt_encoder.parameters():
#             p.requires_grad = False
        
#         self.mask_decoder = sam.mask_decoder

        
#         # Decoder parameters are trainable
#         for p in self.mask_decoder.parameters():
#             p.requires_grad = True

    
#     def forward(self, x):
#         # 1. Upscale input to SAM's expected 1024x1024 resolution
#         x_up = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        
#         # 2. Extract image embeddings
#         features = self.image_encoder(x_up)
#         B = x.shape[0]
        
#         # 3. Create empty prompt embeddings for the entire batch
#         # sparse: (B, 0, 256) -> No point/box prompts
#         # dense:  (B, 256, 64, 64) -> No mask prompts (zeros)
#         sparse = torch.zeros(B, 0, 256, device=x.device, dtype=x.dtype)
#         dense = torch.zeros(B, 256, 64, 64, device=x.device, dtype=x.dtype)

#         # 4. Decode masks
#         low_res, _ = self.mask_decoder(
#             image_embeddings=features,
#             image_pe=self.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse,
#             dense_prompt_embeddings=dense,
#             multimask_output=False, # Outputs exactly 1 mask per image
#         )
        
#         # 5. Downscale back to original resolution
#         return F.interpolate(low_res, size=x.shape[-2:], mode="bilinear", align_corners=False)
    
    
    


# def build_sam_lora():
#     return SAMLoRABinary()