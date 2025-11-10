import torch
from torch import nn
from torch.nn.functional import interpolate
from transformers import OFATokenizer, OFAModel
from torchvision.transforms import Normalize


class BioMedGPT(nn.Module):
    def __init__(self, biomedgpt_path=None, embed_dim=768, cast_dtype=None, image_size=(480, 480), output_dict=False, **kwargs):
        super(BioMedGPT, self).__init__()
        self.model = OFAModel.from_pretrained(biomedgpt_path)
        self.tokenizer = OFATokenizer.from_pretrained(biomedgpt_path)
        # print(f"😊😊😊 Loading BioMedGPT model from {biomedgpt_path} 😊😊😊")
        self.image_size = image_size
        self.output_dict = output_dict
        self.normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def forward(self, images, texts):
        # print(f"Input images shape: {images.shape}, dtype: {images.dtype}")
        # print(f"Input text length: {len(texts)}")
        images = interpolate(images, size=self.image_size, mode='bilinear') # [B * N, C, H, W]
        images = self.normalize(images)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids.to(next(self.model.parameters()).device)

        empty_inputs = self.tokenizer([""] * len(images), return_tensors="pt", padding=True, truncation=True).input_ids.to(next(self.model.parameters()).device)
        
        image_features = self.model.get_encoder()(empty_inputs, patch_images=images).last_hidden_state[:, 0, :]
        text_features = self.model.get_encoder()(inputs).last_hidden_state[:, 0, :]
        # print(f"Image features shape: {image_features.shape}, Text features shape: {text_features.shape}")
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": nn.Parameter((torch.Tensor([1.0])).to(image_features.device)),
            }
        else:
            return image_features, text_features, nn.Parameter((torch.Tensor([1.0])).to(image_features.device))


if __name__ == "__main__":
    biomedgpt_path = "/home/hzhanguw/research-projects/BiomedGPT-Base-Pretrained"
    model = BioMedGPT(biomedgpt_path=biomedgpt_path)
    
    # # Example usage
    # images = torch.randn(2, 3, 480, 480)  # Batch of 2 images
    # texts = ["This is a sample text input for BioMedGPT.", "Another text input."]
    
    # output = model(images, texts)
    # print(output)