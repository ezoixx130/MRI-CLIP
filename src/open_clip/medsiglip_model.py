import torch
from torch import nn
from huggingface_hub import login
from transformers import AutoProcessor, AutoModel
import numpy as np
from PIL import Image
from tensorflow.image import resize as tf_resize

def to_pil(images):
    for image in images:
        assert np.asarray(image).max() <= 255 and np.asarray(image).min() >= 0, f"Image {image} has max value {np.asarray(image).max()} which is greater than 255 or min value {np.asarray(image).min()} which is less than 0. Ensure images are normalized to [0, 255] range."
        break
    return [Image.fromarray(np.asarray(image).astype(np.uint8), 'RGB') for image in images]

def resize(images, image_size=(448, 448)):
    return tf_resize(images.permute(0, 2, 3, 1).cpu(), size=image_size, method='bilinear', antialias=False)

class MedSigLip(nn.Module):
    def __init__(self, medsiglip_id=None, embed_dim=1152, cast_dtype=None, image_size=(448, 448), output_dict=False, **kwargs):
        super(MedSigLip, self).__init__()
        self.model = AutoModel.from_pretrained(medsiglip_id)
        self.processor = AutoProcessor.from_pretrained(medsiglip_id)
        self.image_size = image_size
        self.output_dict = output_dict

    def forward(self, images, texts):
        images = resize(images * 225, self.image_size) # type(images) should be tensorflow.python.framework.ops.EagerTensor
        images = to_pil(images)
        inputs = self.processor(text=texts, images=images, padding="max_length", return_tensors="pt", truncation=True)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(next(self.model.parameters()).device)
        outputs = self.model(**inputs)
        if self.output_dict:
            return {
                "image_features": outputs.image_embeds,
                "text_features": outputs.text_embeds,
                "logit_scale": nn.Parameter((torch.Tensor([1.0])).to(outputs.image_embeds.device)),
            }
        else:
            return outputs.image_embeds, outputs.text_embeds, nn.Parameter((torch.Tensor([1.0])).to(outputs.image_embeds.device))