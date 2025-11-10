# Open-CLIP (brief)

Short README for the Open-CLIP fork extended with DINOv2 and biomedical models.

Purpose
- This repo integrates DINOv2 visual backbones and biomedical multimodal variants (BiomedCLIP, BiomedGPT, MedSiglip) to train and evaluate on MRI datasets (multi-slice inputs).

Quick start
- Create a Python environment and install dependencies (torch, timm, transformers, Pillow, nibabel, etc.).
- Inspect available entrypoint scripts:

```bash
bash scripts/run.sh
bash scripts/run_cuda.sh
bash eval.sh
bash eval_medsiglip.sh
bash eval_biomedgpt.sh
```


Notes
- For 3D MRI you can either store each slice as a separate item or implement a custom dataloader that stacks slices into multi-channel inputs.
- Do not share identifiable patient data; anonymize before sharing.

Contact
- For details about experiments, scripts, or environment used, see the full `open_clip/README.md` or ask for a pinned `requirements.txt` and example commands.
