# zeroshot/config.py
# DINOv2 (timm) defaults
DEFAULT_MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"  # DINOv2 ViT-B/14
DEFAULT_PRETRAINED = "timm"  # not used by our code, kept for symmetry

# Batching (safe for RTX 4060)
DEFAULT_TEXT_BATCH = 32   # unused now (text), kept for compatibility
DEFAULT_IMG_BATCH  = 8

# How many source images per class to build prototypes (keep modest for speed/VRAM)
DEFAULT_MAX_PER_CLASS = 20

# Not used anymore but left for CLI compatibility
DEFAULT_TEMPLATES = 1
