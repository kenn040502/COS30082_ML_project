# utils/losses.py
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.1):
    """使用餘弦相似度的 Triplet Loss，通常更穩定"""
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    
    # 注意：這裡我們希望正樣本相似度更高，所以是 neg_sim - pos_sim
    losses = torch.relu(neg_sim - pos_sim + margin)
    return losses.mean()