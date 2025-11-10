# models/triplet_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    # Use 128 as a standard embedding dimension for metric learning
    def __init__(self, feature_extractor, feat_dim=768, embedding_dim=128): 
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        self.embedding_dim = embedding_dim

    def forward(self, anchor, positive, negative):
        # Extract all features
        anchor_emb_raw = self.fc(self.feature_extractor(anchor))
        positive_emb_raw = self.fc(self.feature_extractor(positive))
        negative_emb_raw = self.fc(self.feature_extractor(negative))
        
        # L2 normalization
        anchor_emb = F.normalize(anchor_emb_raw, dim=-1)
        positive_emb = F.normalize(positive_emb_raw, dim=-1)
        negative_emb = F.normalize(negative_emb_raw, dim=-1)
        
        return anchor_emb, positive_emb, negative_emb

    def embed(self, x):
        # Ensure consistency in evaluation (L2 normalization)
        with torch.no_grad():
            emb = self.fc(self.feature_extractor(x))
            emb = F.normalize(emb, dim=-1)
        return emb


# Improved triplet loss function with hard negative mining support
def triplet_loss(anchor, positive, negative, margin=0.3, hard_mining=True):
    batch_size = anchor.size(0)
    
    # Calculate positive distances
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    
    if hard_mining:
        # Calculate all anchor-negative pairwise distances
        # Using cdist for efficient batch distance computation
        all_neg_dists = torch.cdist(anchor, negative, p=2)
        
        # Find hardest negative for each anchor (closest wrong sample)
        # Add large value to diagonal to avoid self-comparison
        neg_dist, _ = torch.min(all_neg_dists + torch.eye(batch_size).to(anchor.device) * 1e5, dim=1)
    else:
        # Original random negative samples
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
    
    # Standard triplet loss formula
    losses = torch.relu(pos_dist - neg_dist + margin)
    return losses.mean()


def cosine_triplet_loss(anchor, positive, negative, margin=0.1):
    """Triplet Loss using cosine similarity - often more stable"""
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    
    # Note: We want positive similarity to be higher, so it's neg_sim - pos_sim
    losses = torch.relu(neg_sim - pos_sim + margin)
    return losses.mean()