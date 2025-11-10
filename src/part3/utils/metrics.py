# utils/metrics.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict

@torch.no_grad()
def evaluate_embeddings(model, dataloader, device, k=5):
    """
    Evaluate embeddings using nearest-neighbor classification. 
    Calculates overall Top-1/Top-5 accuracy and prints detailed per-class Top-1 accuracy.
    The 'model' must have an 'embed(x)' method.
    """
    model.eval()

    embeddings = []
    labels = []

    for batch in tqdm(dataloader, desc="Embedding Extraction"):
        imgs = batch['image'].to(device)
        # Note: lbls are the integer class indices used for training/evaluation
        lbls = batch['label'].to(device) 

        # Call the embed method
        emb = model.embed(imgs) 
        emb = F.normalize(emb, dim=1) 

        embeddings.append(emb.cpu())
        labels.append(lbls.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0).numpy()
    num_samples = embeddings.size(0)

    # Compute cosine similarity matrix
    sim_matrix = embeddings @ embeddings.t()
    
    # Exclude self-similarity for nearest neighbor search
    sim_matrix.fill_diagonal_(-float('inf'))

    # --- Top-1 Prediction and Per-Class Accuracy ---
    top1_indices = sim_matrix.topk(1, dim=1).indices.squeeze(1)
    top1_labels = labels[top1_indices]
    is_correct = (top1_labels == labels)

    class_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for gt_label, correct in zip(labels, is_correct):
        class_results[gt_label]['total'] += 1
        if correct:
            class_results[gt_label]['correct'] += 1

    # --- Print Detailed Results ---
    print("\n🌳 Test Evaluation (Per-Class Top-1):")
    
    # Header
    print(f"{'Class Index':<15} | {'Acc':<10} | Samples")
    print("-" * 37)

    overall_correct = 0
    overall_total = 0

    # Print results, using integer index as the class ID
    for label in sorted(class_results.keys()):
        data = class_results[label]
        acc = 100.0 * data['correct'] / data['total']
        
        print(f"{label:<15} | Acc: {acc:>6.2f}% | Samples: {data['total']}")
        
        overall_correct += data['correct']
        overall_total += data['total']

    print("-" * 37)
    
    # --- Overall Accuracy Calculation ---
    top1 = 100.0 * overall_correct / overall_total
    
    # Calculate Top-5
    topk_indices = sim_matrix.topk(k, dim=1).indices
    topk_labels = labels[topk_indices]
    labels_expanded = labels[:, None]
    top5_correct = (topk_labels == labels_expanded).any(axis=1).sum()
    top5 = 100.0 * top5_correct / num_samples

    print(f"📝 Overall Results: Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
    return top1, top5