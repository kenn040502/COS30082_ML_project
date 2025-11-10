# utils/metrics.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from utils.class_utils import get_class_type_mapping, load_class_lists

@torch.no_grad()
def evaluate_embeddings_detailed(model, dataloader, device, data_root, global_class_to_idx, k=5):
    """
    Enhanced evaluation with separate metrics for classes with/without pairs
    """
    model.eval()

    # Load class type information
    classes_with_pairs, classes_without_pairs = load_class_lists(data_root)
    class_type_mapping = get_class_type_mapping(global_class_to_idx, classes_with_pairs, classes_without_pairs)

    embeddings = []
    labels = []

    for batch in tqdm(dataloader, desc="Embedding Extraction"):
        imgs = batch['image'].to(device)
        lbls = batch['label'].to(device)

        emb = model.embed(imgs)
        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu())
        labels.append(lbls.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0).numpy()
    num_samples = embeddings.size(0)

    # Compute similarity matrix
    sim_matrix = embeddings @ embeddings.t()
    sim_matrix.fill_diagonal_(-float('inf'))

    # Get predictions
    topk_indices = sim_matrix.topk(k, dim=1).indices
    topk_labels = labels[topk_indices]
    
    labels_expanded = labels[:, None]

    # Initialize results storage
    results = {
        'overall': {'correct_top1': 0, 'correct_top5': 0, 'total': 0},
        'with_pairs': {'correct_top1': 0, 'correct_top5': 0, 'total': 0},
        'without_pairs': {'correct_top1': 0, 'correct_top5': 0, 'total': 0}
    }
    
    class_results = defaultdict(lambda: {'correct_top1': 0, 'correct_top5': 0, 'total': 0})

    # Calculate accuracy for each sample
    for i in range(num_samples):
        gt_label = labels[i]
        class_type = class_type_mapping.get(gt_label, 'unknown')
        
        # Top-1 accuracy
        top1_correct = (topk_labels[i, 0] == gt_label)
        # Top-5 accuracy  
        top5_correct = (topk_labels[i] == gt_label).any()
        
        # Update overall results
        results['overall']['total'] += 1
        if top1_correct:
            results['overall']['correct_top1'] += 1
        if top5_correct:
            results['overall']['correct_top5'] += 1
        
        # Update class type results
        if class_type in ['with_pairs', 'without_pairs']:
            results[class_type]['total'] += 1
            if top1_correct:
                results[class_type]['correct_top1'] += 1
            if top5_correct:
                results[class_type]['correct_top5'] += 1
        
        # Update per-class results
        class_results[gt_label]['total'] += 1
        if top1_correct:
            class_results[gt_label]['correct_top1'] += 1
        if top5_correct:
            class_results[gt_label]['correct_top5'] += 1

    # Calculate final metrics
    metrics = {}
    
    for category in ['overall', 'with_pairs', 'without_pairs']:
        if results[category]['total'] > 0:
            metrics[f'{category}_top1'] = 100.0 * results[category]['correct_top1'] / results[category]['total']
            metrics[f'{category}_top5'] = 100.0 * results[category]['correct_top5'] / results[category]['total']
        else:
            metrics[f'{category}_top1'] = 0.0
            metrics[f'{category}_top5'] = 0.0

    # Print detailed results
    print("\n🌳 Detailed Test Evaluation:")
    print("=" * 50)
    print(f"{'Category':<15} | {'Top-1':<8} | {'Top-5':<8} | {'Samples':<8}")
    print("-" * 50)
    
    for category in ['overall', 'with_pairs', 'without_pairs']:
        if results[category]['total'] > 0:
            print(f"{category:<15} | {metrics[f'{category}_top1']:>6.2f}% | {metrics[f'{category}_top5']:>6.2f}% | {results[category]['total']:>8}")
    
    print("=" * 50)
    
    # Print per-class results (optional, for debugging)
    print("\n📊 Per-Class Top-1 Accuracy:")
    print(f"{'Class':<15} | {'Type':<12} | {'Acc':<8} | Samples")
    print("-" * 50)
    
    for class_idx in sorted(class_results.keys()):
        data = class_results[class_idx]
        if data['total'] > 0:
            acc = 100.0 * data['correct_top1'] / data['total']
            class_type = class_type_mapping.get(class_idx, 'unknown')
            print(f"{class_idx:<15} | {class_type:<12} | {acc:>6.2f}% | {data['total']:>7}")
    
    print("=" * 50)

    return metrics

# Keep the original function for backward compatibility
@torch.no_grad()
def evaluate_embeddings(model, dataloader, device, k=5):
    """
    Original evaluation function (simplified version)
    """
    model.eval()
    embeddings = []
    labels = []

    for batch in tqdm(dataloader, desc="Embedding Extraction"):
        imgs = batch['image'].to(device)
        lbls = batch['label'].to(device)

        emb = model.embed(imgs)
        emb = F.normalize(emb, dim=1)

        embeddings.append(emb.cpu())
        labels.append(lbls.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0).numpy()

    sim_matrix = embeddings @ embeddings.t()
    sim_matrix.fill_diagonal_(-float('inf'))

    top1_indices = sim_matrix.topk(1, dim=1).indices.squeeze(1)
    top1_labels = labels[top1_indices]
    top1_correct = (top1_labels == labels).sum()
    top1 = 100.0 * top1_correct / len(labels)

    topk_indices = sim_matrix.topk(k, dim=1).indices
    topk_labels = labels[topk_indices]
    labels_expanded = labels[:, None]
    top5_correct = (topk_labels == labels_expanded).any(axis=1).sum()
    top5 = 100.0 * top5_correct / len(labels)

    print(f"📝 Overall Results: Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
    return top1, top5