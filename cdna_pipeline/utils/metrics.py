import torch
import numpy as np

def calculate_accuracy(outputs, labels, topk=(1, 5)):
    """
    Calculate top-k accuracy
    outputs: model predictions (batch_size, num_classes)
    labels: ground truth labels (batch_size,)
    topk: tuple of k values for top-k accuracy
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate_model(feature_extractor, classifier, dataloader, device, topk=(1, 5)):
    """
    Evaluate model on a dataloader and return top-k accuracies
    """
    feature_extractor.eval()
    classifier.eval()
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            features = feature_extractor(images)
            outputs = classifier(features)
            
            # Calculate accuracy
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=topk)
            
            top1_correct += acc1.item() * images.size(0) / 100
            top5_correct += acc5.item() * images.size(0) / 100
            total += images.size(0)
    
    top1_accuracy = 100.0 * top1_correct / total
    top5_accuracy = 100.0 * top5_correct / total
    
    return top1_accuracy, top5_accuracy

def evaluate_domain_accuracy(feature_extractor, classifier, dataloader, device, domain_name=""):
    """
    Simple evaluation that returns top-1 accuracy for a specific domain
    """
    feature_extractor.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            features = feature_extractor(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy