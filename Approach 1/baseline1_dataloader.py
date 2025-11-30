import os
from collections import Counter
from typing import Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _load_labels(annotation_file: str) -> List[int]:
    """Load labels from an annotation file."""
    labels: List[int] = []
    if not os.path.exists(annotation_file):
        return labels
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    labels.append(int(parts[1]))
                except ValueError:
                    continue
    return labels


class ImprovedPlantDataset(Dataset):
    def __init__(
        self,
        root_dir,
        annotation_file,
        transform=None,
        use_albumentations=False,
        class_with_pairs_file=None,
        class_without_pairs_file=None,
        label_map: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to train.txt or test.txt
            transform: Torchvision transforms (if use_albumentations=False)
            use_albumentations: Whether to use Albumentations (more powerful)
            class_with_pairs_file: Path to class_with_pairs.txt (species ids or indices)
            class_without_pairs_file: Path to class_without_pairs.txt (species ids or indices)
            label_map: Optional mapping from original class ids to 0..N-1
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.label_map = label_map or {}
        
        # Load annotations
        self.data = []
        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                img_path, class_id = parts
                class_id = int(class_id)
                if self.label_map:
                    if class_id in self.label_map:
                        mapped = self.label_map[class_id]
                    elif class_id in self.label_map.values():
                        # Already mapped value in file
                        mapped = class_id
                    else:
                        raise ValueError(f"Found label {class_id} not present in label_map")
                else:
                    mapped = class_id
                self.data.append((img_path, mapped))
        
        # Load species pair information
        self.classes_with_pairs = set()
        self.classes_without_pairs = set()
        
        if class_with_pairs_file and os.path.exists(class_with_pairs_file):
            with open(class_with_pairs_file, "r") as f:
                raw_ids = [int(line.strip()) for line in f if line.strip()]
            mapped_ids = []
            skipped = 0
            for cid in raw_ids:
                if self.label_map and cid in self.label_map:
                    mapped_ids.append(self.label_map[cid])
                elif self.label_map and cid in self.label_map.values():
                    mapped_ids.append(cid)
                elif not self.label_map:
                    mapped_ids.append(cid)
                else:
                    skipped += 1
            self.classes_with_pairs = set(mapped_ids)
            if skipped:
                print(f"   Warning: skipped {skipped} ids from {class_with_pairs_file} (not in label_map)")
        
        if class_without_pairs_file and os.path.exists(class_without_pairs_file):
            with open(class_without_pairs_file, "r") as f:
                raw_ids = [int(line.strip()) for line in f if line.strip()]
            mapped_ids = []
            skipped = 0
            for cid in raw_ids:
                if self.label_map and cid in self.label_map:
                    mapped_ids.append(self.label_map[cid])
                elif self.label_map and cid in self.label_map.values():
                    mapped_ids.append(cid)
                elif not self.label_map:
                    mapped_ids.append(cid)
                else:
                    skipped += 1
            self.classes_without_pairs = set(mapped_ids)
            if skipped:
                print(f"   Warning: skipped {skipped} ids from {class_without_pairs_file} (not in label_map)")
        
        print(f"Loaded {len(self.data)} images")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        # Check if dataset is empty
        if len(self.data) == 0:
            print("\nWARNING: Dataset is EMPTY!")
            print("   No images were loaded. Please check:")
            print("   1. Annotation file exists and is not empty")
            print("   2. Image paths in annotation file are correct")
            print("   3. Root directory path is correct")
            print("="*60 + "\n")
            return
        
        # Count by domain
        herbarium_count = sum(1 for img_path, _ in self.data if 'herb' in img_path.lower())
        field_count = len(self.data) - herbarium_count
        
        print(f"\nImage Distribution:")
        print(f"   Herbarium: {herbarium_count}")
        print(f"   Field: {field_count}")
        print(f"   Total: {len(self.data)}")
        
        # Class distribution
        class_counts = Counter([class_id for _, class_id in self.data])
        num_classes = len(class_counts)
        
        print(f"\nSpecies Distribution:")
        print(f"   Number of species: {num_classes}")
        if num_classes > 0:
            print(f"   Min samples: {min(class_counts.values())}")
            print(f"   Max samples: {max(class_counts.values())}")
            print(f"   Avg samples: {sum(class_counts.values()) / num_classes:.1f}")
        
        # Pair analysis
        if self.classes_with_pairs or self.classes_without_pairs:
            print(f"\nSpecies Pair Analysis:")
            print(f"   Species WITH pairs: {len(self.classes_with_pairs)}")
            print(f"   Species WITHOUT pairs: {len(self.classes_without_pairs)}")
        
        print("="*60 + "\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        full_path = os.path.join(self.root_dir, img_path)
        
        try:
            if self.use_albumentations:
                # Use Albumentations (more powerful)
                image = cv2.imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
            else:
                # Use torchvision transforms
                image = Image.open(full_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
            
            return image, class_id
        
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return blank image that matches transform output format
            # Create a normalized blank tensor that can be properly batched
            blank = torch.zeros(3, 224, 224, dtype=torch.float32)
            # Apply same normalization as transforms
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            blank = (blank - mean) / std
            return blank.clone(), class_id


def get_improved_transforms(img_size=224, is_training=True, use_albumentations=False):
    """
    Args:
        img_size: Target image size
        is_training: Training or validation
        use_albumentations: Use Albumentations (more powerful) or torchvision
    
    Returns:
        transform: Augmentation pipeline
    """
    
    if use_albumentations:
        # Albumentations - MORE POWERFUL augmentation
        if is_training:
            return A.Compose([
                A.Resize(img_size + 32, img_size + 32),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.8),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=3),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    else:
        if is_training:
            return transforms.Compose([
                transforms.Resize(img_size + 32),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                # Stronger color augmentation
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                # Occasionally convert to grayscale (helps with domain shift!)
                transforms.RandomGrayscale(p=0.1),
                # Random affine for slight distortions
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Random erasing (helps regularization)
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))
            ])
        else:
            # TEST/VAL - Use CenterCrop to ensure exact size!
            return transforms.Compose([
                transforms.Resize(256),  # Resize shorter side to 256
                transforms.CenterCrop(img_size),  # Crop to exact img_size x img_size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


def create_improved_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=224,
                                use_albumentations=False):
    """
    Args:
        data_dir: Root directory with train/, test/, list/ folders
        batch_size: Batch size (32 recommended, or 16 if memory limited)
        num_workers: Number of data loading workers
        img_size: Input image size (224 standard, 320 or 384 for better accuracy)
        use_albumentations: Use Albumentations library (more powerful)
    
    Returns:
        train_loader, test_loader, num_classes, dataset_info
    """
    
    # File paths
    train_annotation = os.path.join(data_dir, 'list', 'train.txt')
    test_annotation = os.path.join(data_dir, 'list', 'test.txt')
    class_with_pairs = os.path.join(data_dir, 'list', 'class_with_pairs_indices.txt')
    class_without_pairs = os.path.join(data_dir, 'list', 'class_without_pairs_indices.txt')
    if not os.path.exists(class_with_pairs):
        class_with_pairs = os.path.join(data_dir, 'list', 'class_with_pairs.txt')
    if not os.path.exists(class_without_pairs):
        class_without_pairs = os.path.join(data_dir, 'list', 'class_without_pairs.txt')
    
    # Validate files exist
    print(f"\nValidating data files...")
    if not os.path.exists(train_annotation):
        raise FileNotFoundError(f"❌ Training annotation file not found: {train_annotation}")
    if not os.path.exists(test_annotation):
        print(f"Test annotation file not found: {test_annotation}")
        print(f"   Creating empty test dataset...")
    
    # Build label map so labels are contiguous 0..N-1 (required by CrossEntropyLoss)
    train_labels = _load_labels(train_annotation)
    test_labels = _load_labels(test_annotation)
    all_labels = sorted(set(train_labels + test_labels))
    if len(all_labels) == 0:
        raise ValueError("No labels found in train/test annotation files.")
    is_contiguous_zero_based = all_labels == list(range(len(all_labels))) and all_labels[0] == 0
    label_map = {lbl: lbl for lbl in all_labels} if is_contiguous_zero_based else {label: idx for idx, label in enumerate(all_labels)}
    if is_contiguous_zero_based:
        print("\nLabel mapping: labels already 0..N-1; using identity map.")
    else:
        print("\nLabel mapping created (labels were non-contiguous or not zero-based):")
        print(f"   Original id range: {min(all_labels)} - {max(all_labels)} ({len(all_labels)} unique)")
        print(f"   Remapped to range: 0 - {len(all_labels) - 1}")
    
    # Get transforms
    train_transform = get_improved_transforms(img_size, is_training=True, use_albumentations=use_albumentations)
    test_transform = get_improved_transforms(img_size, is_training=False, use_albumentations=use_albumentations)
    
    print(f"\n{'='*60}")
    print(f"Using {'Albumentations' if use_albumentations else 'Torchvision'} augmentation")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Create datasets
    print("\nCreating Training Dataset...")
    train_dataset = ImprovedPlantDataset(
        root_dir=data_dir,
        annotation_file=train_annotation,
        transform=train_transform,
        use_albumentations=use_albumentations,
        class_with_pairs_file=class_with_pairs,
        class_without_pairs_file=class_without_pairs,
        label_map=label_map
    )
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty! Cannot proceed.")
    
    print("\nCreating Test Dataset...")
    if os.path.exists(test_annotation):
        test_dataset = ImprovedPlantDataset(
            root_dir=data_dir,
            annotation_file=test_annotation,
            transform=test_transform,
            use_albumentations=use_albumentations,
            class_with_pairs_file=class_with_pairs,
            class_without_pairs_file=class_without_pairs,
            label_map=label_map
        )
    else:
        print("Test dataset file not found. Creating empty placeholder...")
        # Create an empty test dataset using a copy of train with no samples
        test_dataset = ImprovedPlantDataset(
            root_dir=data_dir,
            annotation_file=train_annotation,
            transform=test_transform,
            use_albumentations=use_albumentations,
            class_with_pairs_file=class_with_pairs,
            class_without_pairs_file=class_without_pairs,
            label_map=label_map
        )
        test_dataset.data = []  # Empty it
    
    # Get number of classes
    num_classes = len(label_map)
    print(f"\nTotal classes: {num_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    # Only create test loader if we have test data
    if len(test_dataset) > 0:
        # Use fewer workers for test to avoid worker process errors
        test_workers = min(num_workers, 2) if num_workers > 0 else 0
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=test_workers,  # Fewer workers for stability
            pin_memory=True
        )
        if test_workers < num_workers:
            print(f"Using {test_workers} workers for test loader (reduced for stability)")
    else:
        print("Test dataset is empty - validation will be skipped")
        test_loader = None
    
    # Dataset info
    dataset_info = {
        'num_classes': num_classes,
        'classes_with_pairs': train_dataset.classes_with_pairs,
        'classes_without_pairs': train_dataset.classes_without_pairs,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset) if test_dataset else 0,
        'label_map': label_map
    }
    
    print(f"Train batches: {len(train_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    else:
        print(f"No test batches (test set empty)")
    
    return train_loader, test_loader, num_classes, dataset_info


if __name__ == "__main__":
    # Test dataloader
    print("Testing dataloader...")
    
    train_loader, test_loader, num_classes, info = create_improved_dataloaders(
        data_dir='.',
        batch_size=16,
        img_size=224,
        use_albumentations=False 
    )
    
    # Test loading one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch loaded successfully!")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Sample labels: {labels[:5].tolist()}")
