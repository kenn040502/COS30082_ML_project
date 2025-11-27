import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random

class PlantFolderDataset(Dataset):
    def __init__(self, data_root, domain='herbarium', split='train', transform=None, val_ratio=0.2):
        self.data_root = data_root
        self.transform = transform
        self.domain = domain
        self.split = split
        
        # Path to the domain folder
        domain_path = os.path.join(data_root, 'train', domain)
        
        if not os.path.exists(domain_path):
            raise ValueError(f"Domain path {domain_path} does not exist!")
        
        # Collect all images and labels from folder structure
        self.samples = []
        
        # Assuming class folders are subdirectories
        class_folders = sorted([f for f in os.listdir(domain_path) 
                              if os.path.isdir(os.path.join(domain_path, f))])
        
        # CRITICAL: Use consistent class mapping across ALL datasets
        if not hasattr(PlantFolderDataset, 'global_class_to_idx'):
            herbarium_path = os.path.join(data_root, 'train', 'herbarium')
            photo_path = os.path.join(data_root, 'train', 'photo')

            herbarium_classes = set([
                f for f in os.listdir(herbarium_path)
                if os.path.isdir(os.path.join(herbarium_path, f))
            ])
            photo_classes = set([
                f for f in os.listdir(photo_path)
                if os.path.isdir(os.path.join(photo_path, f))
            ])

            shared_classes = sorted(list(herbarium_classes & photo_classes))
            print(f"🌍 Shared classes detected: {len(shared_classes)} common between herbarium & photo")

            PlantFolderDataset.global_class_to_idx = {cls: idx for idx, cls in enumerate(shared_classes)}
            PlantFolderDataset.global_idx_to_class = {idx: cls for idx, cls in enumerate(shared_classes)}

        
        # Map current domain classes to global indices
        self.class_to_idx = {}
        for class_name in class_folders:
            if class_name in PlantFolderDataset.global_class_to_idx:
                self.class_to_idx[class_name] = PlantFolderDataset.global_class_to_idx[class_name]
            else:
                # This class doesn't exist in herbarium - skip or handle specially
                print(f"⚠️  Class {class_name} not in global mapping, skipping")
                continue
        
        # Collect all image paths with GLOBAL labels
        for class_name in class_folders:
            if class_name not in self.class_to_idx:
                continue
                
            class_path = os.path.join(domain_path, class_name)
            global_class_idx = self.class_to_idx[class_name]
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join('train', domain, class_name, img_file)
                    self.samples.append((img_path, global_class_idx))
        
        print(f"📊 {domain} domain - {len(self.samples)} images, {len(self.class_to_idx)}/{len(PlantFolderDataset.global_class_to_idx)} classes")
        
        # Split into train/val if needed
        if split in ['train', 'val'] and domain == 'herbarium':  # Only split herbarium
            random.seed(42)
            random.shuffle(self.samples)
            split_idx = int(len(self.samples) * (1 - val_ratio))
            
            if split == 'train':
                self.samples = self.samples[:split_idx]
            else:  # val
                self.samples = self.samples[split_idx:]
            
            print(f"📊 After {split} split: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_img_path = os.path.join(self.data_root, img_path)
        
        image = Image.open(full_img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Domain: 0 for herbarium, 1 for photo
        domain_label = 0 if self.domain == 'herbarium' else 1
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'domain': domain_label
        }


class PlantTestDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        # groundtruth file
        gt_path = os.path.join(data_root, "list", "groundtruth.txt")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Groundtruth file not found: {gt_path}")

        # must have mapping from training set
        if not hasattr(PlantFolderDataset, 'global_idx_to_class'):
            raise RuntimeError("❌ global_idx_to_class not found. Initialize PlantFolderDataset first.")

        # reverse mapping: class_name → internal idx
        class_name_to_idx = PlantFolderDataset.global_class_to_idx
        idx_to_name = PlantFolderDataset.global_idx_to_class

        valid_class_names = set(class_name_to_idx.keys())

        skipped = 0
        with open(gt_path, "r") as f:
            for line in f:
                path, raw_label = line.strip().split()
                img_path = os.path.join(data_root, path)
                raw_label = str(raw_label)  # your folder names are likely string IDs

                # skip if label not part of shared class list
                if raw_label not in valid_class_names:
                    skipped += 1
                    continue

                label = class_name_to_idx[raw_label]

                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                else:
                    print(f"⚠️ Warning: image {img_path} not found, skipping.")

        print(f"✅ Loaded {len(self.samples)} valid test samples "
              f"(skipped {skipped} unmatched IDs from groundtruth.txt)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': torch.tensor(label, dtype=torch.long)}
