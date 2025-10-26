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

        test_path = os.path.join(data_root, 'test')
        self.samples = []
        for img_file in os.listdir(test_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join('test', img_file)
                self.samples.append(img_path)

        # Load ground truth (image_name -> original_class_id)
        gt_file = os.path.join(data_root, 'list', 'groundtruth.txt')
        gt_map = {}
        if os.path.exists(gt_file):
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]  # e.g. "test/12345.jpg" or "12345.jpg" depending on file
                        cls = parts[-1]
                        # store the file name (basename) as key for safer matching
                        gt_map[os.path.basename(img_name)] = cls

        # Map ground-truth original class -> global index (shared classes)
        self.labels = {}
        # Must have been created by PlantFolderDataset earlier in the run
        global_map = getattr(PlantFolderDataset, 'global_class_to_idx', None)
        if global_map is None:
            raise RuntimeError("global_class_to_idx is not set. Initialize PlantFolderDataset first.")

        # Build labels mapping for test images, filter out images whose class not in shared mapping
        valid_samples = []
        for img_path in self.samples:
            basename = os.path.basename(img_path)
            orig_cls = gt_map.get(basename, None)
            if orig_cls is None:
                # try full path key if gt uses path
                orig_cls = gt_map.get(img_path, None)
            if orig_cls is None:
                # no ground truth — mark as -1 (if you want to keep unlabeled test)
                continue

            # orig_cls could be numeric id string; convert to int then to class name if necessary.
            # Your original train folder uses class folder names (e.g. '106461'), so the groundtruth likely uses the same numeric ids.
            orig_cls_key = str(int(orig_cls))
            # Only include if that original class exists in global_map
            if orig_cls_key in global_map:
                mapped = global_map[orig_cls_key]
                self.labels[basename] = mapped
                valid_samples.append(img_path)
            else:
                # not in shared mapping -> exclude from evaluation
                continue

        self.samples = valid_samples
        print(f"📊 Test samples (after mapping/filtering): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        full_img_path = os.path.join(self.data_root, img_path)

        image = Image.open(full_img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        basename = os.path.basename(img_path)
        label = self.labels.get(basename, -1)
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'domain': 1
        }
