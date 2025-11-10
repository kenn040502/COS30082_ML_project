import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from itertools import cycle
from collections import defaultdict

class PlantTripletDataset(Dataset):
    """
    Dataset for Triplet Loss training using a Mixed Strategy:
      1. Cross-Domain Triplet (Shared Classes): Anchor/Negative from Herbarium/Photo, Positive from Photo.
      2. Source-Only Triplet (Herbarium-Only Classes): All samples from Herbarium.
    """
    
    # Static variable to hold the combined class-to-index map
    global_class_to_idx = {}

    def __init__(self, data_root, transform=None, mode='train'):
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        self.samples = {'herbarium': defaultdict(list), 'photo': defaultdict(list)}
        self.herbarium_items = []

        herbarium_path = os.path.join(data_root, 'train', 'herbarium')
        photo_path = os.path.join(data_root, 'train', 'photo')

        if not os.path.exists(herbarium_path):
            raise FileNotFoundError(f"{herbarium_path} not found.")
        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"{photo_path} not found.")

        # --- STEP 1: Build class lists and samples ---
        
        herbarium_classes = set(
            c for c in os.listdir(herbarium_path) if os.path.isdir(os.path.join(herbarium_path, c))
        )
        photo_classes = set(
            c for c in os.listdir(photo_path) if os.path.isdir(os.path.join(photo_path, c))
        )

        self.shared_classes = sorted(list(herbarium_classes & photo_classes))
        self.herbarium_only_classes = sorted(list(herbarium_classes - photo_classes))
        self.all_train_classes = sorted(list(herbarium_classes))
        
        print(f"🌍 Shared classes: {len(self.shared_classes)}")
        print(f"🌿 Herbarium-Only classes: {len(self.herbarium_only_classes)}")

        # Create a class-to-index mapping for ALL classes in the source domain
        PlantTripletDataset.global_class_to_idx = {
            c: i for i, c in enumerate(self.all_train_classes)
        }

        # Collect image paths per class (for ALL classes)
        for domain, dpath in [('herbarium', herbarium_path), ('photo', photo_path)]:
            classes_to_process = herbarium_classes if domain == 'herbarium' else photo_classes
            
            for cls in classes_to_process:
                cpath = os.path.join(dpath, cls)
                images = [
                    os.path.join('train', domain, cls, f)
                    for f in os.listdir(cpath)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if images:
                    self.samples[domain][cls] = images

        # --- STEP 2: Flatten ALL herbarium samples for anchor indexing ---
        self.herbarium_items = []
        for cls in self.all_train_classes:
            if cls in self.samples['herbarium']:
                label = PlantTripletDataset.global_class_to_idx[cls]
                for img in self.samples['herbarium'][cls]:
                    # Append class type for sampling logic in __getitem__
                    cls_type = 'shared' if cls in self.shared_classes else 'herbarium_only'
                    self.herbarium_items.append((img, label, cls, cls_type))

        print(f"✅ Total Herbarium Anchors: {len(self.herbarium_items)}")
        
        # --- STEP 3: Prepare Negative Class Cycles ---
        self.shared_classes_cycle = cycle(self.shared_classes)
        self.all_classes_cycle = cycle(self.all_train_classes)


    def __len__(self):
        return len(self.herbarium_items)

    def __getitem__(self, idx):
        anchor_path, label, cls_name, cls_type = self.herbarium_items[idx]

        # Anchor is always from Herbarium
        anchor_img = Image.open(os.path.join(self.data_root, anchor_path)).convert('RGB')

        pos_cls = cls_name
        
        if cls_type == 'shared':
            # === STRATEGY 1: Cross-Domain Triplet (A: Herbarium, P/N: Photo) ===
            pos_domain = 'photo'
            neg_domain = 'photo'
            neg_class_pool = self.shared_classes
            neg_class_cycle = self.shared_classes_cycle
        else: # cls_type == 'herbarium_only'
            # === STRATEGY 2: Source-Only Triplet (A/P/N: Herbarium) ===
            pos_domain = 'herbarium'
            neg_domain = 'herbarium'
            neg_class_pool = self.all_train_classes # Use all classes in Herbarium for negative mining
            neg_class_cycle = self.all_classes_cycle

        # --- Positive Sample ---
        pos_imgs = self.samples[pos_domain][pos_cls]
        
        # Ensure positive image is not the anchor image itself if domains are the same (Herbarium-Only case)
        pos_path = random.choice(pos_imgs)
        if pos_domain == 'herbarium' and pos_path == anchor_path:
            # Re-sample if the chosen positive is the anchor itself
            pos_path = random.choice([p for p in pos_imgs if p != anchor_path])
        
        pos_img = Image.open(os.path.join(self.data_root, pos_path)).convert('RGB')

        # --- Negative Sample ---
        neg_cls = pos_cls
        while neg_cls == pos_cls:
            # Find a different class
            neg_cls = next(neg_class_cycle) 
        
        neg_imgs = self.samples[neg_domain][neg_cls]
        neg_path = random.choice(neg_imgs)
        neg_img = Image.open(os.path.join(self.data_root, neg_path)).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return {
            'anchor': anchor_img,
            'positive': pos_img,
            'negative': neg_img,
            'label': torch.tensor(label, dtype=torch.long)
        }

class PlantTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transform=None):
        """
        Load all test images and map their ground truth labels to integer indices 
        using the format: 'test/filename.jpg class_id'.
        """
        self.transform = transform
        self.samples = []

        test_dir = os.path.join(data_root, "test")
        # 🎯 FIX 1: Corrected path to the ground truth file
        gt_path = os.path.join(data_root, "list/groundtruth.txt") 
        
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test folder not found: {test_dir}")
        
        # --- 1. Get the global class mapping ---
        class_name_to_idx = getattr(PlantTripletDataset, 'global_class_to_idx', None)
        if class_name_to_idx is None:
             raise RuntimeError("PlantTripletDataset must be initialized first to create class mapping.")

        # --- 2. Load Ground Truth Mapping ---
        gt_map = {}
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    try:
                        # 🎯 FIX 2: Parse using space delimiter (e.g., 'test/1745.jpg 105951')
                        full_path, class_name = line.strip().split(' ')
                        # Use only the filename (e.g., '1745.jpg') as the key
                        fname = os.path.basename(full_path) 
                        gt_map[fname] = class_name
                    except ValueError:
                        # Skip lines that don't match the expected format
                        continue
        else:
             print("⚠️ Warning: Ground truth file not found at list/groundtruth.txt. Test results will be meaningless.")
        
        # --- 3. Match Images to Labels ---
        skipped = 0
        for fname in os.listdir(test_dir):
            img_path = os.path.join(test_dir, fname)
            
            if os.path.isfile(img_path) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                # Use filename (fname) to look up the class name
                class_name = gt_map.get(fname, None) 
                
                if class_name is not None and class_name in class_name_to_idx:
                    # Found a valid class name in the training map
                    label = class_name_to_idx[class_name]
                    self.samples.append((img_path, label))
                else:
                    # Image not in ground truth or class not in training set 
                    self.samples.append((img_path, -1)) 
                    skipped += 1
            else:
                skipped += 1
        
        # Since the test set is expected to be part of the training classes, 
        # printing only samples with valid labels is more informative.
        valid_samples = sum(1 for _, label in self.samples if label != -1)
        print(f"✅ Loaded {len(self.samples)} test samples (Valid labels: {valid_samples}, Skipped: {len(self.samples) - valid_samples})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Note: label is passed as is. The metrics function must filter out -1 labels.
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}