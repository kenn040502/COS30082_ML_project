# utils/class_utils.py
import os

def load_class_lists(data_root):
    """
    Load classes with pairs and without pairs from text files
    """
    with_pairs_path = os.path.join(data_root, "list", "class_with_pairs.txt")
    without_pairs_path = os.path.join(data_root, "list", "class_without_pairs.txt")
    
    # Load classes with pairs
    with open(with_pairs_path, 'r') as f:
        classes_with_pairs = [line.strip() for line in f.readlines() if line.strip()]
    
    # Load classes without pairs  
    with open(without_pairs_path, 'r') as f:
        classes_without_pairs = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📊 Loaded {len(classes_with_pairs)} classes with pairs")
    print(f"📊 Loaded {len(classes_without_pairs)} classes without pairs")
    
    return classes_with_pairs, classes_without_pairs

def get_class_type_mapping(global_class_to_idx, classes_with_pairs, classes_without_pairs):
    """
    Create mapping from class index to class type (with_pairs/without_pairs)
    """
    class_type_mapping = {}
    
    for class_name, class_idx in global_class_to_idx.items():
        if class_name in classes_with_pairs:
            class_type_mapping[class_idx] = 'with_pairs'
        elif class_name in classes_without_pairs:
            class_type_mapping[class_idx] = 'without_pairs'
        else:
            class_type_mapping[class_idx] = 'unknown'
            print(f"⚠️  Warning: Class {class_name} (idx {class_idx}) not found in either list")
    
    return class_type_mapping