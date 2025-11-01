"""
Fix train.txt and test.txt:
1. Remove path doubling (train/train -> train)
2. Remap species IDs to class indices (0-99)
"""
import os
from collections import OrderedDict

def fix_annotation_files(data_dir='.'):
    """
    Fix annotation files by:
    1. Removing path doubling
    2. Remapping species IDs to 0-99 range
    """
    
    print("="*60)
    print("FIXING ANNOTATION FILES")
    print("="*60)
    
    list_dir = os.path.join(data_dir, 'list')
    
    # Files to process
    files_to_fix = {
        'train.txt': os.path.join(list_dir, 'train.txt'),
        'test.txt': os.path.join(list_dir, 'test.txt')
    }
    
    # Step 1: Build species ID to class index mapping from train.txt
    print("\nStep 1: Building species ID mapping...")
    
    train_file = files_to_fix['train.txt']
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found!")
        return
    
    # Read all species IDs from train.txt
    species_ids = set()
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                _, species_id = parts
                species_ids.add(int(species_id))
    
    # Sort and create mapping: species_id -> class_index (0-99)
    species_ids_sorted = sorted(species_ids)
    species_to_class = {species_id: idx for idx, species_id in enumerate(species_ids_sorted)}
    
    print(f"   Found {len(species_ids)} unique species")
    print(f"   Species ID range: {min(species_ids)} - {max(species_ids)}")
    print(f"   Will remap to: 0 - {len(species_ids)-1}")
    
    # Show sample mapping
    print(f"\n   Sample mappings:")
    for i, (species_id, class_idx) in enumerate(list(species_to_class.items())[:5]):
        print(f"      Species ID {species_id} → Class {class_idx}")
    
    # Step 2: Fix each annotation file
    for file_name, file_path in files_to_fix.items():
        if not os.path.exists(file_path):
            print(f"\n  Skipping {file_name} (not found)")
            continue
        
        print(f"\n Fixing {file_name}...")
        
        # Read original
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Fix each line
        fixed_lines = []
        errors = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            if len(parts) == 2:
                img_path, species_id = parts
                species_id = int(species_id)
                
                # Remove train/ or test/ prefix since dataloader adds it
                if img_path.startswith('train/'):
                    img_path = img_path.replace('train/', '', 1)
                elif img_path.startswith('test/'):
                    img_path = img_path.replace('test/', '', 1)
                
                # Remap species ID to class index
                if species_id in species_to_class:
                    class_idx = species_to_class[species_id]
                    fixed_lines.append(f"{img_path} {class_idx}\n")
                else:
                    errors.append(f"Line {i+1}: Unknown species ID {species_id}")
            else:
                # Handle test.txt entries with only image path (no label yet)
                if len(parts) == 1:
                    img_path = parts[0]
                    # Remove test/ prefix since dataloader adds it
                    if img_path.startswith('test/'):
                        img_path = img_path.replace('test/', '', 1)
                    # Keep as is (will need groundtruth for label)
                    fixed_lines.append(f"{img_path}\n")
                else:
                    errors.append(f"Line {i+1}: Invalid format")
        
        # Backup original
        backup_path = file_path + '.original_backup'
        if not os.path.exists(backup_path):
            print(f"   Backing up original to {os.path.basename(backup_path)}")
            with open(backup_path, 'w') as f:
                f.writelines(lines)
        
        # Write fixed version
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"   Fixed {len(fixed_lines)} entries")
        
        if errors:
            print(f"    {len(errors)} errors:")
            for err in errors[:5]:
                print(f"      {err}")
    
    # Step 3: Save species mapping for reference
    mapping_file = os.path.join(list_dir, 'species_id_to_class_mapping.txt')
    print(f"\n Saving species mapping to {os.path.basename(mapping_file)}...")
    with open(mapping_file, 'w') as f:
        f.write("# Species ID to Class Index Mapping\n")
        f.write("# Format: species_id class_index\n")
        for species_id, class_idx in species_to_class.items():
            f.write(f"{species_id} {class_idx}\n")
    
    print(f"   Saved mapping for {len(species_to_class)} species")
    
    # Step 4: Verify fixes
    print(f"\n Verifying fixes...")
    
    for file_name, file_path in files_to_fix.items():
        if not os.path.exists(file_path):
            continue
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Check for train/ or test/ prefixes (should be removed)
        prefixed_paths = [line for line in lines if line.startswith('train/') or line.startswith('test/')]
        
        # Check for out-of-range class IDs
        out_of_range = []
        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                _, class_id = parts
                class_id = int(class_id)
                if class_id < 0 or class_id >= len(species_to_class):
                    out_of_range.append(line)
        
        print(f"\n   {file_name}:")
        print(f"      Total entries: {len(lines)}")
        print(f"       Paths with train/test prefix: {len(prefixed_paths)}")
        print(f"       Out-of-range IDs: {len(out_of_range)}")
        
        if prefixed_paths:
            print(f"      Sample prefixed path: {prefixed_paths[0]}")
        if out_of_range:
            print(f"      Sample out-of-range: {out_of_range[0]}")
    
    print("\n" + "="*60)
    print("FIXES COMPLETE!")
    print("="*60)
    print("\n Next steps:")
    print("   1. Verify the fixes look correct")
    print("   2. Original files backed up with .original_backup extension")
    print("   3. Re-run training: python src/improved_baseline1_train.py")
    print("="*60)

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    fix_annotation_files(data_dir)