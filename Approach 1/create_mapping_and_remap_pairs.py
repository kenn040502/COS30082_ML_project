"""
Extract species ID to class index mapping from train.txt image paths
and remap pair files
"""
import os
from collections import defaultdict

def create_mapping_and_remap_pairs(data_dir='.'):
    """
    Extract mapping from image paths and remap pair files
    """
    
    print("="*60)
    print("CREATING MAPPING AND REMAPPING PAIR FILES")
    print("="*60)
    
    list_dir = os.path.join(data_dir, 'list')
    train_file = os.path.join(list_dir, 'train.txt')
    
    # Step 1: Extract species ID to class index mapping from train.txt
    print("\nStep 1: Extracting mapping from train.txt...")
    
    species_to_class = {}
    class_to_species = {}
    
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_path, class_idx = parts
                class_idx = int(class_idx)
                
                # Extract species ID from path
                # Format: photo/SPECIES_ID/IMAGE_ID.jpg
                try:
                    path_parts = img_path.split('/')
                    if len(path_parts) >= 2:
                        species_id = int(path_parts[1])
                        
                        # Map species ID to class index
                        if species_id not in species_to_class:
                            species_to_class[species_id] = class_idx
                            class_to_species[class_idx] = species_id
                except (ValueError, IndexError):
                    # Skip if can't extract species ID
                    continue
    
    print(f"   ✅ Extracted mapping for {len(species_to_class)} species")
    print(f"   Species ID range: {min(species_to_class.keys())} - {max(species_to_class.keys())}")
    print(f"   Class index range: 0 - {max(class_to_species.keys())}")
    
    # Show sample mappings
    print(f"\n   Sample mappings:")
    for i, (species_id, class_idx) in enumerate(list(species_to_class.items())[:10]):
        print(f"      Species {species_id} → Class {class_idx}")
    
    # Step 2: Save mapping to file
    mapping_file = os.path.join(list_dir, 'species_to_class_mapping.txt')
    print(f"\nStep 2: Saving mapping to {os.path.basename(mapping_file)}...")
    
    with open(mapping_file, 'w') as f:
        f.write("# Species ID to Class Index Mapping\n")
        f.write("# Format: species_id class_index\n")
        for species_id in sorted(species_to_class.keys()):
            class_idx = species_to_class[species_id]
            f.write(f"{species_id} {class_idx}\n")
    
    print(f"   ✅ Saved mapping file")
    
    # Step 3: Remap pair files
    print(f"\nStep 3: Remapping pair files...")
    
    pair_files = {
        'class_with_pairs.txt': 'class_with_pairs_indices.txt',
        'class_without_pairs.txt': 'class_without_pairs_indices.txt'
    }
    
    for original_name, remapped_name in pair_files.items():
        original_path = os.path.join(list_dir, original_name)
        remapped_path = os.path.join(list_dir, remapped_name)
        
        if not os.path.exists(original_path):
            print(f"\n   ⚠️  {original_name} not found, skipping...")
            continue
        
        print(f"\n   Processing {original_name}...")
        
        # Read species IDs
        with open(original_path, 'r') as f:
            species_ids = set([int(line.strip()) for line in f if line.strip()])
        
        print(f"      Original: {len(species_ids)} species IDs")
        print(f"      Sample: {sorted(list(species_ids))[:5]}")
        
        # Convert to class indices
        class_indices = set()
        not_found = []
        
        for species_id in species_ids:
            if species_id in species_to_class:
                class_indices.add(species_to_class[species_id])
            else:
                not_found.append(species_id)
        
        print(f"      Converted: {len(class_indices)} class indices")
        print(f"      Sample: {sorted(list(class_indices))[:10]}")
        
        if not_found:
            print(f"      ⚠️  {len(not_found)} species IDs not found in training data:")
            for sid in list(not_found)[:5]:
                print(f"         - {sid}")
        
        # Write remapped file
        with open(remapped_path, 'w') as f:
            for idx in sorted(class_indices):
                f.write(f"{idx}\n")
        
        print(f"      ✅ Created {remapped_name}")
    
    # Step 4: Verify
    print(f"\nStep 4: Verification...")
    
    with_indices_file = os.path.join(list_dir, 'class_with_pairs_indices.txt')
    without_indices_file = os.path.join(list_dir, 'class_without_pairs_indices.txt')
    
    if os.path.exists(with_indices_file):
        with open(with_indices_file, 'r') as f:
            with_count = len([line for line in f if line.strip()])
        print(f"   ✅ class_with_pairs_indices.txt: {with_count} classes")
    
    if os.path.exists(without_indices_file):
        with open(without_indices_file, 'r') as f:
            without_count = len([line for line in f if line.strip()])
        print(f"   ✅ class_without_pairs_indices.txt: {without_count} classes")
    
    total = with_count + without_count if os.path.exists(with_indices_file) else 0
    print(f"   Total: {total} classes (should be 100)")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("   - species_to_class_mapping.txt")
    print("   - class_with_pairs_indices.txt")
    print("   -class_without_pairs_indices.txt")
    print("\nNext step:")
    print("   Update baseline1_dataloader.py to use *_indices.txt files")
    print("="*60)

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    create_mapping_and_remap_pairs(data_dir)