"""
Merge test.txt and groundtruth.txt to create properly formatted test annotations
"""
import os

def merge_test_groundtruth(data_dir='.'):
    """
    Merge test.txt (image paths) with groundtruth.txt (labels)
    to create properly formatted test annotations.
    """
    
    test_file = os.path.join(data_dir, 'list', 'test.txt')
    groundtruth_file = os.path.join(data_dir, 'list', 'groundtruth.txt')
    output_file = os.path.join(data_dir, 'list', 'test_with_labels.txt')
    
    print("="*60)
    print("MERGING TEST.TXT WITH GROUNDTRUTH.TXT")
    print("="*60)
    
    # Check files exist
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return
    
    if not os.path.exists(groundtruth_file):
        print(f"Error: {groundtruth_file} not found!")
        return
    
    # Read test.txt (image paths only)
    print(f"\n Reading {test_file}...")
    with open(test_file, 'r') as f:
        test_paths = [line.strip() for line in f if line.strip()]
    print(f"   Found {len(test_paths)} image paths")
    
    # Read groundtruth.txt (labels)
    print(f"\n Reading {groundtruth_file}...")
    with open(groundtruth_file, 'r') as f:
        groundtruth_lines = [line.strip() for line in f if line.strip()]
    print(f"   Found {len(groundtruth_lines)} ground truth entries")
    
    # Parse groundtruth - could be in different formats
    # Try format 1: "image_path class_id"
    # Try format 2: "class_id" only (one per line, matching order of test.txt)
    
    labels = []
    if len(groundtruth_lines) > 0:
        first_line_parts = groundtruth_lines[0].split()
        
        if len(first_line_parts) == 2:
            # Format: "image_path class_id"
            print("   Format detected: image_path class_id")
            groundtruth_dict = {}
            for line in groundtruth_lines:
                parts = line.split()
                if len(parts) == 2:
                    img_path, class_id = parts
                    groundtruth_dict[img_path] = int(class_id)
            
            # Match with test paths
            for path in test_paths:
                if path in groundtruth_dict:
                    labels.append(groundtruth_dict[path])
                else:
                    print(f"     Warning: No label found for {path}")
                    labels.append(-1)  # Placeholder
        
        elif len(first_line_parts) == 1:
            # Format: "class_id" only (one per line)
            print("   Format detected: class_id per line")
            labels = [int(line) for line in groundtruth_lines]
        
        else:
            print(f"    Unknown format in groundtruth.txt")
            return
    
    # Verify counts match
    if len(test_paths) != len(labels):
        print(f"\n  WARNING: Mismatch in counts!")
        print(f"   Test paths: {len(test_paths)}")
        print(f"   Labels: {len(labels)}")
        print(f"   Will only use matching entries")
        min_len = min(len(test_paths), len(labels))
        test_paths = test_paths[:min_len]
        labels = labels[:min_len]
    
    # Write merged file
    print(f"\n  Writing to {output_file}...")
    with open(output_file, 'w') as f:
        for img_path, class_id in zip(test_paths, labels):
            f.write(f"{img_path} {class_id}\n")
    
    print(f" Successfully created {output_file}")
    print(f"   Total entries: {len(test_paths)}")
    
    # Show sample entries
    print(f"\n Sample entries:")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:5], 1):
            print(f"   {i}. {line.strip()}")
    
    print("\n" + "="*60)
    print(" NEXT STEPS:")
    print("="*60)
    print("1. Verify the output looks correct")
    print("2. Update your training script to use 'test_with_labels.txt'")
    print("   OR")
    print("3. Backup original test.txt and replace it:")
    print(f"   cp {test_file} {test_file}.backup")
    print(f"   cp {output_file} {test_file}")
    print("="*60)
    
    return output_file

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    merge_test_groundtruth(data_dir)