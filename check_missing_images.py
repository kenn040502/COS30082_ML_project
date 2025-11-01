"""
Check which test images are missing
"""
import os

def check_missing_images(data_dir='.'):
    """Check which images in annotation files don't exist"""
    
    print("="*60)
    print("CHECKING FOR MISSING IMAGES")
    print("="*60)
    
    files_to_check = {
        'train': ('list/train.txt', 'train'),
        'test': ('list/test.txt', 'test')
    }
    
    for dataset_name, (ann_file, img_dir) in files_to_check.items():
        ann_path = os.path.join(data_dir, ann_file)
        
        if not os.path.exists(ann_path):
            print(f"\n{ann_file} not found, skipping...")
            continue
        
        print(f"\nChecking {dataset_name} dataset...")
        
        with open(ann_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        missing = []
        found = 0
        
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            
            img_path = parts[0]
            full_path = os.path.join(data_dir, img_dir, img_path)
            
            if os.path.exists(full_path):
                found += 1
            else:
                missing.append((line, full_path))
        
        print(f"   Found: {found} images")
        print(f"   Missing: {len(missing)} images")
        
        if missing:
            print(f"\n   Missing images (first 10):")
            for i, (line, full_path) in enumerate(missing[:10], 1):
                print(f"      {i}. {full_path}")
            
            if len(missing) > 10:
                print(f"      ... and {len(missing) - 10} more")
            
            # Save full list
            output_file = os.path.join(data_dir, f'missing_{dataset_name}_images.txt')
            with open(output_file, 'w') as f:
                for line, full_path in missing:
                    f.write(f"{full_path}\n")
            print(f"\n   Full list saved to: {output_file}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    check_missing_images(data_dir)