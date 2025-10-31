"""
Diagnostic script to check data directory structure
"""
import os
import sys

def check_data_structure(data_dir='.'):
    """Check if all required files and directories exist"""
    
    print("="*60)
    print("DATA STRUCTURE DIAGNOSTIC")
    print("="*60)
    print(f"\nChecking directory: {os.path.abspath(data_dir)}")
    
    # Expected structure
    expected = {
        'directories': ['train', 'test', 'list'],
        'files': [
            'list/train.txt',
            'list/test.txt',
            'list/class_with_pairs.txt',
            'list/class_without_pairs.txt'
        ]
    }
    
    # Check directories
    print("\nDirectories:")
    for dirname in expected['directories']:
        path = os.path.join(data_dir, dirname)
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {dirname:20s} {'EXISTS' if exists else 'MISSING'}")
        
        if exists and os.path.isdir(path):
            # Count files in directory
            try:
                files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                print(f"      - Files: {len(files)}, Subdirectories: {len(subdirs)}")
            except Exception as e:
                print(f"      - Error reading: {e}")
    
    # Check files
    print("\nFiles:")
    for filepath in expected['files']:
        full_path = os.path.join(data_dir, filepath)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        
        if exists:
            # Get file size and line count
            size = os.path.getsize(full_path)
            try:
                with open(full_path, 'r') as f:
                    lines = len(f.readlines())
                print(f"   {status} {filepath:40s} {size:>10,} bytes, {lines:>6} lines")
            except Exception as e:
                print(f"   {status} {filepath:40s} {size:>10,} bytes, Error: {e}")
        else:
            print(f"   {status} {filepath:40s} MISSING")
    
    # Check annotation files content
    print("\nAnnotation Files Details:")
    
    for ann_file in ['list/train.txt', 'list/test.txt']:
        full_path = os.path.join(data_dir, ann_file)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                if len(lines) == 0:
                    print(f"\n    {ann_file} is EMPTY!")
                else:
                    print(f"\n   {ann_file}:")
                    print(f"      Total entries: {lines}")
                    
                    # Show first few entries
                    print(f"      First 3 entries:")
                    for i, line in enumerate(lines[:3]):
                        print(f"         {i+1}. {line}")
                    
                    # Parse and check format
                    valid = 0
                    invalid = 0
                    for line in lines:
                        parts = line.split()
                        if len(parts) == 2:
                            valid += 1
                        else:
                            invalid += 1
                    
                    print(f"      Valid entries: {valid}")
                    if invalid > 0:
                        print(f"       Invalid entries: {invalid}")
                        
            except Exception as e:
                print(f"\n   Error reading {ann_file}: {e}")
        else:
            print(f"\n   {ann_file} NOT FOUND")
    
    print("\n" + "="*60)
    
    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    
    test_txt = os.path.join(data_dir, 'list/test.txt')
    if not os.path.exists(test_txt):
        print("    test.txt is missing!")
        print("   → If you don't have a test set, you can:")
        print("      1. Split your training data into train/val sets")
        print("      2. Create an empty test.txt file (training will skip validation)")
        print("      3. Use train.txt for both training and testing (not recommended)")
    
    elif os.path.exists(test_txt):
        with open(test_txt, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) == 0:
            print("    test.txt exists but is EMPTY!")
            print("   → Training will proceed without validation")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    check_data_structure(data_dir)