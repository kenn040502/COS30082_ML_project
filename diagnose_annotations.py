"""
Diagnostic: Check for path doubling and class ID issues
"""
import os

def diagnose_annotation_issues(data_dir='.'):
    """Check for common issues in annotation files"""
    
    print("="*60)
    print("ANNOTATION FILES DIAGNOSTIC")
    print("="*60)
    
    list_dir = os.path.join(data_dir, 'list')
    
    files = {
        'train.txt': os.path.join(list_dir, 'train.txt'),
        'test.txt': os.path.join(list_dir, 'test.txt')
    }
    
    all_species_ids = set()
    
    for file_name, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"\n{file_name} not found")
            continue
        
        print(f"\n{file_name}")
        print("-" * 60)
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Check for train/ or test/ prefixes
        prefixed_paths = []
        for line in lines:
            parts = line.split()
            if parts and (parts[0].startswith('train/') or parts[0].startswith('test/')):
                prefixed_paths.append(line)
        
        # Parse entries
        valid = 0
        invalid = 0
        species_ids = []
        
        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                valid += 1
                img_path, class_id = parts
                species_ids.append(int(class_id))
                all_species_ids.add(int(class_id))
            elif len(parts) == 1:
                # Test file might have only paths
                valid += 1
            else:
                invalid += 1
        
        # Report
        print(f"Total entries: {len(lines)}")
        print(f"Valid entries: {valid}")
        if invalid > 0:
            print(f"Invalid entries: {invalid}")
        
        # Path prefix issue
        if prefixed_paths:
            print(f"\nPATH PREFIX ISSUE!")
            print(f"   Found {len(prefixed_paths)} paths with train/test prefix")
            print(f"   Sample: {prefixed_paths[0]}")
            img_path = prefixed_paths[0].split()[0]
            if img_path.startswith('train/'):
                fixed = img_path.replace('train/', '', 1)
            elif img_path.startswith('test/'):
                fixed = img_path.replace('test/', '', 1)
            else:
                fixed = img_path
            print(f"   Should be: {fixed} ...")
            print(f"   (Dataloader adds the train/ or test/ prefix automatically)")
        else:
            print(f"\nNo path prefix issues")
        
        # Class ID range issue
        if species_ids:
            min_id = min(species_ids)
            max_id = max(species_ids)
            unique_ids = len(set(species_ids))
            
            print(f"\nClass ID Statistics:")
            print(f"   Unique IDs: {unique_ids}")
            print(f"   Min ID: {min_id}")
            print(f"   Max ID: {max_id}")
            
            if max_id >= 100:
                print(f"\nCLASS ID OUT OF RANGE!")
                print(f"   Model expects class IDs: 0-99 (100 classes)")
                print(f"   Your data has IDs: {min_id}-{max_id}")
                print(f"   These look like species IDs, not class indices!")
            else:
                print(f"\nClass IDs in valid range")
            
            # Show sample
            print(f"\n   Sample entries:")
            for i, line in enumerate(lines[:3]):
                print(f"      {i+1}. {line}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    issues_found = []
    
    # Check if species IDs need remapping
    if all_species_ids:
        if max(all_species_ids) >= 100:
            issues_found.append("Class IDs out of range (need remapping)")
    
    # Check for prefixed paths
    any_prefixed = False
    for file_name, file_path in files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and (parts[0].startswith('train/') or parts[0].startswith('test/')):
                        any_prefixed = True
                        break
            if any_prefixed:
                break
    
    if any_prefixed:
        issues_found.append("Paths contain train/test prefix (will cause doubling)")
    
    if issues_found:
        print("\nIssues Found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        print("\nSolution: Run fix_annotation_files.py")
        print("   python fix_annotation_files.py")
    else:
        print("\nNo issues detected! Ready to train.")
    
    print("="*60)

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    diagnose_annotation_issues(data_dir)