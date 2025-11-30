import os

# Check files
print("Checking pair files...")
with_file = './list/class_with_pairs.txt'
without_file = './list/class_without_pairs.txt'

if os.path.exists(with_file):
    with open(with_file, 'r') as f:
        classes_with = set([int(line.strip()) for line in f if line.strip()])
    print(f"✅ WITH pairs: {len(classes_with)} classes")
    print(f"   Sample: {list(classes_with)[:10]}")
else:
    print(f"❌ {with_file} not found!")

if os.path.exists(without_file):
    with open(without_file, 'r') as f:
        classes_without = set([int(line.strip()) for line in f if line.strip()])
    print(f"✅ WITHOUT pairs: {len(classes_without)} classes")
    print(f"   Sample: {list(classes_without)[:10]}")
else:
    print(f"❌ {without_file} not found!")