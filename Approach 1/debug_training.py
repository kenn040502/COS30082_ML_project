import os
import torch

print("="*60)
print("TRAINING DIAGNOSTIC")
print("="*60)

print("\n1. Checking model file...")
from baseline1_model import create_improved_model

model = create_improved_model(num_classes=100)
print(f"   Model class: {model.__class__.__name__}")
print(f"   Has CBAM: {hasattr(model, 'cbam')}")
print(f"   use_cbam: {model.use_cbam}")
print(f"   use_gem: {model.use_gem}")

print("\n2. Checking pair data files...")
files = ['list/class_with_pairs.txt', 'list/class_without_pairs.txt']
for f in files:
    if os.path.exists(f):
        lines = len(open(f).readlines())
        print(f"   ✅ {f} ({lines} classes)")
    else:
        print(f"   ❌ {f} MISSING!")

print("\n3. Testing dataloader...")
from baseline1_dataloader import create_improved_dataloaders

train_loader, test_loader, num_classes, dataset_info = create_improved_dataloaders(
    data_dir='.', batch_size=16, num_workers=0, img_size=224
)

print(f"   ✅ Dataloaders created")
print(f"   Num classes: {num_classes}")
print(f"   Classes WITH pairs: {len(dataset_info['classes_with_pairs'])}")
print(f"   Classes WITHOUT pairs: {len(dataset_info['classes_without_pairs'])}")

if len(dataset_info['classes_with_pairs']) > 0:
    print(f"   ✅ Pair data loaded successfully!")
else:
    print(f"   ❌ Pair data NOT loading!")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)