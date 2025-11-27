import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from baseline1_model import create_improved_model
from baseline1_dataloader import create_improved_dataloaders


class ModelTester:
    """
    Comprehensive model testing with Top-1/Top-5 accuracy and gap analysis
    """
    def __init__(self, model, test_loader, device, dataset_info):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.dataset_info = dataset_info
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate_comprehensive(self):
        """
        Comprehensive evaluation with all metrics
        """
        self.model.eval()
        
        # Overall counters
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        running_loss = 0.0
        
        # With/without pairs tracking
        classes_with = self.dataset_info['classes_with_pairs']
        classes_without = self.dataset_info['classes_without_pairs']
        
        # Top-1 with/without pairs
        correct_top1_with = 0
        total_with = 0
        correct_top1_without = 0
        total_without = 0
        
        # Top-5 with/without pairs
        correct_top5_with = 0
        correct_top5_without = 0
        
        # Per-class accuracy (for detailed analysis)
        class_correct_top1 = {}
        class_correct_top5 = {}
        class_total = {}
        
        print("\n🔍 Running comprehensive evaluation...")
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                # Top-1 predictions
                _, predicted_top1 = outputs.max(1)
                
                # Top-5 predictions
                _, predicted_top5 = outputs.topk(5, dim=1)
                
                total += labels.size(0)
                
                # Process each sample
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred_top1 = predicted_top1[i].item()
                    preds_top5 = predicted_top5[i].tolist()
                    
                    # Initialize per-class counters
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct_top1[label] = 0
                        class_correct_top5[label] = 0
                    
                    class_total[label] += 1
                    
                    # Top-1 overall
                    if pred_top1 == label:
                        correct_top1 += 1
                        class_correct_top1[label] += 1
                    
                    # Top-5 overall
                    if label in preds_top5:
                        correct_top5 += 1
                        class_correct_top5[label] += 1
                    
                    # Track by species pairs
                    if label in classes_with:
                        total_with += 1
                        if pred_top1 == label:
                            correct_top1_with += 1
                        if label in preds_top5:
                            correct_top5_with += 1
                            
                    elif label in classes_without:
                        total_without += 1
                        if pred_top1 == label:
                            correct_top1_without += 1
                        if label in preds_top5:
                            correct_top5_without += 1
        
        # Calculate accuracies
        test_loss = running_loss / len(self.test_loader)
        
        # Top-1 accuracies
        top1_overall = 100. * correct_top1 / total if total > 0 else 0
        top1_with = 100. * correct_top1_with / total_with if total_with > 0 else 0
        top1_without = 100. * correct_top1_without / total_without if total_without > 0 else 0
        
        # Top-5 accuracies
        top5_overall = 100. * correct_top5 / total if total > 0 else 0
        top5_with = 100. * correct_top5_with / total_with if total_with > 0 else 0
        top5_without = 100. * correct_top5_without / total_without if total_without > 0 else 0
        
        # Per-class accuracies
        per_class_top1 = {cls: 100. * class_correct_top1[cls] / class_total[cls] 
                         for cls in class_total}
        per_class_top5 = {cls: 100. * class_correct_top5[cls] / class_total[cls] 
                         for cls in class_total}
        
        results = {
            'test_loss': test_loss,
            # Top-1
            'top1_overall': top1_overall,
            'top1_with_pairs': top1_with,
            'top1_without_pairs': top1_without,
            'top1_gap': top1_with - top1_without,
            # Top-5
            'top5_overall': top5_overall,
            'top5_with_pairs': top5_with,
            'top5_without_pairs': top5_without,
            'top5_gap': top5_with - top5_without,
            # Samples
            'total_samples': total,
            'samples_with_pairs': total_with,
            'samples_without_pairs': total_without,
            # Per-class
            'per_class_top1': per_class_top1,
            'per_class_top5': per_class_top5,
            'class_total': class_total,
        }
        
        return results
    
    def print_results(self, results):
        """Print comprehensive results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        print(f"\n{'Metric':<35} {'Overall':>12} {'With Pairs':>14} {'Without Pairs':>16} {'Gap':>10}")
        print("-"*80)
        print(f"{'Top-1 Accuracy (%)':<35} {results['top1_overall']:>12.2f} {results['top1_with_pairs']:>14.2f} {results['top1_without_pairs']:>16.2f} {results['top1_gap']:>10.2f}")
        print(f"{'Top-5 Accuracy (%)':<35} {results['top5_overall']:>12.2f} {results['top5_with_pairs']:>14.2f} {results['top5_without_pairs']:>16.2f} {results['top5_gap']:>10.2f}")
        print("-"*80)
        print(f"{'Test Loss':<35} {results['test_loss']:>12.4f}")
        print("="*80)
        
        # Sample distribution
        print(f"\nSample Distribution:")
        print(f"   Total samples: {results['total_samples']}")
        if results['samples_with_pairs'] > 0:
            print(f"   With pairs: {results['samples_with_pairs']} ({100.*results['samples_with_pairs']/results['total_samples']:.1f}%)")
        if results['samples_without_pairs'] > 0:
            print(f"   Without pairs: {results['samples_without_pairs']} ({100.*results['samples_without_pairs']/results['total_samples']:.1f}%)")
        
        print("="*80 + "\n")
    
    def plot_results(self, results, save_path='test_results.png'):
        """Generate comprehensive visualization with crystal clear aesthetics"""
        # Set clean style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.linewidth'] = 1.5
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        
        # Crystal clear color scheme - high contrast
        color_palette = {
            'primary': '#1f77b4',       # Strong blue
            'secondary': '#d62728',     # Strong red
            'success': '#2ca02c',       # Strong green
            'warning': '#ff7f0e',       # Strong orange
            'light_blue': '#7EC8E3',
            'light_green': '#98D8AA',
            'light_orange': '#FFBB5C'
        }
        
        # Plot 1: Top-1 vs Top-5 Overall
        ax = axes[0, 0]
        categories = ['Top-1', 'Top-5']
        values = [results['top1_overall'], results['top5_overall']]
        colors_plot1 = [color_palette['primary'], color_palette['secondary']]
        
        bars = ax.bar(categories, values, color=colors_plot1, 
                     edgecolor='black', linewidth=2, width=0.6)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=14)
        ax.set_title('Overall Top-1 vs Top-5 Accuracy', fontweight='bold', fontsize=15, pad=20)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=12)
        
        # Add value labels - clean, no background boxes
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontweight='bold', fontsize=13)
        
        # Plot 2: Top-1 Breakdown
        ax = axes[0, 1]
        categories = ['Overall', 'With Pairs', 'Without Pairs']
        values = [results['top1_overall'], results['top1_with_pairs'], results['top1_without_pairs']]
        colors_plot2 = [color_palette['light_blue'], color_palette['light_green'], color_palette['light_orange']]
        
        bars = ax.bar(categories, values, color=colors_plot2,
                     edgecolor='black', linewidth=2, width=0.65)
        ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold', fontsize=14)
        ax.set_title('Top-1 Accuracy Breakdown', fontweight='bold', fontsize=15, pad=20)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=12)
        
        # Gap annotation - moved to right
        gap_color = color_palette['success'] if results['top1_gap'] <= 15 else color_palette['secondary']
        ax.text(0.80, 0.95, f'Gap: {results["top1_gap"]:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=13, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor=gap_color,
                        edgecolor='black', linewidth=2))
        
        # Plot 3: Top-5 Breakdown
        ax = axes[1, 0]
        values = [results['top5_overall'], results['top5_with_pairs'], results['top5_without_pairs']]
        
        bars = ax.bar(categories, values, color=colors_plot2,
                     edgecolor='black', linewidth=2, width=0.65)
        ax.set_ylabel('Top-5 Accuracy (%)', fontweight='bold', fontsize=14)
        ax.set_title('Top-5 Accuracy Breakdown', fontweight='bold', fontsize=15, pad=20)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=12)
        
        # Gap annotation - moved to right
        gap_color = color_palette['success'] if results['top5_gap'] <= 10 else color_palette['secondary']
        ax.text(0.80, 0.95, f'Gap: {results["top5_gap"]:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=13, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.8',
                        facecolor=gap_color,
                        edgecolor='black', linewidth=2))
        
        # Plot 4: Gap Comparison
        ax = axes[1, 1]
        categories = ['Top-1 Gap', 'Top-5 Gap']
        values = [results['top1_gap'], results['top5_gap']]
        colors_plot4 = [color_palette['primary'], color_palette['secondary']]
        
        bars = ax.bar(categories, values, color=colors_plot4,
                     edgecolor='black', linewidth=2, width=0.6)
        ax.set_ylabel('Performance Gap (%)', fontweight='bold', fontsize=14)
        ax.set_title('Domain Shift Gap Analysis', fontweight='bold', fontsize=15, pad=20)
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=12)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=13)
        
        # Overall title
        plt.suptitle('Comprehensive Test Results: Top-1 & Top-5 Analysis',
                    fontsize=17, fontweight='bold', y=0.995)
        
        # Adjust spacing
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Results visualization saved: {save_path}")
        plt.close()
    
    def save_results(self, results, save_path='test_results.json'):
        """Save results to JSON"""
        # Convert numpy types to Python types
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    str(k): float(v) if isinstance(v, (np.number, float)) else int(v)
                    for k, v in value.items()
                }
            elif isinstance(value, (np.number, float)):
                serializable_results[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                serializable_results[key] = int(value)
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Results saved to JSON: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test trained model with comprehensive metrics')
    parser.add_argument('--checkpoint', type=str, default='baseline1_optimized_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--img_size', type=int, default=320,
                       help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_prefix', type=str, default='test',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MODEL TESTING - Comprehensive Evaluation")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Checkpoint not found: {args.checkpoint}")
        print(f"   Please provide a valid checkpoint path")
        return
    
    print(f"\nLoading checkpoint: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'best_accuracy' in checkpoint:
        print(f"   Best accuracy: {checkpoint['best_accuracy']:.2f}%")
    
    # Get dataset info
    dataset_info = checkpoint.get('dataset_info', {})
    num_classes = dataset_info.get('num_classes', 100)
    
    print(f"\nDataset Info:")
    print(f"   Number of classes: {num_classes}")
    if 'classes_with_pairs' in dataset_info:
        print(f"   Classes with pairs: {len(dataset_info['classes_with_pairs'])}")
    if 'classes_without_pairs' in dataset_info:
        print(f"   Classes without pairs: {len(dataset_info['classes_without_pairs'])}")
    
    # Create dataloaders
    print(f"\nLoading test data...")
    _, test_loader, _, dataset_info = create_improved_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_albumentations=True
    )
    
    if test_loader is None or len(test_loader) == 0:
        print(f"\n❌ Error: No test data found!")
        print(f"   Please ensure test.txt exists in {args.data_dir}/list/")
        return
    
    # Create model
    print(f"\nCreating model...")
    model = create_improved_model(
        model_name='convnext_small',
        num_classes=num_classes,
        pretrained=False,
        dropout=0.2
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"✅ Model loaded successfully!")
    
    # Create tester
    tester = ModelTester(model, test_loader, device, dataset_info)
    
    # Run evaluation
    results = tester.evaluate_comprehensive()
    
    # Print results
    tester.print_results(results)
    
    # Generate plots
    plot_path = f"{args.output_prefix}_results.png"
    tester.plot_results(results, save_path=plot_path)
    
    # Save JSON
    json_path = f"{args.output_prefix}_results.json"
    tester.save_results(results, save_path=json_path)
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"   - {plot_path} (visualization)")
    print(f"   - {json_path} (detailed metrics)")
    print("\n")


if __name__ == "__main__":
    main()