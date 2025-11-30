import os
import sys

# Usage method:
# python visualize_results.py [results_file.json]

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style - use fallback if specific style not available
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
        print("Using default matplotlib style")

sns.set_palette("husl")


def load_results(results_file='baseline1_best_results.json'):
    """Load training results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Error: {results_file} not found!")
        print(f"   Make sure training has completed first.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        print(f"   File may be corrupted. Try re-running training.")
        sys.exit(1)


def plot_training_curves(history, save_path='plots'):
    """
    Plot training loss and accuracy over epochs
    """
    Path(save_path).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress - Baseline-1', fontsize=16, fontweight='bold')
    
    # Extract data
    epochs = list(range(1, len(history['train_loss']) + 1))
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc_overall']
    learning_rates = history['learning_rates']
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, train_loss, linewidth=2, color='#e74c3c', label='Training Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Add phase markers if visible (assumes Phase 1 = 13 epochs)
    if len(epochs) >= 13:
        axes[0, 0].axvline(x=13, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].text(13, max(train_loss)*0.9, 'Phase 2\nStarts', 
                       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Training Accuracy
    axes[0, 1].plot(epochs, train_acc, linewidth=2, color='#3498db', label='Training Accuracy')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 100])
    
    # Plot 3: Validation Accuracy - FIXED!
    if val_acc and len(val_acc) > 0:
        # Check if validation was done every epoch or every N epochs
        if len(val_acc) == len(epochs):
            # Validation every epoch - use epochs directly
            val_epochs = epochs
        else:
            # Validation every N epochs (subsample)
            val_epochs = list(range(5, len(epochs)+1, 5))[:len(val_acc)]
            if len(val_epochs) != len(val_acc):
                # Adjust if mismatch
                val_epochs = [epochs[i] for i in range(4, min(len(epochs), len(val_acc)*5), 5)][:len(val_acc)]
        
        axes[1, 0].plot(val_epochs, val_acc, linewidth=2, marker='o', 
                       markersize=6, color='#2ecc71', label='Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1, 0].set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 100])
        
        # Annotate best accuracy
        if val_acc:
            best_acc = max(val_acc)
            best_epoch = val_epochs[val_acc.index(best_acc)]
            axes[1, 0].annotate(f'Best: {best_acc:.2f}%', 
                               xy=(best_epoch, best_acc),
                               xytext=(best_epoch+5, best_acc-5),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2),
                               fontsize=11, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    else:
        axes[1, 0].text(0.5, 0.5, 'No validation data', 
                       transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 0].set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
    
    # Plot 4: Learning Rate Schedule
    axes[1, 1].plot(epochs, learning_rates, linewidth=2, color='#9b59b6', label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}/training_curves.png")
    plt.close()


def plot_with_without_pairs(history, save_path='plots'):
    """
    Plot performance comparison: Species WITH vs WITHOUT field pairs
    """
    Path(save_path).mkdir(exist_ok=True)
    
    # Check if we have the data
    if not history.get('val_acc_with_pairs') or not history.get('val_acc_without_pairs'):
        print("Skipping with/without pairs plot - data not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract data - FIXED!
    epochs = list(range(1, len(history['train_loss']) + 1))
    acc_with = history['val_acc_with_pairs']
    acc_without = history['val_acc_without_pairs']
    
    # Use epochs directly if validation was done every epoch
    if len(acc_with) == len(epochs):
        val_epochs = epochs
    else:
        val_epochs = list(range(5, len(epochs)+1, 5))[:len(acc_with)]
    
    # Ensure lengths match
    min_len = min(len(val_epochs), len(acc_with), len(acc_without))
    val_epochs = val_epochs[:min_len]
    acc_with = acc_with[:min_len]
    acc_without = acc_without[:min_len]
    
    # Plot both lines
    ax.plot(val_epochs, acc_with, linewidth=3, marker='o', markersize=8,
            color='#27ae60', label='Species WITH Field Pairs (60 species)')
    ax.plot(val_epochs, acc_without, linewidth=3, marker='s', markersize=8,
            color='#e67e22', label='Species WITHOUT Field Pairs (40 species)')
    
    # Fill area between
    ax.fill_between(val_epochs, acc_with, acc_without, alpha=0.2, color='gray')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Cross-Domain Challenge: Performance Gap Between Species Types', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0, 100])
    
    # Add text showing final gap
    if acc_with and acc_without:
        final_gap = acc_with[-1] - acc_without[-1]
        ax.text(0.5, 0.5, f'Performance Gap: {final_gap:.1f}%', 
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/with_vs_without_pairs.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}/with_vs_without_pairs.png")
    plt.close()


def plot_top1_vs_top5(history, save_path='plots'):
    """
    Plot Top-1 vs Top-5 accuracy comparison
    """
    Path(save_path).mkdir(exist_ok=True)
    
    # Check if we have the data
    if not history.get('val_acc_overall') or not history.get('val_acc_top5'):
        print("Skipping Top-1 vs Top-5 plot - data not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data - FIXED!
    epochs = list(range(1, len(history['train_loss']) + 1))
    top1_acc = history['val_acc_overall']
    top5_acc = history['val_acc_top5']
    
    # Use epochs directly if validation was done every epoch
    if len(top1_acc) == len(epochs):
        val_epochs = epochs
    else:
        val_epochs = list(range(5, len(epochs)+1, 5))[:len(top1_acc)]
    
    # Ensure lengths match
    min_len = min(len(val_epochs), len(top1_acc), len(top5_acc))
    val_epochs = val_epochs[:min_len]
    top1_acc = top1_acc[:min_len]
    top5_acc = top5_acc[:min_len]
    
    # Plot both
    ax.plot(val_epochs, top1_acc, linewidth=3, marker='o', markersize=8,
            color='#3498db', label='Top-1 Accuracy')
    ax.plot(val_epochs, top5_acc, linewidth=3, marker='D', markersize=8,
            color='#e74c3c', label='Top-5 Accuracy')
    
    # Fill area between
    ax.fill_between(val_epochs, top1_acc, top5_acc, alpha=0.15, color='purple')
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Top-1 vs Top-5 Accuracy Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 100])
    
    # Annotate final values
    if top1_acc and top5_acc:
        ax.annotate(f'Final Top-1: {top1_acc[-1]:.2f}%', 
                   xy=(val_epochs[-1], top1_acc[-1]),
                   xytext=(val_epochs[-1]-3, top1_acc[-1]-8),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.annotate(f'Final Top-5: {top5_acc[-1]:.2f}%', 
                   xy=(val_epochs[-1], top5_acc[-1]),
                   xytext=(val_epochs[-1]-3, top5_acc[-1]+3),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/top1_vs_top5.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}/top1_vs_top5.png")
    plt.close()


def plot_final_comparison(results, save_path='plots'):
    """
    Bar chart comparing final results
    """
    Path(save_path).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract final results
    final_results = results['results']
    
    categories = ['Overall\nTop-1', 'Overall\nTop-5', 
                  'With Pairs\n(60 species)', 'Without Pairs\n(40 species)']
    values = [
        final_results['top1_acc'],
        final_results['top5_acc'],
        final_results['acc_with_pairs'],
        final_results['acc_without_pairs']
    ]
    colors = ['#3498db', '#e74c3c', '#27ae60', '#e67e22']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Final Test Set Performance', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add baseline reference line
    ax.axhline(y=65, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Basic Baseline (~65%)')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/final_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}/final_comparison.png")
    plt.close()


def plot_phase_comparison(history, save_path='plots'):
    """
    Show training/validation accuracy across Phase 1 and Phase 2
    """
    Path(save_path).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    epochs = list(range(1, len(history['train_acc']) + 1))
    train_acc = history['train_acc']
    
    # Get validation data if available
    val_acc = history.get('val_acc_overall', [])
    if len(val_acc) == len(epochs):
        val_epochs = epochs
    else:
        val_epochs = list(range(5, len(epochs)+1, 5))[:len(val_acc)]
    
    # Plot
    ax.plot(epochs, train_acc, linewidth=2, color='#3498db', label='Training Accuracy', alpha=0.7)
    if val_acc:
        ax.plot(val_epochs, val_acc, linewidth=3, marker='o', markersize=7,
                color='#e74c3c', label='Validation Accuracy')
    
    # Phase divider (Phase 1 = 13 epochs)
    phase1_epochs = 13
    if len(epochs) >= phase1_epochs:
        ax.axvline(x=phase1_epochs, color='black', linestyle='--', linewidth=2, label='Phase 1 → Phase 2')
        
        # Add shaded regions
        ax.axvspan(0, phase1_epochs, alpha=0.1, color='blue', label='Phase 1: Head Only')
        ax.axvspan(phase1_epochs, len(epochs), alpha=0.1, color='orange', label='Phase 2: Full Fine-tune')
        
        # Add text annotations
        ax.text(phase1_epochs/2, 95, 'Phase 1\n(Backbone Frozen)', ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.text((phase1_epochs + len(epochs))/2, 95, 'Phase 2\n(Full Fine-tuning)', ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Two-Phase Training Strategy', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/phase_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}/phase_comparison.png")
    plt.close()


def create_summary_figure(results, save_path='plots'):
    """
    Create a comprehensive summary figure - SIMPLIFIED
    """
    Path(save_path).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    history = results['history']
    final_results = results['results']
    
    # Extract data
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # 1. Training curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], linewidth=2, color='#e74c3c', label='Training Loss')
    if 'val_loss' in history:
        val_loss = history['val_loss']
        if len(val_loss) == len(epochs):
            ax1.plot(epochs, val_loss, linewidth=2, color='#3498db', label='Validation Loss')
        else:
            val_epochs = list(range(5, len(epochs)+1, 5))[:len(val_loss)]
            ax1.plot(val_epochs, val_loss, linewidth=2, color='#3498db', label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training accuracy (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_acc'], linewidth=2, color='#3498db', label='Training')
    if history.get('val_acc_overall'):
        val_acc = history['val_acc_overall']
        if len(val_acc) == len(epochs):
            ax2.plot(epochs, val_acc, linewidth=2, color='#e74c3c', label='Validation')
        else:
            val_epochs = list(range(5, len(epochs)+1, 5))[:len(val_acc)]
            ax2.plot(val_epochs, val_acc, linewidth=2, color='#e74c3c', label='Validation')
    ax2.set_title('Training and Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. With vs Without pairs (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    if history.get('val_acc_with_pairs') and history.get('val_acc_without_pairs'):
        acc_with = history['val_acc_with_pairs']
        acc_without = history['val_acc_without_pairs']
        
        if len(acc_with) == len(epochs):
            val_epochs = epochs
        else:
            val_epochs = list(range(5, len(epochs)+1, 5))[:len(acc_with)]
        
        min_len = min(len(val_epochs), len(acc_with), len(acc_without))
        ax3.plot(val_epochs[:min_len], acc_with[:min_len], linewidth=3, marker='o',
                markersize=7, color='#27ae60', label='With Field Pairs (60 species)')
        ax3.plot(val_epochs[:min_len], acc_without[:min_len], linewidth=3, marker='s',
                markersize=7, color='#e67e22', label='Without Field Pairs (40 species)')
        ax3.fill_between(val_epochs[:min_len], acc_with[:min_len], acc_without[:min_len], 
                        alpha=0.2, color='gray')
        ax3.set_title('Performance by Species Type', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim([0, 100])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Species pair data not available', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('Performance by Species Type', fontweight='bold', fontsize=12)
    
    # 4. Final results bar chart (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    categories = ['Top-1', 'Top-5', 'With\nPairs', 'Without\nPairs']
    values = [
        final_results['top1_acc'], 
        final_results['top5_acc'],
        final_results['acc_with_pairs'],
        final_results['acc_without_pairs']
    ]
    colors = ['#3498db', '#e74c3c', '#27ae60', '#e67e22']
    bars = ax4.bar(categories, values, color=colors, alpha=0.8)
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax4.set_title('Final Test Accuracy', fontweight='bold')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim([0, 100])
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Add overall title
    fig.suptitle('Baseline-1: Complete Training Summary', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(f'{save_path}/complete_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}/complete_summary.png")
    plt.close()


def generate_all_plots(results_file='baseline1_best_results.json'):
    """
    Generate all visualization plots
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*60 + "\n")
    
    # Load results
    print("Loading results...")
    results = load_results(results_file)
    history = results['history']
    
    # Handle missing keys for compatibility
    if 'epoch_times' not in history:
        print("    epoch_times not tracked (using simplified training script)")
        history['epoch_times'] = []
    
    # Validate required keys
    required_keys = ['train_loss', 'train_acc', 'learning_rates']
    missing_keys = [key for key in required_keys if key not in history]
    if missing_keys:
        print(f"❌ Error: Missing required keys in history: {missing_keys}")
        print(f"   Results file may be from incompatible training script")
        sys.exit(1)
    
    print(f"   Loaded {len(history['train_loss'])} epochs of training data")
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Generate all plots
    print("\n1. Creating training curves...")
    plot_training_curves(history)
    
    print("\n2. Creating with/without pairs comparison...")
    plot_with_without_pairs(history)
    
    print("\n3. Creating Top-1 vs Top-5 comparison...")
    plot_top1_vs_top5(history)
    
    print("\n4. Creating final results comparison...")
    plot_final_comparison(results)
    
    print("\n5. Creating phase comparison...")
    plot_phase_comparison(history)
    
    print("\n6. Creating complete summary figure...")
    create_summary_figure(results)
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nPlots saved in: ./plots/")
    print("\nGenerated files:")
    print("  1. training_curves.png")
    print("  2. with_vs_without_pairs.png")
    print("  3. top1_vs_top5.png")
    print("  4. final_comparison.png")
    print("  5. phase_comparison.png")
    print("  6. complete_summary.png")
    print("\nUse these plots in your assignment report!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Check if results file exists
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'baseline1_best_results.json'
    
    if not Path(results_file).exists():
        print(f"Error: {results_file} not found!")
        print(f"\nMake sure training has completed first:")
        print(f"   python baseline1_train.py")
        print(f"\n   Then run:")
        print(f"   python visualize_results.py")
        sys.exit(1)
    
    # Generate all plots
    generate_all_plots(results_file)