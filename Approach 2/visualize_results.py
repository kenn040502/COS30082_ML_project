import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(results_file='improved_baseline1_best_results.json'):
    """Load training results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def plot_training_curves(history, save_path='plots'):
    """
    Plot training loss and accuracy over epochs
    """
    Path(save_path).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress - Improved Baseline-1', fontsize=16, fontweight='bold')
    
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
    
    # Add phase markers if visible
    if len(epochs) >= 25:
        axes[0, 0].axvline(x=25, color='gray', linestyle='--', alpha=0.5, label='Phase 2 Start')
        axes[0, 0].text(25, max(train_loss)*0.9, 'Phase 2\nStarts', 
                       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Training Accuracy
    axes[0, 1].plot(epochs, train_acc, linewidth=2, color='#3498db', label='Training Accuracy')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 100])
    
    # Plot 3: Validation Accuracy (sampled every 5 epochs)
    # Need to reconstruct epoch indices for validation
    val_epochs = list(range(5, len(epochs)+1, 5))[:len(val_acc)]
    if len(val_epochs) != len(val_acc):
        # Adjust if mismatch
        val_epochs = [epochs[i] for i in range(4, len(epochs), 5)][:len(val_acc)]
    
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
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract data
    val_epochs = list(range(5, len(history['train_loss'])+1, 5))[:len(history['val_acc_with_pairs'])]
    acc_with = history['val_acc_with_pairs']
    acc_without = history['val_acc_without_pairs']
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    val_epochs = list(range(5, len(history['train_loss'])+1, 5))[:len(history['val_acc_overall'])]
    top1_acc = history['val_acc_overall']
    top5_acc = history['val_acc_top5']
    
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
                   xytext=(val_epochs[-1]-10, top1_acc[-1]-8),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.annotate(f'Final Top-5: {top5_acc[-1]:.2f}%', 
                   xy=(val_epochs[-1], top5_acc[-1]),
                   xytext=(val_epochs[-1]-10, top5_acc[-1]+3),
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
    accuracies = [
        final_results['top1_acc'],
        final_results['top5_acc'],
        final_results['acc_with_pairs'],
        final_results['acc_without_pairs']
    ]
    colors = ['#3498db', '#e74c3c', '#27ae60', '#e67e22']
    
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.1f}%', ha='center', va='bottom', 
               fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Final Test Results - Improved Baseline-1', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add horizontal line at 80% (your target)
    ax.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target: 80%')
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
    val_epochs = list(range(5, len(epochs)+1, 5))[:len(history['val_acc_overall'])]
    val_acc = history['val_acc_overall']
    
    # Plot
    ax.plot(epochs, train_acc, linewidth=2, color='#3498db', label='Training Accuracy', alpha=0.7)
    ax.plot(val_epochs, val_acc, linewidth=3, marker='o', markersize=7,
            color='#e74c3c', label='Validation Accuracy')
    
    # Phase divider (assuming Phase 1 = 25 epochs)
    if len(epochs) >= 25:
        ax.axvline(x=25, color='black', linestyle='--', linewidth=2, label='Phase 1 → Phase 2')
        
        # Add shaded regions
        ax.axvspan(0, 25, alpha=0.1, color='blue', label='Phase 1: Head Only')
        ax.axvspan(25, len(epochs), alpha=0.1, color='orange', label='Phase 2: Full Fine-tune')
        
        # Add text annotations
        ax.text(12.5, 95, 'Phase 1\n(Backbone Frozen)', ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.text((25 + len(epochs))/2, 95, 'Phase 2\n(Full Fine-tuning)', ha='center', fontsize=12,
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
    Create a comprehensive summary figure for your report
    """
    Path(save_path).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    history = results['history']
    final_results = results['results']
    
    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = list(range(1, len(history['train_loss']) + 1))
    ax1.plot(epochs, history['train_loss'], linewidth=2, color='#e74c3c')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_acc'], linewidth=2, color='#3498db')
    ax2.set_title('Training Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['learning_rates'], linewidth=2, color='#9b59b6')
    ax3.set_title('Learning Rate Schedule', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top-1 vs Top-5
    ax4 = fig.add_subplot(gs[1, :2])
    val_epochs = list(range(5, len(epochs)+1, 5))[:len(history['val_acc_overall'])]
    ax4.plot(val_epochs, history['val_acc_overall'], linewidth=3, marker='o', 
            markersize=7, color='#3498db', label='Top-1')
    ax4.plot(val_epochs, history['val_acc_top5'], linewidth=3, marker='D',
            markersize=7, color='#e74c3c', label='Top-5')
    ax4.set_title('Validation Accuracy: Top-1 vs Top-5', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim([0, 100])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Final Results Bar Chart
    ax5 = fig.add_subplot(gs[1, 2])
    categories = ['Top-1', 'Top-5']
    values = [final_results['top1_acc'], final_results['top5_acc']]
    bars = ax5.bar(categories, values, color=['#3498db', '#e74c3c'], alpha=0.8)
    for bar, val in zip(bars, values):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax5.set_title('Final Test Accuracy', fontweight='bold')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_ylim([0, 100])
    ax5.grid(True, axis='y', alpha=0.3)
    
    # 6. With vs Without Pairs
    ax6 = fig.add_subplot(gs[2, :])
    ax6.plot(val_epochs, history['val_acc_with_pairs'], linewidth=3, marker='o',
            markersize=7, color='#27ae60', label='With Field Pairs (60 species)')
    ax6.plot(val_epochs, history['val_acc_without_pairs'], linewidth=3, marker='s',
            markersize=7, color='#e67e22', label='Without Field Pairs (40 species)')
    ax6.fill_between(val_epochs, history['val_acc_with_pairs'], 
                    history['val_acc_without_pairs'], alpha=0.2, color='gray')
    ax6.set_title('Performance by Species Type: Cross-Domain Challenge', 
                 fontweight='bold', fontsize=12)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_ylim([0, 100])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Improved Baseline-1: Complete Training Summary', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(f'{save_path}/complete_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}/complete_summary.png")
    plt.close()


def generate_all_plots(results_file='improved_baseline1_best_results.json'):
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
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check if results file exists
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'improved_baseline1_best_results.json'
    
    if not Path(results_file).exists():
        print(f"Error: {results_file} not found!")
        print(f"   Please run training first: python improved_baseline1_train.py")
        sys.exit(1)
    
    # Generate all plots
    generate_all_plots(results_file)