# experimentation/plot_tensorboard.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def merge_tensorboard_logs(log_dir):
    """
    Merge all TensorBoard event files, with later logs overwriting earlier ones for the same step.
    """
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    event_files.sort()  # Sort by filename (timestamp-based)
    
    merged_data = {}  # step -> {metric -> value}
    
    for f in event_files:
        print(f"Loading: {os.path.basename(f)}")
        try:
            ea = EventAccumulator(f)
            ea.Reload()
            
            tags = ea.Tags().get('scalars', [])
            
            for tag in tags:
                events = ea.Scalars(tag)
                for e in events:
                    step = e.step
                    if step not in merged_data:
                        merged_data[step] = {}
                    merged_data[step][tag] = e.value
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    # Convert to long-form DataFrame
    records = []
    for step in sorted(merged_data.keys()):
        for metric, value in merged_data[step].items():
            records.append({'step': step, 'metric': metric, 'value': value})
    
    if not records:
        return None
    return pd.DataFrame(records)

def plot_dqn_training(log_dir, output_dir):
    """
    Generate plots from merged TensorBoard logs.
    """
    df = merge_tensorboard_logs(log_dir)
    if df is None:
        print("No TensorBoard data found.")
        return
    
    sns.set_theme(style="whitegrid")
    
    # Get unique metrics
    metrics = df['metric'].unique()
    print(f"\nAvailable metrics: {list(metrics)}")
    
    # Training phase transitions
    phase_changes = {
        2500: {
            'label': 'Phase 2 (Ep 2500)\nBS:256, LR:1.5e-4, Len:3000\nEps:0.95 (d=0.995)\nRew: W-0.5, A-1.0, S-0.0025',
            'color': 'orange'
        },
        3133: {
            'label': 'Phase 3 (Ep 3133)\nBS:128, LR:2e-4\nEps:0.65 (d=0.999)\nRew: W-0.75, A-1.25',
            'color': 'red'
        }
    }
    
    # Plot each metric
    for metric in metrics:
        metric_df = df[df['metric'] == metric].sort_values('step')
        
        if metric_df.empty:
            continue
        
        plt.figure(figsize=(12, 6))
        plt.plot(metric_df['step'], metric_df['value'], linewidth=2, alpha=0.8)
        
        # Add phase change annotations (only if x-axis is episodes)
        if metric_df['step'].max() > 100:  # Likely episodes not timesteps
            for ep, info in phase_changes.items():
                if ep <= metric_df['step'].max():
                    plt.axvline(x=ep, color=info['color'], linestyle='--', alpha=0.7, linewidth=2)
                    
                    # Position label dynamically based on plot height
                    y_pos = plt.gca().get_ylim()[1] * 0.6 if ep == 2500 else plt.gca().get_ylim()[1] * 0.4
                    plt.text(ep + 50, y_pos, info['label'], 
                             color=info['color'], fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                       edgecolor=info['color'], alpha=0.9))
        
        # Clean up metric name for title
        title = metric.replace('/', ' - ').replace('_', ' ').title()
        plt.title(f"DQN Training: {title}", fontsize=14, fontweight='bold')
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel(metric.split('/')[-1].replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with sanitized filename
        filename = metric.replace('/', '_').replace(' ', '_').lower() + '.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

if __name__ == "__main__":
    log_dir = "runs"
    output_dir = "experimentation/plots"
    
    print("=== Merging and Plotting DQN TensorBoard Logs ===\n")
    plot_dqn_training(log_dir, output_dir)
    print("\nAll plots saved to experimentation/plots/")
