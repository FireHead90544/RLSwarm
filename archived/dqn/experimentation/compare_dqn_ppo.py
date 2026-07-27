# experimentation/compare_dqn_ppo.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_ppo_data():
    """Load PPO evaluation data from neural_swarm_ppo project."""
    ppo_path = "../neural_swarm_ppo/experiments/results/efficiency_benchmark_20251129_161706.csv"
    
    if not os.path.exists(ppo_path):
        print(f"Warning: PPO data not found at {ppo_path}")
        return None
    
    df = pd.read_csv(ppo_path)
    # Filter only trained policy
    df_trained = df[df['policy_type'] == 'trained']
    return df_trained

def compare_algorithms(dqn_results_path, output_dir):
    """
    Compare DQN and PPO performance.
    """
    # Load DQN data
    df_dqn = pd.read_csv(dqn_results_path)
    
    # Load PPO data
    df_ppo = load_ppo_data()
    
    if df_ppo is None:
        print("Skipping PPO comparison (data not available)")
        return
    
    # Prepare comparison data
    dqn_reward = df_dqn['total_reward'].values
    ppo_reward = df_ppo['net_total_reward'].values
    
    comparison_data = pd.DataFrame({
        'Algorithm': ['DQN'] * len(dqn_reward) + ['PPO'] * len(ppo_reward),
        'Total Reward': np.concatenate([dqn_reward, ppo_reward])
    })
    
    # Plot comparison
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(data=comparison_data, x='Algorithm', y='Total Reward', 
                     palette=['#e74c3c', '#3498db'], capsize=0.1, errorbar='sd')
    
    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and abs(height) >= 0.01:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 8),
                        textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor='none', alpha=0.8))
    
    plt.title("DQN vs PPO: Performance Comparison (5 Agents)", 
              fontsize=14, fontweight='bold')
    plt.ylabel("Total Episode Reward", fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    ymin = min(comparison_data['Total Reward'].min() * 1.1, 0)
    ymax = comparison_data['Total Reward'].max() * 1.2
    plt.ylim(bottom=ymin, top=ymax)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'dqn_vs_ppo_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {save_path}")
    
    # Print statistics
    print("\n=== Performance Statistics ===")
    print(f"DQN: {dqn_reward.mean():.2f} ± {dqn_reward.std():.2f}")
    print(f"PPO: {ppo_reward.mean():.2f} ± {ppo_reward.std():.2f}")
    print(f"PPO Improvement: +{((ppo_reward.mean() - dqn_reward.mean()) / abs(dqn_reward.mean()) * 100):.1f}%")

if __name__ == "__main__":
    dqn_results = "experimentation/results/dqn_evaluation.csv"
    output_dir = "experimentation/plots"
    
    print("=== Comparing DQN vs PPO ===\n")
    compare_algorithms(dqn_results, output_dir)
