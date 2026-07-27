import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tensorboard_logs():
    tb_dir = os.path.join(LOG_DIR, "PPO_1")
    event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    event_files.sort()
    
    merged_data = {}
    
    for f in event_files:
        print(f"Loading TB log: {os.path.basename(f)}")
        try:
            ea = EventAccumulator(f)
            ea.Reload()
            
            tags = ea.Tags()['scalars']
            
            for tag in ['train/entropy_loss', 'train/approx_kl', 'rollout/ep_rew_mean']:
                if tag in tags:
                    events = ea.Scalars(tag)
                    for e in events:
                        step = e.step
                        if step not in merged_data:
                            merged_data[step] = {}
                        merged_data[step][tag] = e.value
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    records = []
    steps = sorted(merged_data.keys())
    for step in steps:
        for metric, value in merged_data[step].items():
            records.append({'step': step, 'metric': metric, 'value': value})
            
    if not records:
        return None
    return pd.DataFrame(records)

def plot_training_dynamics():
    df_tb = load_tensorboard_logs()
    if df_tb is None:
        print("No TensorBoard data found.")
        return
    
    # 1. Episode Reward from TensorBoard (complete data)
    df_rew = df_tb[df_tb['metric'] == 'rollout/ep_rew_mean']
    if not df_rew.empty:
        # Apply smoothing
        window = 50
        df_rew = df_rew.sort_values('step')
        df_rew['value_smoothed'] = df_rew['value'].rolling(window=window, center=True).mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_rew['step'], df_rew['value_smoothed'], label=f'Episode Reward (MA {window})', linewidth=2)
        
        # Annotations with proper positioning
        # Reward Reshaping at 4,914,080
        plt.axvline(x=4914080, color='g', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(4914080 + 200000, plt.gca().get_ylim()[1] * 0.5, 
                 'Reward Reshaped\n(-0.5 Wall)', color='g', fontsize=10, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='g', alpha=0.9))
        
        # LR Change at 6,921,120
        plt.axvline(x=6921120, color='r', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(6921120 + 200000, plt.gca().get_ylim()[1] * 0.7, 
                 'LR Reduced\n(1e-4)', color='r', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='r', alpha=0.9))
        
        plt.title("Training Progress: Episode Reward", fontsize=14, fontweight='bold')
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Episode Reward", fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "training_reward_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    # 2. Entropy
    df_ent = df_tb[df_tb['metric'] == 'train/entropy_loss']
    if not df_ent.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df_ent['step'], df_ent['value'], linewidth=2)
        plt.title("Policy Entropy (Exploration)", fontsize=14, fontweight='bold')
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Entropy Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "training_entropy.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    # 3. KL Divergence
    df_kl = df_tb[df_tb['metric'] == 'train/approx_kl']
    if not df_kl.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df_kl['step'], df_kl['value'], linewidth=2)
        plt.title("Approx KL Divergence", fontsize=14, fontweight='bold')
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("KL", fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "training_kl.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    plot_training_dynamics()
