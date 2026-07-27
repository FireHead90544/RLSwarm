import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(experiment_name):
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(experiment_name) and f.endswith(".csv")]
    if not files:
        print(f"No data found for {experiment_name}")
        return None
    files.sort()
    latest = files[-1]
    print(f"Loading {latest}...")
    return pd.read_csv(os.path.join(RESULTS_DIR, latest))

def add_value_labels(ax, fmt='{:.2f}', inside=False):
    """Add labels to bars in a bar chart."""
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height) or abs(height) < 0.01: continue  # Skip zero or near-zero bars
        
        if inside:
            # Place inside bar, near top
            y_pos = height * 0.5
            va = 'center'
            color = 'white'
            weight = 'bold'
        else:
            # Place above bar with white background
            y_pos = height
            va = 'bottom'
            color = 'black'
            weight = 'normal'
        
        ax.annotate(fmt.format(height),
                    (p.get_x() + p.get_width() / 2., y_pos),
                    ha='center', va=va,
                    xytext=(0, 8 if not inside else 0),
                    textcoords='offset points',
                    fontsize=9, color=color, weight=weight,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8) if not inside else None)

def plot_scalability():
    df = load_data("scalability_fixed")
    if df is None: return
    
    # 1. Net Reward vs Agents
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='num_agents', y='net_total_reward', errorbar='sd', palette="viridis", capsize=0.1)
    add_value_labels(ax)
    plt.title("Scalability (Fixed Map): Net Total Reward", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Agents", fontsize=12)
    plt.ylabel("Net Episode Reward", fontsize=12)
    plt.ylim(bottom=0, top=df['net_total_reward'].max() * 1.2)  # Extra space for labels
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_reward_total.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Agent Reward
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='num_agents', y='mean_agent_reward', errorbar='sd', palette="viridis", capsize=0.1)
    add_value_labels(ax)
    plt.title("Scalability (Fixed Map): Per-Agent Reward", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Agents", fontsize=12)
    plt.ylabel("Mean Reward per Agent", fontsize=12)
    plt.ylim(bottom=0, top=df['mean_agent_reward'].max() * 1.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_reward_per_agent.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Collisions
    df_melt = df.melt(id_vars=['num_agents'], value_vars=['total_wall_collisions', 'total_agent_collisions'], 
                      var_name='Collision Type', value_name='Count')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melt, x='num_agents', y='Count', hue='Collision Type', palette="rocket", capsize=0.1)
    add_value_labels(ax)
    plt.title("Scalability: Collision Analysis", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Agents", fontsize=12)
    plt.ylabel("Total Collisions per Episode", fontsize=12)
    plt.ylim(bottom=0, top=df_melt['Count'].max() * 1.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_collisions.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency():
    df = load_data("efficiency_benchmark")
    if df is None: return
    
    # Bar plot for clearer values
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x='policy_type', y='net_total_reward', errorbar='sd', palette="Set2", capsize=0.1)
    add_value_labels(ax)
    plt.title("Efficiency: Trained vs Random", fontsize=14, fontweight='bold')
    plt.ylabel("Net Episode Reward", fontsize=12)
    plt.xlabel("Policy", fontsize=12)
    # Set ylim to accommodate negative values
    ymin = df['net_total_reward'].min() * 1.2
    ymax = df['net_total_reward'].max() * 1.2
    plt.ylim(bottom=ymin, top=ymax)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.savefig(os.path.join(OUTPUT_DIR, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Food Collected
    plt.figure(figsize=(8, 6))
    df_melt = df.melt(id_vars=['policy_type'], value_vars=['total_food_touched', 'total_food_deposited'],
                      var_name='Metric', value_name='Count')
    ax = sns.barplot(data=df_melt, x='policy_type', y='Count', hue='Metric', palette="Set2", capsize=0.1)
    add_value_labels(ax)
    plt.title("Foraging Performance", fontsize=14, fontweight='bold')
    plt.xlabel("Policy", fontsize=12)
    plt.ylabel("Food Count", fontsize=12)
    plt.ylim(bottom=0, top=df_melt['Count'].max() * 1.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "efficiency_food.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_human_vs_ai():
    df = load_data("human_vs_ai")
    if df is None: return
    
    # Fix Labels: "ai" -> "trained_best"
    df['policy_type'] = df['policy_type'].replace({'ai': 'trained_best'})
    
    # Comparison Bar Plot
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x='policy_type', y='net_total_reward', errorbar='sd', palette="coolwarm", capsize=0.1)
    add_value_labels(ax)
    plt.title("Human vs AI: Total Reward", fontsize=14, fontweight='bold')
    plt.ylabel("Total Reward", fontsize=12)
    plt.xlabel("Policy Type", fontsize=12)
    plt.ylim(bottom=0, top=df['net_total_reward'].max() * 1.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "human_vs_ai_reward.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Efficiency Score
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x='policy_type', y='efficiency_score', errorbar='sd', palette="coolwarm", capsize=0.1)
    add_value_labels(ax, fmt='{:.4f}')
    plt.title("Human vs AI: Efficiency Score", fontsize=14, fontweight='bold')
    plt.ylabel("Reward per Step", fontsize=12)
    plt.xlabel("Policy Type", fontsize=12)
    plt.ylim(bottom=0, top=df['efficiency_score'].max() * 1.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "human_vs_ai_efficiency.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Collisions Comparison
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x='policy_type', y='total_wall_collisions', errorbar='sd', palette="Reds", capsize=0.1)
    add_value_labels(ax)
    plt.title("Human vs AI: Wall Collisions", fontsize=14, fontweight='bold')
    plt.ylabel("Total Wall Collisions", fontsize=12)
    plt.xlabel("Policy Type", fontsize=12)
    plt.ylim(bottom=0, top=df['total_wall_collisions'].max() * 1.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "human_vs_ai_collisions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Food Interaction Comparison
    plt.figure(figsize=(8, 6))
    df_food = df.melt(id_vars=['policy_type'], value_vars=['total_food_touched', 'total_food_deposited'],
                      var_name='Metric', value_name='Count')
    ax = sns.barplot(data=df_food, x='policy_type', y='Count', hue='Metric', palette="Greens", capsize=0.1)
    add_value_labels(ax)
    plt.title("Human vs AI: Food Interaction", fontsize=14, fontweight='bold')
    plt.ylabel("Food Count", fontsize=12)
    plt.xlabel("Policy Type", fontsize=12)
    plt.ylim(bottom=0, top=df_food['Count'].max() * 1.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "human_vs_ai_food.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    plot_scalability()
    plot_efficiency()
    plot_human_vs_ai()
