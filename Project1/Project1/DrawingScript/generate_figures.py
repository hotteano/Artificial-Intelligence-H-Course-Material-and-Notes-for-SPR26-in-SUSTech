#!/usr/bin/env python3
"""
Generate figures for IEM Project Report
使用matplotlib生成学术论文风格的图表
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体和学术风格
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# 创建输出目录
output_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(output_dir, exist_ok=True)

# ==================== 数据 (来自 all_methods_summary.csv) ====================

# Table A - Heuristic Old Machine
heuristic_cases = ['map1', 'map2', 'map3', 'map4', 'map5']
heuristic_runtime_old = [17.5589, 21.6625, 23.4944, 55.2672, 38.4275]
heuristic_runtime_new = [16.6776, 19.8945, 22.1203, 52.5990, 37.8946]
heuristic_score_old = [451.031200, 36025.039400, 36219.086800, 5626.073400, 3082.512000]
heuristic_nodes = [475, 36742, 36742, 7115, 3454]
heuristic_edges = [13289, 49248, 49248, 103689, 32140]
heuristic_prob = [0.01, 0.01, 0.01, 0.5, 0.5]  # 近似传播概率

# Table B - Evolutionary Old Machine
evolutionary_cases = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7']
evolutionary_runtime_old = [115.4429, 67.2518, 122.1948, 17.3971, 19.3814, 50.4984, 58.0731]
evolutionary_runtime_new = [136.3996, 51.7604, 87.7575, 11.9673, 13.1667, 38.3057, 39.6616]
evolutionary_score_old = [449.114400, 13693.213600, 13665.447000, 3098.405800, 3038.697200, 2488.492800, 2366.854400]
evolutionary_nodes = [475, 13984, 13984, 3454, 3454, 3454, 3454]
evolutionary_prob = [0.01, 0.01, 0.01, 0.5, 0.5, 0.7, 0.7]  # 近似传播概率

# Table C3 - 跨机器对比（Heuristic）
cross_cases_h = ['map1', 'map2', 'map3', 'map4', 'map5']
old_machine_h = [17.5589, 21.6625, 23.4944, 55.2672, 38.4275]
new_machine_h = [16.6776, 19.8945, 22.1203, 52.5990, 37.8946]

# Table C4 - 跨机器对比（Evolutionary）
cross_cases_e = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7']
old_machine_e = [115.4429, 67.2518, 122.1948, 17.3971, 19.3814, 50.4984, 58.0731]
new_machine_e = [136.3996, 51.7604, 87.7575, 11.9673, 13.1667, 38.3057, 39.6616]

# Table D - Evaluator 多阶段数据 (Old Machine)
simulations = [5000, 10000, 20000, 30000, 40000, 50000, 60000]
map1_time = [3.5174, 8.1301, 16.7364, 16.9350, 22.3746, 28.7068, 33.6887]
map1_score = [423.9388, 423.6748, 423.5561, 423.5149, 423.7382, 423.7666, 423.6112]
map2_time = [4.4105, 12.0964, 23.7645, 22.8284, 31.8855, 38.8280, 46.3011]
map2_score = [35562.3430, 35561.4352, 35561.2809, 35560.4609, 35560.6718, 35560.9353, 35560.9804]


# ==================== 图1: 启发式方法 Runtime Scaling ====================
def plot_heuristic_scaling():
    """启发式方法在不同cases上的runtime scaling，突出高概率图的影响"""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(heuristic_cases))
    colors = ['#4472C4' if p < 0.05 else '#C55A11' for p in heuristic_prob]

    bars = ax.bar(x, heuristic_runtime_old, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Heuristic Method: Runtime Scaling by Test Case (Old Machine)')
    ax.set_xticks(x)
    ax.set_xticklabels(heuristic_cases)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, heuristic_runtime_old)):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
        # 添加节点数和概率标注
        ax.annotate(f'n={heuristic_nodes[i]}\np≈{heuristic_prob[i]}',
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, color='gray')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', edgecolor='black', label='Low probability (p<0.01)'),
        Patch(facecolor='#C55A11', edgecolor='black', label='Medium probability (p≈0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_heuristic_scaling.pdf'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig1_heuristic_scaling.png'),
                bbox_inches='tight', dpi=300)
    print("Saved: fig1_heuristic_scaling")
    plt.close()


# ==================== 图2: 跨机器加速比 ====================
def plot_speedup_comparison():
    """新机器相对于旧机器的加速比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Heuristic 加速比
    speedup_h = [new/old for new, old in zip(new_machine_h, old_machine_h)]
    colors_h = ['#70AD47' if s < 1 else '#C55A11' for s in speedup_h]

    bars1 = ax1.barh(cross_cases_h, speedup_h, color=colors_h, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='Baseline (1x)')
    ax1.set_xlabel('Speedup Ratio (New/Old)')
    ax1.set_title('Heuristic: Cross-Machine Speedup')
    ax1.set_xlim(0.8, 1.2)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, speedup_h)):
        ax1.text(val + 0.02, i, f'{val:.2f}x', va='center', fontsize=9)

    # Evolutionary 加速比
    speedup_e = [new/old for new, old in zip(new_machine_e, old_machine_e)]
    colors_e = ['#70AD47' if s < 1 else '#C55A11' for s in speedup_e]

    bars2 = ax2.barh(cross_cases_e, speedup_e, color=colors_e, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='Baseline (1x)')
    ax2.set_xlabel('Speedup Ratio (New/Old)')
    ax2.set_title('Evolutionary: Cross-Machine Speedup')
    ax2.set_xlim(0, 1.5)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars2, speedup_e)):
        ax2.text(val + 0.03, i, f'{val:.2f}x', va='center', fontsize=9)

    # 添加图例说明颜色
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#70AD47', edgecolor='black', label='Faster on New'),
                       Patch(facecolor='#C55A11', edgecolor='black', label='Slower on New')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, 'fig2_speedup_comparison.pdf'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig2_speedup_comparison.png'),
                bbox_inches='tight', dpi=300)
    print("Saved: fig2_speedup_comparison")
    plt.close()


# ==================== 图3: Evaluator 多阶段收敛 ====================
def plot_evaluator_convergence():
    """Evaluator 模拟次数 vs 时间和分数稳定性"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 运行时间
    ax1.plot(simulations, map1_time, 'o-', color='#4472C4', linewidth=2,
             markersize=6, label='map1')
    ax1.plot(simulations, map2_time, 's-', color='#ED7D31', linewidth=2,
             markersize=6, label='map2')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Evaluator: Runtime vs Simulation Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 添加趋势线
    z1 = np.polyfit(simulations, map1_time, 1)
    p1 = np.poly1d(z1)
    ax1.plot(simulations, p1(simulations), '--', color='#4472C4', alpha=0.5,
             label=f'map1 trend (slope={z1[0]*1000:.3f}ms/sim)')

    z2 = np.polyfit(simulations, map2_time, 1)
    p2 = np.poly1d(z2)
    ax1.plot(simulations, p2(simulations), '--', color='#ED7D31', alpha=0.5,
             label=f'map2 trend (slope={z2[0]*1000:.3f}ms/sim)')

    ax1.legend(loc='upper left', fontsize=9)

    # 右图: 分数稳定性 (相对变化)
    map1_relative = [(s - map1_score[0]) / map1_score[0] * 100 for s in map1_score]
    map2_relative = [(s - map2_score[0]) / map2_score[0] * 100 for s in map2_score]

    ax2.plot(simulations, map1_relative, 'o-', color='#4472C4', linewidth=2,
             markersize=6, label='map1')
    ax2.plot(simulations, map2_relative, 's-', color='#ED7D31', linewidth=2,
             markersize=6, label='map2')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Relative Score Change (%)')
    ax2.set_title('Evaluator: Score Stability vs Simulation Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 添加方差标注
    map1_std = np.std(map1_relative)
    map2_std = np.std(map2_relative)
    ax2.text(0.98, 0.98, f'map1 σ = {map1_std:.4f}%\nmap2 σ = {map2_std:.4f}%',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_evaluator_convergence.pdf'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig3_evaluator_convergence.png'),
                bbox_inches='tight', dpi=300)
    print("Saved: fig3_evaluator_convergence")
    plt.close()


# ==================== 图4: 进化算法 High-Probability 影响分析 ====================
def plot_evolutionary_highprob_analysis():
    """分析高概率图对进化算法的影响（map4/map5 vs map6/map7）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cases = ['map4', 'map5', 'map6', 'map7']
    # 使用新机器数据
    runtimes_new = [11.9673, 13.1667, 38.3057, 39.6616]
    scores_new = [3077.316200, 3053.795000, 2448.146000, 2419.661200]
    runtimes_old = [17.3971, 19.3814, 50.4984, 58.0731]
    scores_old = [3098.405800, 3038.697200, 2488.492800, 2366.854400]
    probs = [0.5, 0.5, 0.7, 0.7]

    x = np.arange(len(cases))
    width = 0.35
    colors = ['#4472C4' if p <= 0.5 else '#C55A11' for p in probs]

    # 左图: Runtime (New Machine)
    bars1 = ax1.bar(x - width/2, runtimes_old, width, label='Old Machine',
                    color='#5B9BD5', edgecolor='black', linewidth=0.5, alpha=0.8)
    bars2 = ax1.bar(x + width/2, runtimes_new, width, label='New Machine',
                    color='#70AD47', edgecolor='black', linewidth=0.5, alpha=0.8)

    ax1.set_xlabel('Test Case')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Evolutionary: Impact of Propagation Probability on Runtime')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加分组标注
    ax1.axvspan(-0.4, 1.4, alpha=0.1, color='blue', label='p=0.5')
    ax1.axvspan(1.6, 3.4, alpha=0.1, color='red', label='p=0.7')
    ax1.legend(loc='upper left')

    # 右图: Score (New Machine)
    bars3 = ax2.bar(x - width/2, scores_old, width, label='Old Machine',
                    color='#5B9BD5', edgecolor='black', linewidth=0.5, alpha=0.8)
    bars4 = ax2.bar(x + width/2, scores_new, width, label='New Machine',
                    color='#70AD47', edgecolor='black', linewidth=0.5, alpha=0.8)

    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Final Score')
    ax2.set_title('Evolutionary: Impact of Propagation Probability on Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加下降百分比标注 (New Machine)
    avg_p50_score = (scores_new[0] + scores_new[1]) / 2
    avg_p70_score = (scores_new[2] + scores_new[3]) / 2
    decline = (avg_p50_score - avg_p70_score) / avg_p50_score * 100
    ax2.text(0.98, 0.98, f'Score decline (p=0.5→0.7):\n{decline:.1f}%',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_evolutionary_highprob.pdf'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig4_evolutionary_highprob.png'),
                bbox_inches='tight', dpi=300)
    print("Saved: fig4_evolutionary_highprob")
    plt.close()


# ==================== 图5: Evolutionary 扩展测试 ====================
def plot_evolutionary_extended():
    """Evolutionary 在 map6/map7 的扩展测试"""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(evolutionary_cases))
    width = 0.35

    bars1 = ax.bar(x - width/2, old_machine_e, width, label='Old Machine',
                   color='#5B9BD5', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, new_machine_e, width, label='New Machine',
                   color='#70AD47', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Evolutionary Method: Cross-Machine Performance (All Cases)')
    ax.set_xticks(x)
    ax.set_xticklabels(evolutionary_cases)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_evolutionary_extended.pdf'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig5_evolutionary_extended.png'),
                bbox_inches='tight', dpi=300)
    print("Saved: fig5_evolutionary_extended")
    plt.close()


# ==================== 主函数 ====================
if __name__ == '__main__':
    print("Generating figures for IEM Project Report...")
    print(f"Output directory: {output_dir}")
    print()

    plot_heuristic_scaling()
    plot_speedup_comparison()
    plot_evaluator_convergence()
    plot_evolutionary_highprob_analysis()
    plot_evolutionary_extended()

    print()
    print("All figures generated successfully!")
    print("Files saved as both PDF (for LaTeX) and PNG (for preview)")
