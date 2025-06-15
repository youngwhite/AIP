import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter

def plot_search(csv_file: str, save_name: str = None, acc_threshold: float = 0.93):
    plt.rcParams["font.family"] = "DejaVu Serif"

    df = pd.read_csv(csv_file)
    df["params"] = df["params"] / 1e6
    index = df.index

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(index[0], df['params'].iloc[0],
               marker='o', facecolors='none', edgecolors='red', s=120,
               label='Start Point', zorder=5)

    # ===== 阶段识别（顺序推进）=====
    stage_type = []
    for _, row in df.iterrows():
        if row['head_pruning_ratio'] == 0 and row['pruning_ratio'] == 0:
            current_stage = "layer"
        if row['head_pruning_ratio'] != 0 and row['pruning_ratio'] == 0:
            current_stage = "head"
        if row['pruning_ratio'] != 0:
            current_stage = "ffn"
        stage_type.append(current_stage)
    df['stage'] = stage_type

    # ===== 染色 + 横轴上方注释 =====
    def highlight_region(condition, color, label):
        idx = df.index[condition]
        for k, g in groupby(enumerate(idx), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) > 0:
                left = group[0] - 0.5
                right = group[-1] + 0.5
                center = (left + right) / 2
                ax.axvspan(left, right, facecolor=color, alpha=0.35)

                # 横轴上方文字注释
                y_max = df["params"].max()
                offset = 0.05 * y_max
                ax.text(center, y_max + offset, label,
                        ha='center', va='bottom', fontsize=10,
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # 三个阶段染色和注释
    highlight_region(df["stage"] == "layer", "#c6dbef", "Layer Pruning Stage")
    highlight_region(df["stage"] == "head", "#fdd49e", "Head Pruning Stage")
    highlight_region(df["stage"] == "ffn", "#fc9272", "FFN Pruning Stage")

    # ===== 主图线条与关键点 =====
    layer_index = df["stage"] == "layer"
    ax.plot(index[layer_index], df['params'][layer_index], marker='o', linestyle='--',
            color='gray', alpha=0.5, label='Search Points')

    head_index = df["stage"] == "head"
    ax.plot(index[head_index], df['params'][head_index], marker='o', linestyle='--',
            color='gray', alpha=0.5)

    ffn_index = df["stage"] == "ffn"
    ax.plot(index[ffn_index], df['params'][ffn_index], marker='o', linestyle='--',
            color='gray', alpha=0.5)

    # ===== 选中点（当前最佳） =====
    accepted = (df['acc'] >= acc_threshold) & (df['acc'] != df['acc'].iloc[0])

    if accepted.any():
        ax.scatter(df.loc[accepted].index, df.loc[accepted, 'params'],
                marker='x', color='blue', label = "Accepted (Acc > Thres.)")
        minimal_index = df.loc[accepted, 'params'].idxmin()
        ax.scatter(minimal_index, df.loc[minimal_index, 'params'],
                   marker='o', s=120, facecolors='none', edgecolors='green',
                   label='Decision Point', zorder=10)

    # ===== 精度文本标注 =====
    y_offset = 0.015 * max(df['params'])
    for i in df.index:
        row = df.loc[i]
        label = f"{100 * row['acc']:.2f}%"
        ax.text(i + 0.5, row['params'] + y_offset,
                label, ha='center', fontsize=10, color='blue')

    # ===== 样式设置 =====
    fold_id = df["fold"].iloc[0] if "fold" in df.columns else "?"
    ax.set_xlabel("Search Trial (No.)", fontsize=12)
    ax.set_ylabel("Parameter Count (M)", fontsize=12)
    # ax.set_title(f"Minimal Parameters Search (Fold {fold_id})", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True)

    # ===== legend 去除染色标签，只保留关键点类别 =====
    handles, labels = ax.get_legend_handles_labels()
    stage_labels = {"Layer Pruning", "Head Pruning", "FFN Pruning"}
    uniq = [(h, l) for h, l in zip(handles, labels) if l not in stage_labels]
    if uniq:
        ax.legend(*zip(*uniq), fontsize=12)

    # 获取当前最大值
    y_max = df["params"].max()
    # y_min = df["params"].min()
    # 扩大上边界 10%，下边界不动
    ax.set_ylim(0, y_max * 1.1)

    plt.tight_layout()
    if save_name:
        plt.savefig(f"{save_name}.svg", bbox_inches='tight')
        print(f"✅ 图像已保存至：{save_name}.svg")
    else:
        plt.show()

if __name__ == "__main__":
    csv_file = 'results/AIP-esc50-ast-fold_1.csv'
    plot_search(csv_file, save_name='AIP-esc50-ast-fold_1', acc_threshold=0.93)
