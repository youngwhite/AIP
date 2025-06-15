import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_avg_best_acc_per_layer_with_zero(csv_paths):
    all_best_accs = []

    for path in csv_paths:
        df = pd.read_csv(path)
        # 每层取最大 acc
        best_accs = [df[f'acc_layer_{i}'].max() for i in range(12)]
        all_best_accs.append(best_accs)

    # 求每层平均准确率
    all_best_accs = np.array(all_best_accs)  # shape: (num_folds, 12)
    avg_best_accs = all_best_accs.mean(axis=0)

    # 插入起点 0.0，表示“起始状态”
    avg_best_accs_with_zero = np.insert(avg_best_accs, 0, 0.0)

    # 可视化
    plt.figure(figsize=(8, 5))
    plt.plot(range(13), avg_best_accs_with_zero, marker='o', linewidth=2, label='Average Best Acc')
    plt.title('Average Best Accuracy per Layer (Across Folds)')
    plt.xlabel('Layer Index')
    plt.ylabel('Accuracy')
    plt.xlim(0, 12)     # 横坐标从 0 到 12（因为你现在有13个点）
    plt.ylim(0, 1.0)    # 纵坐标从 0 到 1.0（准确率）    
    # plt.xticks(range(13))
    # plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return avg_best_accs_with_zero

if __name__ == "__main__":
    # 示例 CSV 文件路径
    csv_files = [
        'ADP-esc50-ast-fold1.csv',
        'ADP-esc50-ast-fold2.csv',
        'ADP-esc50-ast-fold3.csv',
        'ADP-esc50-ast-fold4.csv',
        'ADP-esc50-ast-fold5.csv'
    ]

    avg_best_accs = plot_avg_best_acc_per_layer_with_zero(csv_files)
    print("Average Best Accuracies per Layer:", avg_best_accs)
    