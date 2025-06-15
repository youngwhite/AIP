import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def plot_curve(para_list, vacc_list, savename):
    vaccs = [100*vacc for vacc in vacc_list]
    plt.figure()
    plt.plot(para_list, vaccs, marker='o')
    plt.xlabel("Parameter Number (M)")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Parameter Number vs. Average Accuracy")
    plt.xticks(range(0, 100, 10))
    plt.yticks(range(0, 101, 10))
    plt.grid(True)
    # 在坐标点旁边标记数值
    for x, y in zip(para_list, vaccs):
        # plt.text(x, y, f'{y:.1f}', fontsize=12, ha='left', va='top')
        offset_x, offset_y = -2.5, 3
        plt.text(x + offset_x, y + offset_y, f'{y:.2f}', fontsize=9, ha='center', va='center')

    plt.savefig(savename, dpi=300, bbox_inches='tight')  # PNG 格式，300 DPI
    plt.close()

def calculate_metrics(y_true: np.array, y_pred: np.array, target2label: dict, top_k=10):
    # 计算整体准确率
    acc = accuracy_score(y_true, y_pred)
    
    # 计算所有类别的分类指标
    targets = sorted(target2label.keys())  # 确保类别按顺序排列
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, labels=targets, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=targets)
    cm_df = pd.DataFrame(cm, index=[target2label[t] for t in targets], columns=[target2label[t] for t in targets])

    class_metrics = {
        target2label[target]: {
            'target': target,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1_score[i],
            'support': support[i]
        }
        for i, target in enumerate(targets)
    }

    return {
        "acc": acc,
        "cm_df": cm_df,
        "class_metrics": class_metrics
    }
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm_df, normalize=True, cmap="Blues", save_path=None, dpi=300): 
    # 归一化处理
    cm = cm_df.values
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = np.where(cm.sum(axis=1, keepdims=True) == 0, 0, cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100)
        fmt = ".2f"
    else:
        fmt = "d"

    # 获取类别信息
    labels = cm_df.index.tolist()
    num_classes = len(labels)

    # 自动调整图像大小（避免过大或过小）
    figsize = (min(20, max(8, num_classes * 0.3)), min(15, max(6, num_classes * 0.3)))
    plt.figure(figsize=figsize, dpi=dpi)

    # 自动调整字体大小
    tick_size = max(8, 200 // num_classes)
    annot_size = max(6, 300 // num_classes)
    
    # 如果类别太多，不显示数字（annot）
    show_annot = num_classes <= 50

    sns.set(font_scale=max(1, num_classes / 20))

    # 绘制热图（去掉色条 cbar=False）
    sns.heatmap(cm, annot=show_annot, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels,
                annot_kws={"size": annot_size}, linewidths=0.5, linecolor="gray", cbar=False)

    # 旋转 X 轴标签，避免重叠
    plt.xticks(rotation=90, ha="right", fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # 轴标签
    plt.xlabel("Predicted Label", fontsize=max(12, tick_size))
    plt.ylabel("Actual Label", fontsize=max(12, tick_size))
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=max(14, tick_size))

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Confusion Matrix saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    target2label = {0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "rabbit", 5: "tiger", 6: "lion", 7: "wolf", 8: "elephant", 9: "horse"}
    y_true = np.random.randint(0, 10, 100)  # 生成 100 个样本
    y_pred = np.random.randint(0, 10, 100)

    metrics = calculate_metrics(y_true, y_pred, target2label, top_k=2)  # 选择 F1-score 最高的 5 类

    # 打印整体准确率
    print("Accuracy:", metrics["acc"])

    # 打印混淆矩阵
    print("\nConfusion Matrix:")
    print(metrics["cm_df"])

    # 打印每个类别的分类指标
    print("\nClass-wise Metrics:")
    for label, metric in metrics['class_metrics'].items():
        print(f"{label}: {metric}")

    # 画混淆矩阵（使用类别名称）
    plot_confusion_matrix(metrics["cm_df"], normalize=False, cmap="Blues")
