import matplotlib.pyplot as plt
import pandas as pd


def plot_bars(df, metric_name, colors, figsize=(12, 8), ylim=None, title_fontsize=30,
              xlabel_fontsize=27, ylabel_fontsize=24, tick_fontsize_y=18, tick_fontsize_x=22,
              text_fontsize=17):
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    # 筛选出非缺失值的数据
    non_missing_df = df[df[metric_name].notnull()]
    bars = ax.bar(non_missing_df['model'], non_missing_df[metric_name], color=colors[:len(non_missing_df)])
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(metric_name + ' Large_Intestine', fontsize=title_fontsize)
    ax.set_xlabel('Model', fontsize=xlabel_fontsize)
    ax.set_ylabel(metric_name, fontsize=ylabel_fontsize)
    ax.tick_params(axis='y', which='major', labelsize=tick_fontsize_y)
    ax.tick_params(axis='x', which='major', labelsize=tick_fontsize_x)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 3),
                ha='center', va='bottom', fontsize=text_fontsize)

    plt.tight_layout()
    plt.show()


# 假设你的数据在一个DataFrame中，列名包括'model', 'ri', 'ari', 'nmi', 'fmi'等
data = {
    'model': ['scETM', 'harmony', 'lda', 'sc_VI', 'seurat', 'pca'],
    'ARI': [0.7237,0.6225,0.0523,0.0359,0.0366,0.0469],
    'RI': [None,None,0.73033,0.771388,0.77881,0.779233],
    'NMI': [0.7924, 0.7613, 0.1318, 0.1029,0.1029,0.1296],
    'MACRO - F1': [None,None,0.066503,0.055652,0.054283,0.055942],  # 添加macro - f1数据，包含空值
    'MICRO - F1': [None,None,0.184211,0.153801,0.145468,0.150146],   # 添加micro - f1数据，包含空值
    'FMI': [None,None,0.209769,0.168144,0.163996,0.174228]
}
df = pd.DataFrame(data)

# 定义颜色
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000']

# 对每一列数据（除了'model'列）绘制条形图
for metric in df.columns[1:]:
    plot_bars(df, metric, colors)
