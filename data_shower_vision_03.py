import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV（尝试多种编码）
file_path = 'D:/data/data_arrangement/Output/亚组分析_按性别分层_20260530_161449.csv'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='gbk')

# 解析 OR 和 95%CI
def parse_or_ci(or_str, ci_str):
    or_val = float(or_str) if pd.notna(or_str) else np.nan
    ci_low, ci_high = np.nan, np.nan
    if pd.notna(ci_str):
        parts = ci_str.replace('(', '').replace(')', '').split('~')
        if len(parts) == 2:
            ci_low = float(parts[0])
            ci_high = float(parts[1])
    return or_val, ci_low, ci_high

df[['OR', 'CI_low', 'CI_high']] = df.apply(
    lambda row: parse_or_ci(row['OR值'], row['95%CI']), axis=1, result_type='expand'
)
df['P值'] = pd.to_numeric(df['P值'], errors='coerce')

# 按亚组拆分
subgroups = df['亚组'].unique()
# 定义颜色映射
colors = {'男性': '#5B9BD5', '女性': '#ED7D31'}

# 创建图形和两个子图，增大面板间距
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.6)  # 增大面板间距，避免纵轴重叠

# 自定义图例句柄
font_prop = FontProperties(family='SimHei', size=10)
legend_handles = []

# 遍历亚组绘制森林图
for ax, subgroup in zip([ax1, ax2], subgroups):
    sub_df = df[df['亚组'] == subgroup].copy()
    # 变量名处理：长度>10自动换行
    vars_order = sub_df['变量'].tolist()
    wrapped_labels = []
    for label in vars_order:
        if len(label) > 10:
            mid = len(label) // 2
            wrapped = label[:mid] + '\n' + label[mid:]
            wrapped_labels.append(wrapped)
        else:
            wrapped_labels.append(label)
    
    y_pos = range(len(vars_order))
    
    # 绘制森林图
    color = colors.get(subgroup, 'gray')
    ax.errorbar(sub_df['OR'], y_pos, 
                xerr=[sub_df['OR'] - sub_df['CI_low'], sub_df['CI_high'] - sub_df['OR']],
                fmt='o', color=color, ecolor=color, elinewidth=2, capsize=4, markersize=6,
                label=subgroup)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1)
    
    # 设置 y 轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrapped_labels, fontsize=10)
    ax.set_xlabel('Odds Ratio (95% CI)')
    ax.set_title(f'{subgroup}亚组', fontsize=12, pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    # 动态计算文本位置（避免与线条重叠）
    max_ci = sub_df['CI_high'].max()
    text_x = max_ci * 1.5
    ax.set_xlim(left=0, right=text_x * 1.2)
    
    # 添加 OR 和 P 值文本
    for i, (_, row) in enumerate(sub_df.iterrows()):
        or_text = f'{row["OR"]:.2f} ({row["CI_low"]:.2f}-{row["CI_high"]:.2f})'
        p_text = f'p={row["P值"]:.4f}' if pd.notna(row['P值']) else ''
        ax.text(text_x, i, f'{or_text}  {p_text}', va='center', fontsize=9)
    
    # 收集图例句柄
    legend_handles.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=12, label=subgroup)
    )

# 添加全局图例（方形标记，消除乱码）
fig.legend(handles=legend_handles, prop=font_prop, frameon=True, edgecolor='black',
           fancybox=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

# 底部预留空间（如需添加注释可在此调整）
plt.subplots_adjust(bottom=0.12, top=0.88)

plt.show()
