import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 备选字体
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV 文件
file_path = 'D:/data/data_arrangement/Output/用药记忆情况分析_20260530_161449.csv'
df = pd.read_csv(file_path, encoding='gbk')


# --- 数据解析 ---
# 1. 交叉表数据
cross_df = df[df['分析类型'].str.contains('交叉表分析', na=False)].copy()
# 转换数值列
cross_df['行百分比(%)'] = pd.to_numeric(cross_df['行百分比(%)'], errors='coerce')
cross_df['例数'] = pd.to_numeric(cross_df['例数'], errors='coerce')
# 提取卡方值和 P 值（取第一行非空值）
chi2 = cross_df['卡方值'].dropna().iloc[0] if not cross_df['卡方值'].dropna().empty else 'N/A'
p_val = cross_df['P值'].dropna().iloc[0] if not cross_df['P值'].dropna().empty else 'N/A'

# 2. 多因素 Logistic 回归数据
logit_df = df[df['分析类型'].str.contains('多因素Logistic回归', na=False)].copy()
# 排除 AIC 行
aic_row = logit_df[logit_df['用药记忆情况'].str.contains('AIC', na=False)]
logit_df = logit_df[~logit_df['用药记忆情况'].str.contains('AIC', na=False)]
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

logit_df[['OR', 'CI_low', 'CI_high']] = logit_df.apply(
    lambda row: parse_or_ci(row['OR值'], row['95%CI']), axis=1, result_type='expand'
)
logit_df['P值'] = pd.to_numeric(logit_df['P值'], errors='coerce')
# Logistic 回归的变量名实际在“服药习惯”列，AIC 行则在“用药记忆情况”列
logit_df['变量'] = logit_df['服药习惯'].fillna(logit_df['用药记忆情况']).str.strip()


# 提取 AIC 值
aic_val = aic_row['P值'].values[0] if not aic_row.empty else '598.8'

# --- 开始绘图 ---
fig = plt.figure(figsize=(14, 6))
# 使用 gridspec 控制子图宽度和间距
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.5)  # wspace 增大面板间距
ax1 = fig.add_subplot(gs[0, 0])  # 交叉表图
ax2 = fig.add_subplot(gs[0, 1])  # 森林图

# ========== 左图：交叉表堆叠条形图 ==========
# 准备数据：按用药记忆情况分组，行百分比堆叠
categories = cross_df['用药记忆情况'].unique()
habits = cross_df['服药习惯'].unique()
# 构建矩阵：行=用药记忆，列=服药习惯
data_matrix = cross_df.pivot(index='用药记忆情况', columns='服药习惯', values='行百分比(%)')
# 确保顺序
data_matrix = data_matrix.reindex(index=categories, columns=habits)

# 绘制堆叠条形图
bottom = np.zeros(len(categories))
colors = ['#5B9BD5', '#ED7D31']  # 空腹、餐后颜色
for i, habit in enumerate(habits):
    values = data_matrix[habit].fillna(0).values
    ax1.bar(categories, values, bottom=bottom, color=colors[i], label=habit, width=0.6)
    bottom += values

# 添加百分比标签
for j, cat in enumerate(categories):
    cum = 0
    for i, habit in enumerate(habits):
        val = data_matrix.loc[cat, habit] if pd.notna(data_matrix.loc[cat, habit]) else 0
        if val > 0:
            ax1.text(j, cum + val/2, f'{val:.1f}%', ha='center', va='center', fontsize=9)
            cum += val

# 标题和轴标签
ax1.set_title('交叉表分析：用药记忆 vs 服药习惯', fontsize=12, pad=15)
ax1.set_ylabel('行百分比 (%)')
ax1.set_ylim(0, 110)
# 添加统计注释
ax1.text(0.5, -0.25, f'卡方值(Yates) = {chi2}\nP = {p_val}', 
         transform=ax1.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# 自定义图例：使用 Line2D 方形标记 + FontProperties
font_prop = FontProperties(family='SimHei', size=10)  # 指定字体
legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], markersize=12, label=habits[0]),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markersize=12, label=habits[1])
]
ax1.legend(handles=legend_handles, prop=font_prop, frameon=True, edgecolor='black', 
           fancybox=False, loc='upper right')

# ========== 右图：森林图 ==========
# 变量顺序：从上到下按原数据顺序（或自定义）
vars_order = logit_df['变量'].tolist()
# 长度超过10的标签自动换行（在中点处插入换行符）
wrapped_labels = []
for label in vars_order:
    if len(label) > 10:
        mid = len(label) // 2
        wrapped = label[:mid] + '\n' + label[mid:]
        wrapped_labels.append(wrapped)
    else:
        wrapped_labels.append(label)
y_pos = range(len(vars_order))

# 绘制 OR 点估计和置信区间
ax2.errorbar(logit_df['OR'], y_pos, xerr=[logit_df['OR'] - logit_df['CI_low'], 
                                          logit_df['CI_high'] - logit_df['OR']],
             fmt='o', color='black', ecolor='gray', elinewidth=2, capsize=4, markersize=6)
# 参考线 OR=1
ax2.axvline(x=1, color='red', linestyle='--', linewidth=1)
# 设置 y 轴标签
ax2.set_yticks(y_pos)
ax2.set_yticklabels(wrapped_labels, fontsize=10)
ax2.set_xlabel('Odds Ratio (95% CI)')
ax2.set_title('多因素 Logistic 回归森林图', fontsize=12, pad=15)
# 添加网格
ax2.grid(axis='x', linestyle=':', alpha=0.6)

# 在右侧添加 OR 和 P 值文本
# 动态计算文本放置的 x 坐标（取所有 CI 上限的最大值，并留出边距）
max_ci = logit_df['CI_high'].max()
text_x = max_ci * 1.4  # 根据实际数据调整倍率，确保文本在右侧空白区

for i, (_, row) in enumerate(logit_df.iterrows()):
    or_text = f'{row["OR"]:.2f} ({row["CI_low"]:.2f}-{row["CI_high"]:.2f})'
    p_text = f'p={row["P值"]:.4f}' if pd.notna(row['P值']) else ''
    ax2.text(text_x, i, f'{or_text}  {p_text}', va='center', fontsize=9)

# 扩展 x 轴右边界，为文本留出空间
ax2.set_xlim(left=0, right=text_x * 1.1)

# 底部标注 AIC（预留足够空间）
fig.text(0.5, 0.02, f'AIC = {aic_val}', ha='center', fontsize=11, 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))

# 调整整体布局，底部留够空间
plt.subplots_adjust(bottom=0.15, top=0.9, left=0.08, right=0.95)
# 显示图形
plt.show()
