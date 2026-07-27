import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime
import os
import warnings
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
FILE_PATH = 'D:/data/Output/Enroll_Majorrevision-0726-289-含DM去餐后服用药物-无服药要求.xlsx'
Y_COL = '服药习惯_category'
SPARSE_THRESHOLD = 20

# ============================================================
# 全部变量列表（已按需求更新）
# ============================================================
ALL_REQUIRED_COLS = [
    '服药习惯_category',
    '性别_category',
    '年龄_Crit',
    '文化程度_category',
    '高血压病程_category',
    '服药种类_category',
    '合并慢性病种类_category',
    '用药知识掌握情况_category',
    '规律参与基层随访_category',
    '接受用药指导_category',
    '应餐前或餐时服药但未执行',
    '是否服用焦点药物',
    # '联合降脂治疗_category',
    # '美托洛尔治疗_category',
    # '抗血小板或抗凝_category',
    # '利尿剂_category',
    # 'ARBorACEI_category',
    # 'CCB_category',
]

X_COLS = [c for c in ALL_REQUIRED_COLS if c != Y_COL]

# ============================================================
# 排除变量配置
# ============================================================
UNIVARIATE_EXCLUDE = ['是否服用焦点药物',
    '应餐前或餐时服药但未执行',
    '是否服用焦点药物',                      
    # '美托洛尔治疗_category',
    # '抗血小板或抗凝_category',
    # '利尿剂_category',
    # 'ARBorACEI_category',
    # 'CCB_category',
    
    ]
EXHAUSTIVE_EXCLUDE = ['是否服用焦点药物',
    '应餐前或餐时服药但未执行',
    '是否服用焦点药物',                      
    # '美托洛尔治疗_category',
    # '抗血小板或抗凝_category',
    # '利尿剂_category',
    # 'ARBorACEI_category',
    # 'CCB_category',
    ]

# ============================================================
# 协变量配置（强制纳入单因素分析的变量，优先级高于排除变量）
# ============================================================
UNIVARIATE_COVARIATE_COLS = []   # 可随时增删变量

# ============================================================
# 协变量配置（强制纳入多因素分析的变量，优先级高于排除变量）
# ============================================================
MULTIVARIATE_COVARIATE_COLS = []   # 可随时增删变量


# ============================================================
# 年龄分组阈值
# ============================================================
AGE_THRESHOLD = 70



# ============================================================
# 基线特征表（Table 1）配置
# ============================================================
BASELINE_TABLE_ENABLED = True

# 基线表变量配置：[(原始列名, 显示名称, 变量类型)]
# 变量类型: 'categorical' → 显示为频数比（如 168:156）
#          'continuous' → 显示为均值±标准差（如 58.17±7.46）
BASELINE_VARIABLES = [
    ('性别_category',       '性别（男/女）',               'categorical'),
    ('年龄_Crit',           '年龄（岁）',                  'continuous'),   # 原始为连续变量
    ('文化程度_category',   '文化程度（初中及以下/高中及以上）', 'categorical'),
    ('高血压病程_category', '高血压病程（＜5年/≥5年）',     'categorical'),
    ('服药种类_category',   '服药种类（单药/联合）',        'categorical'),
    ('合并慢性病种类_category', '合并慢性病种类',           'categorical'),
    ('用药知识掌握情况_category', '用药知识掌握情况',               'categorical'),
    ('规律参与基层随访_category', '规律参与基层随访',       'categorical'),
    ('接受用药指导_category', '接受用药指导',               'categorical'),
    # ('联合降脂治疗_category', '联合降脂治疗',               'categorical'),
]



# ============================================================
# 中文标签映射
# ============================================================
LABEL_MAP = {
    'C(年龄_Crit)[T.70岁及以上]': '70岁及以上',
    'C(性别_category)[T.男]': '男性',
    'C(文化程度_category)[T.高中及以上]': '文化程度高',
    'C(高血压病程_category)[T.＜5年]': '病程短（＜5年）',
    'C(服药种类_category)[T.联合]': '联合用药',
    'C(合并慢性病种类_category)[T.仅1种慢性病]': '仅1种慢性病',
    'C(合并慢性病种类_category)[T.合并2种及以上慢性病]': '合并2种及以上慢性病',
    'C(用药知识掌握情况_category)[T.知晓]': '知晓用药知识',
    'C(规律参与基层随访_category)[T.是]': '规律参与基层随访',
    'C(接受用药指导_category)[T.是]': '接受用药指导',
    # 'C(联合降脂治疗_category)[T.是]': '联合降脂治疗',
    'C(用药知识掌握情况_category)[T.不知晓]': '不知晓',
    'C(应餐前或餐时服药但未执行)[T.未执行]': '应餐前或餐时服药但未执行（未执行）',
    'C(是否服用焦点药物)[T.是]': '服用焦点药物（是）',
    # 'C(美托洛尔治疗_category)[T.是]': '接受美托洛尔治疗',
    # 'C(ARBorACEI_category)[T.是]': '接受ARBorACEI治疗',
    # 'C(CCB_category)[T.是]': '接受CCB治疗',
    # 'C(抗血小板或抗凝_category)[T.是]': '接受抗血小板或抗凝治疗',
    # 'C(利尿剂_category)[T.是]': '接受利尿剂治疗',
}

# ============================================================
# 工具函数
# ============================================================
def parse_p_value(p_str):
    if pd.isna(p_str) or str(p_str).strip() in ['-', '']:
        return 1.0
    if str(p_str).strip() == '\uff1c0.001':  # ＜0.001
        return 0.0
    try:
        return float(str(p_str).strip())
    except ValueError:
        return 1.0

def filter_significant(df, p_col='P值', threshold=0.05):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['_P数值'] = df[p_col].apply(parse_p_value)
    sig = df[df['_P数值'] < threshold].drop(columns=['_P数值'])
    return sig

# ============================================================
# 样本量追踪器
# ============================================================
class SampleSizeTracker:
    def __init__(self, total_n, description='原始数据总样本量'):
        self.records = []
        self._add(total_n, description, '')
    
    def _add(self, n, desc, detail):
        prev_n = self.records[-1]['当前样本量'] if self.records else n
        self.records.append({
            '步骤序号': len(self.records) + 1,
            '步骤说明': desc,
            '当前样本量': n,
            '较上一步减少': prev_n - n if self.records else 0,
            '详细说明': detail
        })
    
    def add_step(self, n, desc, detail=''):
        self._add(n, desc, detail)
    
    def to_dataframe(self):
        df = pd.DataFrame(self.records)
        if len(df) > 1:
            first_n = df.loc[0, '当前样本量']
            df['累计排除'] = first_n - df['当前样本量']
        else:
            df['累计排除'] = 0
        return df

# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_preprocess(file_path):
    df = pd.read_excel(file_path, header=0)
    missing = [c for c in ALL_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'缺少列：{missing}')

    tracker = SampleSizeTracker(len(df), '原始数据总样本量')

    # 因变量编码
    df[Y_COL] = df[Y_COL].map({'空腹服药': 0, '餐后服药': 1}).astype(float)
    
    y_valid = df[Y_COL].notnull().sum()
    if y_valid < len(df):
        tracker.add_step(y_valid, '排除因变量无效值',
                         f'因变量{Y_COL}中存在无法映射的值，排除{len(df)-y_valid}条')
    else:
        tracker.add_step(y_valid, '因变量编码完成（全部有效）',
                         '所有样本的因变量值均可映射为空腹服药(0)或餐后服药(1)')

    # 处理缺失值文本标记
    for col in X_COLS:
        df[col] = df[col].replace('缺失值', np.nan)

    # ============================================================
    # 将年龄_Crit 从连续变量转为二分类变量
    # ============================================================
    print('\n--- 年龄变量转换 ---')
    df['年龄_Crit'] = pd.to_numeric(df['年龄_Crit'], errors='coerce')
    
    age_min = df['年龄_Crit'].min()
    age_max = df['年龄_Crit'].max()
    age_missing = df['年龄_Crit'].isnull().sum()
    print(f'  年龄范围：{age_min} ~ {age_max}，缺失：{age_missing}条')
        
    df['年龄_Crit'] = df['年龄_Crit'].apply(
        lambda x: '70岁及以上' if pd.notna(x) and x >= AGE_THRESHOLD else
                  ('70岁以下' if pd.notna(x) and x < AGE_THRESHOLD else np.nan)
    )
    
    age_counts = df['年龄_Crit'].value_counts(dropna=False)
    for cat in ['70岁以下', '70岁及以上']:
        if cat in age_counts.index:
            print(f'  {cat}：{int(age_counts[cat])}人')
    if age_missing > 0:
        print(f'  年龄缺失（已转为NaN）：{age_missing}条')
    
    df['年龄_Crit'] = df['年龄_Crit'].astype('category')
    
    tracker.add_step(
        len(df),
        f'年龄变量转换完成（连续→二分类，阈值{AGE_THRESHOLD}岁）',
        f'分为70岁以下和70岁及以上两组'
    )

    # ============================================================
    # 分类变量缺失值处理：将缺失作为单独的类别
    # ============================================================
    categorical_cols = [c for c in X_COLS if c != '年龄_Crit']
    all_categorical_cols = ['年龄_Crit'] + categorical_cols
    
    for col in all_categorical_cols:
        if df[col].isnull().any():
            n_miss = df[col].isnull().sum()
            df[col] = df[col].astype(object)
            df[col] = df[col].fillna('缺失')
            df[col] = df[col].astype('category')
            print(f'  📌 变量 {col} 存在 {n_miss} 个缺失值，已归为缺失类别')
    
    for col in categorical_cols:
        if col not in all_categorical_cols or df[col].dtype.name != 'category':
            df[col] = df[col].astype('category')

    print(f'\n原始样本量：{len(df)}')
    print(f'因变量有效样本量：{df[Y_COL].notnull().sum()}')

    # 缺失值诊断
    missing_report = []
    for col in ALL_REQUIRED_COLS:
        if col in all_categorical_cols:
            missing_report.append({'变量': col, '缺失数': 0, '缺失率%': 0.0,
                                   '说明': '缺失已归为[缺失]类别'})
        else:
            n = df[col].isnull().sum()
            missing_report.append({'变量': col, '缺失数': n, '缺失率%': round(n / len(df) * 100, 1),
                                   '说明': ''})
    missing_df = pd.DataFrame(missing_report).sort_values('缺失数', ascending=False)
    print('\n缺失值诊断报告（分类变量缺失已归为[缺失]类别）：')
    print(missing_df.to_string(index=False))

    return df, all_categorical_cols, missing_df, tracker

# ============================================================
# 常数变量检测
# ============================================================
def detect_constant_vars(df, categorical_cols, tracker=None):
    constant_vars = []
    df_y_valid = df[df[Y_COL].notnull()]
    for col in categorical_cols:
        valid_vals = df_y_valid[col].dropna()
        if valid_vals.nunique() <= 1:
            constant_vars.append(col)
            print(f'⚠️ 常数变量：{col}，已排除。')
    if constant_vars and tracker:
        tracker.add_step(
            len(df),
            f'排除常数变量（{len(constant_vars)}个，仅排除变量不排除样本）',
            f'常数变量：{", ".join(constant_vars)}'
        )
    return constant_vars


# ============================================================
# 基线特征比较表（Table 1）生成
# ============================================================
def generate_baseline_table(raw_df, y_col, baseline_vars_config):
    """
    生成基线特征比较表（论文 Table 1 格式）
    
    参数:
        raw_df: 原始 DataFrame（未经过任何预处理）
        y_col: 分组变量列名（如 '服药习惯_category'，原始值如 '空腹服药'/'餐后服药'）
        baseline_vars_config: [(列名, 显示名称, 类型), ...]
    
    返回:
        pd.DataFrame: 格式化后的基线比较表
    """
    # 检查分组
    if y_col not in raw_df.columns:
        print(f'  ⚠️ 基线表：分组列 {y_col} 不存在，跳过')
        return pd.DataFrame()
    
    groups = raw_df[y_col].dropna().unique()
    if len(groups) != 2:
        print(f'  ⚠️ 基线表：需要恰好2个分组，当前有 {len(groups)} 个，跳过')
        return pd.DataFrame()
    
    group1_label, group2_label = groups
    n1 = int(raw_df[y_col].value_counts().get(group1_label, 0))
    n2 = int(raw_df[y_col].value_counts().get(group2_label, 0))
    
    print(f'\n{"=" * 60}')
    print('生成基线特征比较表（Table 1）')
    print(f'  分组1：{group1_label}（n={n1}）')
    print(f'  分组2：{group2_label}（n={n2}）')
    print(f'  纳入变量数：{len(baseline_vars_config)}')
    print(f'{"=" * 60}')
    
    # 构建表格
    header_row = ['组别']
    group1_row = [f'{group1_label}（n={n1}）']
    group2_row = [f'{group2_label}（n={n2}）']
    stat_row = ['χ²/t值']
    p_row = ['P值']
    
    for col_name, display_name, var_type in baseline_vars_config:
        header_row.append(display_name)
        
        if col_name not in raw_df.columns:
            print(f'  ⚠️ 基线表：列 {col_name} 不存在，跳过')
            group1_row.append('')
            group2_row.append('')
            stat_row.append('')
            p_row.append('')
            continue
        
        # 提取两分组数据
        data = raw_df[[y_col, col_name]].dropna()
        g1_data = data[data[y_col] == group1_label][col_name]
        g2_data = data[data[y_col] == group2_label][col_name]
        
        if var_type == 'categorical':
            # === 分类变量：频数比 ===
            g1_counts = g1_data.value_counts()
            g2_counts = g2_data.value_counts()
            all_cats = sorted(
                set(g1_counts.index) | set(g2_counts.index),
                key=lambda x: str(x)
            )
            
            if len(all_cats) == 2:
                # 二分类：显示为 "n1:n2"
                g1_str = f"{g1_counts.get(all_cats[0], 0)}:{g1_counts.get(all_cats[1], 0)}"
                g2_str = f"{g2_counts.get(all_cats[0], 0)}:{g2_counts.get(all_cats[1], 0)}"
            else:
                # 多分类：显示为 "n1/n2/n3"
                g1_str = '/'.join([str(g1_counts.get(cat, 0)) for cat in all_cats])
                g2_str = '/'.join([str(g2_counts.get(cat, 0)) for cat in all_cats])
            
            group1_row.append(g1_str)
            group2_row.append(g2_str)
            
            # χ²检验
            try:
                contingency = pd.crosstab(data[y_col], data[col_name])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                stat_row.append(f'{chi2:.3f}')
                p_row.append('＜0.001' if p_val < 0.001 else f'{p_val:.3f}')
            except Exception as e:
                print(f'  ⚠️ 基线表 χ²检验失败（{col_name}）：{e}')
                stat_row.append('')
                p_row.append('')
        
        elif var_type == 'continuous':
            # === 连续变量：均值±标准差 ===
            g1_clean = pd.to_numeric(g1_data, errors='coerce').dropna()
            g2_clean = pd.to_numeric(g2_data, errors='coerce').dropna()
            
            if len(g1_clean) < 2 or len(g2_clean) < 2:
                print(f'  ⚠️ 基线表：{col_name} 有效样本不足（g1={len(g1_clean)}, g2={len(g2_clean)}）')
                group1_row.append('')
                group2_row.append('')
                stat_row.append('')
                p_row.append('')
                continue
            
            g1_mean = g1_clean.mean()
            g1_std = g1_clean.std()
            g2_mean = g2_clean.mean()
            g2_std = g2_clean.std()
            
            group1_row.append(f'{g1_mean:.2f}±{g1_std:.2f}')
            group2_row.append(f'{g2_mean:.2f}±{g2_std:.2f}')
            
            # Welch t检验（不假设方差齐性）
            try:
                t_stat, p_val = stats.ttest_ind(g1_clean, g2_clean, equal_var=False)
                stat_row.append(f'{t_stat:.3f}')
                p_row.append('＜0.001' if p_val < 0.001 else f'{p_val:.3f}')
            except Exception as e:
                print(f'  ⚠️ 基线表 t检验失败（{col_name}）：{e}')
                stat_row.append('')
                p_row.append('')
        
        else:
            print(f'  ⚠️ 基线表：未知变量类型 {var_type}（{col_name}），跳过')
            group1_row.append('')
            group2_row.append('')
            stat_row.append('')
            p_row.append('')
    
    # 构建 DataFrame
    table_data = [header_row, group1_row, group2_row, stat_row, p_row]
    table_df = pd.DataFrame(table_data)
    table_df.columns = table_df.iloc[0]
    table_df = table_df.iloc[1:].reset_index(drop=True)
    
    # 打印预览
    print('\n--- 基线特征比较表 ---')
    for _, row in table_df.iterrows():
        print(' | '.join([str(v) for v in row.values]))
    print('---')
    
    return table_df


# ============================================================
# 完全分离检测
# ============================================================
def detect_complete_separation_vars(df, x_cols, y_col):
    exclude_vars = []
    details = []

    model_data = df[[y_col] + list(x_cols)].dropna().copy()
    model_data[y_col] = model_data[y_col].astype(int)

    for v in x_cols:
        freq = pd.crosstab(model_data[v], model_data[y_col], dropna=False)
        for cat in freq.index:
            v0 = freq.loc[cat, 0] if 0 in freq.columns else 0
            v1 = freq.loc[cat, 1] if 1 in freq.columns else 0
            if v0 == 0 or v1 == 0:
                if v not in exclude_vars:
                    exclude_vars.append(v)
                details.append({
                    '变量': v,
                    '类别': cat,
                    '空腹服药频数': int(v0),
                    '餐后服药频数': int(v1),
                    '问题说明': f'完全分离：{"餐后服药" if v0==0 else "空腹服药"}频数为0，该变量将被排除'
                })
                print(f'  ⚠️ 完全分离检测：{v} 的类别 {cat} '
                      f'（空腹={v0}, 餐后={v1}），将排除此变量')

    details_df = pd.DataFrame(details) if details else pd.DataFrame()
    return exclude_vars, details_df

# ============================================================
# 单因素分析 — 分类变量（χ²检验）
# ============================================================
def univariate_categorical(df, categorical_cols, sparse_excluded_values, tracker=None, force_include_cols=None):
    if force_include_cols is None:
        force_include_cols = []
    results, desc, sparse_records = [], [], []
    for col in categorical_cols:
        if col in UNIVARIATE_EXCLUDE and col not in force_include_cols:
            print(f'  ⏭️ 跳过单因素分析（UNIVARIATE_EXCLUDE）：{col}')
            continue


        valid = df[[Y_COL, col]].dropna()
        n = len(valid)
        freq = pd.crosstab(valid[col], valid[Y_COL], dropna=False)
        row_totals = freq.sum(axis=1)

        sparse_mask = row_totals < SPARSE_THRESHOLD
        sparse_cats = row_totals[sparse_mask].index.tolist()
        valid_cats = row_totals[~sparse_mask].index.tolist()

        if sparse_cats:
            sparse_excluded_values[col] = sparse_cats
            for cat in sparse_cats:
                sparse_records.append({
                    '变量': col, '类别': cat,
                    '累计频数': int(row_totals[cat]),
                    '空腹服药频数': int(freq.loc[cat, 0]) if 0 in freq.columns else 0,
                    '餐后服药频数': int(freq.loc[cat, 1]) if 1 in freq.columns else 0,
                    '处理方式': f'累计频数<{SPARSE_THRESHOLD}，不计入统计'
                })

        if len(valid_cats) <= 1:
            results.append({
                '变量': col, '类型': '分类变量', '方法': 'χ²检验', '有效N': n,
                '统计量(χ²值)': np.nan, 'P值': '-',
                '排除的稀疏类别': ', '.join(sparse_cats) if sparse_cats else '无'
            })
            continue

        freq_valid = freq.loc[valid_cats]
        col_pct = freq_valid.div(freq_valid.sum(axis=0), axis=1) * 100
        for cat in valid_cats:
            desc.append({
                '变量': col, '类别': cat,
                '频数(空腹服药)': int(freq_valid.loc[cat, 0]) if 0 in freq_valid.columns else 0,
                '频数(餐后服药)': int(freq_valid.loc[cat, 1]) if 1 in freq_valid.columns else 0,
                '百分比(空腹服药)': round(col_pct.loc[cat, 0] if 0 in col_pct.columns else 0, 3),
                '百分比(餐后服药)': round(col_pct.loc[cat, 1] if 1 in col_pct.columns else 0, 3)
            })

        try:
            chi2, p_val, _, _ = stats.chi2_contingency(freq_valid)
            results.append({
                '变量': col, '类型': '分类变量', '方法': 'χ²检验', '有效N': n,
                '统计量(χ²值)': round(chi2, 3),
                'P值': '＜0.001' if p_val < 0.001 else f'{p_val:.3f}',
                '排除的稀疏类别': ', '.join(sparse_cats) if sparse_cats else '无'
            })
        except ValueError:
            results.append({
                '变量': col, '类型': '分类变量', '方法': 'χ²检验', '有效N': n,
                '统计量(χ²值)': np.nan, 'P值': '-',
                '排除的稀疏类别': ', '.join(sparse_cats) if sparse_cats else '无'
            })
    return pd.DataFrame(results), pd.DataFrame(desc), pd.DataFrame(sparse_records)

# ============================================================
# 多因素 Logistic 回归 + 完整可信度检验
# ============================================================
def run_multivariate_logistic(df, x_cols, y_col, tracker=None, force_include_cols=None):
    # 处理强制纳入的协变量：即使出现在排除列表中，也予以保留
    if force_include_cols is None:
        force_include_cols = []
    candidate_x = [v for v in x_cols if v not in EXHAUSTIVE_EXCLUDE or v in force_include_cols]
    
    # 输出排除与协变量信息
    if EXHAUSTIVE_EXCLUDE:
        print(f'\n  多因素排除变量（EXHAUSTIVE_EXCLUDE）：{EXHAUSTIVE_EXCLUDE}')
    if force_include_cols:
        print(f'  强制纳入协变量（COVARIATE_COLS）：{force_include_cols}')
        # 检查协变量是否在原始x_cols中存在
        missing_force = [v for v in force_include_cols if v not in x_cols]
        if missing_force:
            print(f'  ⚠️ 警告：以下协变量不在候选自变量列表中，将被忽略：{missing_force}')
        # 检查协变量是否因完全分离等问题可能被排除（后续会检测）
    print(f'  当前候选变量（{len(candidate_x)}个）：{candidate_x}')


    print('\n--- 完全分离检测 ---')
    sep_exclude_vars, sep_details_df = detect_complete_separation_vars(df, candidate_x, y_col)

    final_x_cols = [v for v in candidate_x if v not in sep_exclude_vars]

    if sep_exclude_vars:
        print(f'\n因完全分离排除的变量：{sep_exclude_vars}')
        print(f'最终纳入多因素回归的变量（{len(final_x_cols)}个）：{final_x_cols}')
    else:
        print('未检测到完全分离，所有候选变量均可纳入。')

    if len(final_x_cols) == 0:
        print('⚠️ 所有变量均因完全分离被排除，无法进行多因素回归。')
        return pd.DataFrame(), None, sep_exclude_vars, sep_details_df, None

    formula_parts = [f'C({v})' for v in final_x_cols]
    formula = f'{y_col} ~ ' + ' + '.join(formula_parts)

    model_data = df[[y_col] + list(final_x_cols)].dropna().copy()
    model_data[y_col] = model_data[y_col].astype(int)
    n_valid = len(model_data)

    if tracker:
        n_before = df[y_col].notnull().sum()
        n_excluded = n_before - n_valid
        if n_excluded > 0:
            tracker.add_step(
                n_valid,
                f'排除自变量缺失值（多因素模型纳入{len(final_x_cols)}个变量）',
                f'因{len(final_x_cols)}个自变量中存在缺失值，排除{n_excluded}条样本'
            )
        else:
            tracker.add_step(
                n_valid,
                f'多因素模型数据准备完成（纳入{len(final_x_cols)}个变量）',
                '所有候选变量均无缺失值，样本量保持不变'
            )

    print(f'\n多因素 Logistic 回归有效样本量：{n_valid}')
    n0 = (model_data[y_col] == 0).sum()
    n1 = (model_data[y_col] == 1).sum()
    print(f'  空腹服药：{n0}，餐后服药：{n1}')

    if n_valid < 10 or n0 < 5 or n1 < 5:
        print('⚠️ 样本量不足或结局事件数过少，无法进行多因素回归')
        return pd.DataFrame(), None, sep_exclude_vars, sep_details_df, None

    model = None
    fit_method_used = 'newton'
    try:
        model = logit(formula, data=model_data).fit(maxiter=500, disp=False)
    except Exception:
        fit_method_used = 'bfgs'
        try:
            model = logit(formula, data=model_data).fit(method='bfgs', maxiter=1000, disp=False)
        except Exception as e:
            print(f'⚠️ 多因素回归失败：{e}')
            return pd.DataFrame(), None, sep_exclude_vars, sep_details_df, None

    if model is None:
        return pd.DataFrame(), None, sep_exclude_vars, sep_details_df, None

    iterations = model.mle_retvals.get('iterations', 'N/A') if hasattr(model, 'mle_retvals') else 'N/A'
    print(f'  模型拟合方法：{fit_method_used}，迭代次数：{iterations}')

    # 提取回归结果
    records = []
    for var_name in [v for v in model.params.index if v != 'Intercept']:
        # ========== 新增过滤：跳过所有"[T.缺失]"类别 ==========
        if '[T.缺失]' in var_name:
            continue
        # ======================================================
        
        beta = model.params[var_name]
        se = model.bse[var_name]
        wald = (beta / se) ** 2 if not pd.isna(se) else np.nan
        or_v = np.exp(beta)
        ci_l = np.exp(beta - 1.96 * se) if not pd.isna(se) else np.nan
        ci_h = np.exp(beta + 1.96 * se) if not pd.isna(se) else np.nan
        p_val = model.pvalues[var_name]
        label = LABEL_MAP.get(var_name, var_name)

        is_separated = (
            np.isinf(or_v) or or_v > 1e6 or
            (not pd.isna(se) and se > 1000) or
            (not pd.isna(ci_l) and (np.isinf(ci_l) or ci_l > 1e6)) or
            (not pd.isna(ci_h) and (np.isinf(ci_h) or ci_h > 1e6))
        )

        records.append({
            '自变量': label,
            'β值': round(beta, 3),
            'SE值': round(se, 3) if not pd.isna(se) else np.nan,
            'Waldχ²值': round(wald, 3) if not pd.isna(wald) else np.nan,
            'OR值': 'Inf' if np.isinf(or_v) else round(or_v, 3),
            '95%CI下限': 'Inf' if np.isinf(ci_l) else (round(ci_l, 3) if not pd.isna(ci_l) else '-'),
            '95%CI上限': 'Inf' if np.isinf(ci_h) else (round(ci_h, 3) if not pd.isna(ci_h) else '-'),
            '95%CI': 'Inf~Inf' if is_separated else
                     (f'{ci_l:.3f}~{ci_h:.3f}' if not pd.isna(ci_l) else '-'),
            'P值': '＜0.001' if (not pd.isna(p_val) and p_val < 0.001) else
                   (f'{p_val:.3f}' if not pd.isna(p_val) else '-'),
            '完全分离': '是' if is_separated else '否',
            '有效N': n_valid,
        })

    result_df = pd.DataFrame(records)


    fit_stats = {
        'AIC': round(model.aic, 2),
        'BIC': round(model.bic, 2),
        'Log-likelihood': round(model.llf, 3),
        'Pseudo R² (McFadden)': round(model.prsquared, 4),
        '模型LLR检验P值': '＜0.001' if model.llr_pvalue < 0.001 else f'{model.llr_pvalue:.4f}',
        '有效N': n_valid,
        '事件数(餐后服药)': n1,
        '非事件数(空腹服药)': n0,
        '纳入变量数': len(final_x_cols),
        '排除的完全分离变量': ', '.join(sep_exclude_vars) if sep_exclude_vars else '无',
        '排除的EXHAUSTIVE_EXCLUDE变量': ', '.join(EXHAUSTIVE_EXCLUDE) if EXHAUSTIVE_EXCLUDE else '无',
        '拟合方法': fit_method_used,
    }

    # ================================================================
    # 可信度检验
    # ================================================================
    credibility_results = {}

    y_true = model_data[y_col].values.astype(int)
    y_pred_prob = model.predict(model_data).values
    y_pred_logit = model.predict(model_data, linear=True).values

    # Hosmer-Lemeshow
    try:
        n_groups = min(10, n_valid // 2)
        if n_groups < 6:
            n_groups = max(3, n_valid // 10)
        
        sorted_idx = np.argsort(y_pred_prob)
        y_true_sorted = y_true[sorted_idx]
        y_prob_sorted = y_pred_prob[sorted_idx]
        
        group_edges = np.linspace(0, n_valid, n_groups + 1, dtype=int)
        
        hl_chi2 = 0.0
        hl_table = []
        
        for g in range(n_groups):
            start = group_edges[g]
            end = group_edges[g + 1]
            n_g = end - start
            if n_g == 0:
                continue
            obs_events = np.sum(y_true_sorted[start:end])
            exp_events = np.sum(y_prob_sorted[start:end])
            obs_non_events = n_g - obs_events
            exp_non_events = n_g - exp_events
            
            hl_table.append({
                '分组': g + 1, '样本量': n_g,
                '观测事件数': int(obs_events), '期望事件数': round(exp_events, 2),
                '观测非事件数': int(obs_non_events), '期望非事件数': round(exp_non_events, 2)
            })
            
            if exp_events > 0 and exp_non_events > 0:
                hl_chi2 += (obs_events - exp_events) ** 2 / exp_events
                hl_chi2 += (obs_non_events - exp_non_events) ** 2 / exp_non_events
        
        hl_df = n_groups - 2
        hl_p = 1.0 - stats.chi2.cdf(hl_chi2, hl_df) if hl_df > 0 else 1.0
        
        hl_summary = {
            '检验方法': 'Hosmer-Lemeshow 拟合优度检验',
            'χ²值': round(hl_chi2, 4), '自由度': hl_df,
            'P值': '＜0.001' if hl_p < 0.001 else f'{hl_p:.4f}',
            '判断标准': 'P > 0.05 表示模型校准度良好',
            '结论': '校准度良好（P>0.05）' if hl_p > 0.05 else '校准度不佳（P≤0.05）'
        }
        credibility_results['Hosmer-Lemeshow检验'] = hl_summary
        credibility_results['Hosmer-Lemeshow分组详情'] = pd.DataFrame(hl_table)
        
        print(f'\n  Hosmer-Lemeshow 检验：χ²={hl_chi2:.4f}, df={hl_df}, P={hl_summary["P值"]}')
        print(f'  结论：{hl_summary["结论"]}')
        
    except Exception as e:
        print(f'  ⚠️ Hosmer-Lemeshow 检验失败：{e}')
        credibility_results['Hosmer-Lemeshow检验'] = {
            '检验方法': 'Hosmer-Lemeshow 拟合优度检验', '结果': f'计算失败：{e}'
        }

    # ROC 与 AUC
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        se_auc = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n0 * n1))
        auc_ci_lower = max(0, auc - 1.96 * se_auc)
        auc_ci_upper = min(1, auc + 1.96 * se_auc)
        
        auc_summary = {
            '检验方法': 'ROC曲线与AUC',
            'AUC值': round(auc, 4), 'AUC标准误': round(se_auc, 4),
            'AUC的95%CI': f'{auc_ci_lower:.4f}~{auc_ci_upper:.4f}',
            '最佳截断值': round(thresholds[best_idx], 4),
            '最佳截断值-敏感度': round(tpr[best_idx], 4),
            '最佳截断值-特异度': round(1 - fpr[best_idx], 4),
            '约登指数': round(youden[best_idx], 4),
            '判断标准': 'AUC<0.5=无区分能力, 0.5-0.7=较低, 0.7-0.8=可接受, 0.8-0.9=优秀, >0.9=杰出',
            '结论': '无区分能力' if auc < 0.5 else (
                    '区分度较低' if auc < 0.7 else (
                    '区分度可接受' if auc < 0.8 else (
                    '区分度优秀' if auc < 0.9 else '区分度杰出')))
        }
        credibility_results['ROC与AUC'] = auc_summary
        
        roc_points = pd.DataFrame({'假阳性率(FPR)': fpr, '真阳性率(TPR)': tpr, '阈值': thresholds})
        if len(roc_points) > 100:
            roc_points = roc_points.iloc[np.linspace(0, len(roc_points)-1, 100, dtype=int)]
        credibility_results['ROC曲线数据点'] = roc_points
        
        print(f'\n  AUC = {auc:.4f} (95%CI: {auc_ci_lower:.4f}~{auc_ci_upper:.4f})')
        print(f'  结论：{auc_summary["结论"]}')
        
    except Exception as e:
        print(f'  ⚠️ AUC 计算失败：{e}')
        credibility_results['ROC与AUC'] = {'检验方法': 'ROC曲线与AUC', '结果': f'计算失败：{e}'}

    # 分类矩阵
    try:
        y_pred_class = (y_pred_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_class)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        cm_summary = {
            '检验方法': '分类矩阵（截断值=0.5）',
            '真阴性(TN)': int(tn), '假阳性(FP)': int(fp),
            '假阴性(FN)': int(fn), '真阳性(TP)': int(tp),
            '敏感度': round(sensitivity, 4), '特异度': round(specificity, 4),
            '阳性预测值(PPV)': round(ppv, 4), '阴性预测值(NPV)': round(npv, 4),
            '准确率': round(accuracy, 4), 'F1分数': round(f1, 4),
        }
        credibility_results['分类矩阵'] = cm_summary
        
        print(f'\n  分类矩阵（截断值=0.5）：')
        print(f'    准确率={accuracy:.4f}, 敏感度={sensitivity:.4f}, 特异度={specificity:.4f}')
        print(f'    PPV={ppv:.4f}, NPV={npv:.4f}, F1={f1:.4f}')
        
    except Exception as e:
        print(f'  ⚠️ 分类矩阵计算失败：{e}')
        credibility_results['分类矩阵'] = {'检验方法': '分类矩阵', '结果': f'计算失败：{e}'}

    # VIF
    try:
        vif_data = model_data.copy()
        for v in final_x_cols:
            dummies = pd.get_dummies(vif_data[v], prefix=v, drop_first=True)
            vif_data = pd.concat([vif_data, dummies], axis=1)
            vif_data = vif_data.drop(columns=[v])
        
        numeric_cols = vif_data.select_dtypes(include=[np.number]).columns
        vif_numeric = vif_data[numeric_cols].drop(columns=[y_col], errors='ignore')
        vif_numeric = vif_numeric.dropna()
        
        if vif_numeric.shape[1] >= 2 and vif_numeric.shape[0] > vif_numeric.shape[1]:
            vif_results_list = []
            for i, col in enumerate(vif_numeric.columns):
                try:
                    vif = variance_inflation_factor(vif_numeric.values, i)
                    vif_results_list.append({
                        '变量': col, 'VIF': round(vif, 4),
                        '判断': '严重(VIF>10)' if vif > 10 else (
                                '中等(VIF>5)' if vif > 5 else (
                                '轻微(VIF>2)' if vif > 2 else '无'))
                    })
                except Exception:
                    vif_results_list.append({'变量': col, 'VIF': np.nan, '判断': '计算失败'})
            
            vif_df = pd.DataFrame(vif_results_list)
            credibility_results['多重共线性检验(VIF)'] = vif_df
            
            max_vif = vif_df['VIF'].max() if 'VIF' in vif_df.columns else np.nan
            print(f'\n  多重共线性检验（VIF）：最大VIF = {max_vif:.4f}' if not pd.isna(max_vif) else '  最大VIF = N/A')
        else:
            credibility_results['多重共线性检验(VIF)'] = pd.DataFrame({'提示': ['样本量不足，无法计算VIF']})
            
    except Exception as e:
        print(f'  ⚠️ VIF 计算失败：{e}')
        credibility_results['多重共线性检验(VIF)'] = pd.DataFrame({'结果': [f'计算失败：{e}']})

    # Link Test
    try:
        link_data = pd.DataFrame({
            y_col: y_true, '_y_hat': y_pred_logit, '_y_hat_sq': y_pred_logit ** 2
        })
        link_formula = f'{y_col} ~ _y_hat + _y_hat_sq'
        link_model = logit(link_formula, data=link_data).fit(maxiter=500, disp=False)
        
        p_hat_sq = link_model.pvalues['_y_hat_sq']
        
        link_summary = {
            '检验方法': 'Link Test（模型设定检验）',
            '_y_hat系数': round(link_model.params['_y_hat'], 4),
            '_y_hat²系数': round(link_model.params['_y_hat_sq'], 4),
            '_y_hat²的P值': '＜0.001' if p_hat_sq < 0.001 else f'{p_hat_sq:.4f}',
            '判断标准': '若_y_hat²的P>0.05，则模型设定正确',
            '结论': '模型设定正确（P>0.05）' if p_hat_sq > 0.05 else '模型设定可能存在问题（P≤0.05）'
        }
        credibility_results['LinkTest模型设定检验'] = link_summary
        
        print(f'\n  Link Test：_y_hat²的P值 = {link_summary["_y_hat²的P值"]}')
        print(f'  结论：{link_summary["结论"]}')
        
    except Exception as e:
        print(f'  ⚠️ Link Test 失败：{e}')
        credibility_results['LinkTest模型设定检验'] = {'检验方法': 'Link Test', '结果': f'计算失败：{e}'}

    # 残差分析
    try:
        pearson_resid = model.resid_pearson.values
        deviance_resid = model.resid_dev.values
        
        outlier_mask = (np.abs(pearson_resid) > 2) | (np.abs(deviance_resid) > 2)
        n_outliers = int(np.sum(outlier_mask))
        
        residual_summary = {
            '检验方法': '残差分析与异常值检测',
            'Pearson残差均值': round(float(np.mean(pearson_resid)), 4),
            'Pearson残差标准差': round(float(np.std(pearson_resid)), 4),
            'Deviance残差均值': round(float(np.mean(deviance_resid)), 4),
            'Deviance残差标准差': round(float(np.std(deviance_resid)), 4),
            '异常值数量(|残差|>2)': n_outliers,
            '异常值比例': f'{round(n_outliers / n_valid * 100, 2)}%',
            '判断标准': '异常值比例<5%通常可接受',
            '结论': '残差分布正常' if n_outliers / n_valid < 0.05 else f'存在{n_outliers}个异常值'
        }
        credibility_results['残差分析与异常值检测'] = residual_summary
        
        if n_outliers > 0:
            outlier_indices = np.where(outlier_mask)[0]
            outlier_details = []
            for idx in outlier_indices[:20]:
                outlier_details.append({
                    '样本序号': int(idx), '实际结局': int(y_true[idx]),
                    '预测概率': round(float(y_pred_prob[idx]), 4),
                    'Pearson残差': round(float(pearson_resid[idx]), 4),
                    'Deviance残差': round(float(deviance_resid[idx]), 4)
                })
            credibility_results['异常值详情(前20条)'] = pd.DataFrame(outlier_details)
        
        print(f'\n  残差分析：异常值 {n_outliers}/{n_valid} ({round(n_outliers/n_valid*100,1)}%)')
        
    except Exception as e:
        print(f'  ⚠️ 残差分析失败：{e}')
        credibility_results['残差分析与异常值检测'] = {'检验方法': '残差分析', '结果': f'计算失败：{e}'}

    # 综合评级
    try:
        checks_passed = 0
        checks_total = 0
        check_details = []
        
        checks_total += 1
        if 'Hosmer-Lemeshow检验' in credibility_results and 'P值' in credibility_results['Hosmer-Lemeshow检验']:
            if parse_p_value(credibility_results['Hosmer-Lemeshow检验']['P值']) > 0.05:
                checks_passed += 1
                check_details.append('✓ HL检验通过')
            else:
                check_details.append('✗ HL检验未通过')
        
        checks_total += 1
        if 'ROC与AUC' in credibility_results and 'AUC值' in credibility_results['ROC与AUC']:
            auc_val = credibility_results['ROC与AUC']['AUC值']
            if isinstance(auc_val, (int, float)) and auc_val >= 0.7:
                checks_passed += 1
                check_details.append(f'✓ AUC={auc_val}≥0.7')
            else:
                check_details.append(f'✗ AUC={auc_val}<0.7')
        
        checks_total += 1
        if '多重共线性检验(VIF)' in credibility_results:
            vif_data = credibility_results['多重共线性检验(VIF)']
            if isinstance(vif_data, pd.DataFrame) and 'VIF' in vif_data.columns:
                max_vif = vif_data['VIF'].max()
                if not pd.isna(max_vif) and max_vif < 5:
                    checks_passed += 1
                    check_details.append(f'✓ 最大VIF={max_vif:.2f}<5')
                else:
                    check_details.append(f'✗ 最大VIF={max_vif:.2f}≥5')
            else:
                checks_total -= 1
        
        checks_total += 1
        if 'LinkTest模型设定检验' in credibility_results and '_y_hat²的P值' in credibility_results['LinkTest模型设定检验']:
            if parse_p_value(credibility_results['LinkTest模型设定检验']['_y_hat²的P值']) > 0.05:
                checks_passed += 1
                check_details.append('✓ Link Test通过')
            else:
                check_details.append('✗ Link Test未通过')
        
        checks_total += 1
        if '残差分析与异常值检测' in credibility_results and '异常值比例' in credibility_results['残差分析与异常值检测']:
            outlier_pct = float(credibility_results['残差分析与异常值检测']['异常值比例'].replace('%', ''))
            if outlier_pct < 5:
                checks_passed += 1
                check_details.append(f'✓ 异常值={outlier_pct}%<5%')
            else:
                check_details.append(f'✗ 异常值={outlier_pct}%≥5%')
        
        pass_rate = checks_passed / checks_total if checks_total > 0 else 0
        if pass_rate >= 0.8:
            overall_grade = '★★★★★ 模型可信度优秀'
        elif pass_rate >= 0.6:
            overall_grade = '★★★★☆ 模型可信度良好'
        elif pass_rate >= 0.4:
            overall_grade = '★★★☆☆ 模型可信度一般'
        elif pass_rate >= 0.2:
            overall_grade = '★★☆☆☆ 模型可信度较低'
        else:
            overall_grade = '★☆☆☆☆ 模型可信度差'
        
        credibility_results['可信度综合评级'] = {
            '检验方法': '可信度综合评级',
            '可信度综合评级': overall_grade,
            '通过检验数': checks_passed, '总检验数': checks_total,
            '通过率': f'{round(pass_rate * 100, 1)}%',
            '各项检验详情': '; '.join(check_details)
        }
        
        print(f'\n  {"="*40}')
        print(f'  可信度综合评级：{overall_grade}')
        print(f'  通过率：{checks_passed}/{checks_total} = {round(pass_rate*100,1)}%')
        print(f'  {"="*40}')
        
    except Exception as e:
        print(f'  ⚠️ 综合评级计算失败：{e}')

    if tracker:
        tracker.add_step(
            n_valid,
            f'多因素回归完成（纳入{len(final_x_cols)}个变量）',
            f'AIC={fit_stats["AIC"]}, Pseudo R²={fit_stats["Pseudo R² (McFadden)"]}'
        )

    return result_df, fit_stats, sep_exclude_vars, sep_details_df, credibility_results

# ============================================================
# 导出到 Excel（已优化可信度检验输出）
# ============================================================
def export_to_excel(output_dir, dfs_dict):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'统计分析结果_{timestamp}.xlsx')

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df_data in dfs_dict.items():
            if df_data is not None:
                if isinstance(df_data, pd.DataFrame):
                    if not df_data.empty:
                        df_data.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        pd.DataFrame({'提示': ['无数据']}).to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(df_data, dict):
                    pd.DataFrame([df_data]).to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    pd.DataFrame({'提示': ['不支持的数据类型']}).to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'\n所有结果已保存至：{output_path}')
    return output_path

# ============================================================
# 主流程
# ============================================================
def main():
    # ================================================================
    # 第一步：读取原始数据（用于基线表，保留原始值）
    # ================================================================
    raw_df_for_baseline = pd.read_excel(FILE_PATH, header=0)
    
    # ================================================================
    # 第二步：生成基线特征比较表（Table 1）
    # ================================================================
    if BASELINE_TABLE_ENABLED and BASELINE_VARIABLES:
        baseline_df = generate_baseline_table(
            raw_df=raw_df_for_baseline,
            y_col=Y_COL,
            baseline_vars_config=BASELINE_VARIABLES
        )
    else:
        baseline_df = pd.DataFrame()
    
    # ================================================================
    # 第三步：数据加载与预处理（用于后续分析）
    # ================================================================
    df, categorical_cols, missing_df, tracker = load_and_preprocess(FILE_PATH)
    
    # 常数变量检测
    constant_vars = detect_constant_vars(df, categorical_cols, tracker)
    valid_categorical = [c for c in categorical_cols if c not in constant_vars]
    
    # 稀疏类别容器
    sparse_excluded_values = {}
    
    # 单因素分析
    print(f'\n{"=" * 60}')
    print('单因素分析（所有变量均为分类变量，使用χ²检验）')
    print(f'稀疏阈值：累计频数 < {SPARSE_THRESHOLD}')
    print(f'{"=" * 60}')
    
    uni_cat_df, cat_desc_df, sparse_cat_df = univariate_categorical(
        df, valid_categorical, sparse_excluded_values, tracker,
        force_include_cols=UNIVARIATE_COVARIATE_COLS
    )
    
    print('\n=== 单因素分析汇总 ===')
    print(uni_cat_df.to_string(index=False))
    
    # 多因素 Logistic 回归 + 可信度检验
    print(f'\n{"=" * 60}')
    print('多因素 Logistic 回归（全变量模型，年龄已转为二分类）')
    print(f'{"=" * 60}')
    
    multi_df, fit_stats, excluded_sep_vars, sep_details_df, credibility_results = \
        run_multivariate_logistic(df, valid_categorical, Y_COL, tracker,
                                  force_include_cols=MULTIVARIATE_COVARIATE_COLS)
    
    # 样本量追踪表
    sample_size_df = tracker.to_dataframe()
    
    # ================================================================
    # 构建导出字典
    # ================================================================
    dfs_dict = {
        '样本量追踪': sample_size_df,
        '缺失值诊断': missing_df,
        '分类变量描述性统计': cat_desc_df,
        '单因素分析汇总': uni_cat_df,
        '稀疏类别说明': sparse_cat_df,
    }
    
    # 基线特征比较表放在最前面
    if baseline_df is not None and not baseline_df.empty:
        ordered_dfs = {'基线特征比较表': baseline_df}
        ordered_dfs.update(dfs_dict)
        dfs_dict = ordered_dfs
    
    # 完全分离检测详情
    if sep_details_df is not None and not sep_details_df.empty:
        dfs_dict['完全分离检测详情'] = sep_details_df
    
    if excluded_sep_vars:
        dfs_dict['完全分离排除说明'] = pd.DataFrame({
            '说明': [
                f'因完全分离被排除的变量：{", ".join(excluded_sep_vars)}',
                '完全分离指某个分类变量的某个类别中，所有样本均属于同一结局组，',
                '导致该变量无法纳入Logistic回归模型。'
            ]
        })
    
    if multi_df is not None and not multi_df.empty:
        dfs_dict['多因素Logistic回归结果'] = multi_df
        
        if fit_stats:
            dfs_dict['多因素模型拟合统计量'] = pd.DataFrame([fit_stats])
        
        sig_multi = filter_significant(multi_df, 'P值', 0.05)
        if not sig_multi.empty:
            if '完全分离' in sig_multi.columns:
                sig_multi_clean = sig_multi[sig_multi['完全分离'] != '是'].copy()
            else:
                sig_multi_clean = sig_multi.copy()
            if not sig_multi_clean.empty:
                dfs_dict['多因素显著结果_P<0.05'] = sig_multi_clean
    
    # 可信度检验结果
    if credibility_results:
        summary_dicts = {}
        detail_dfs = {}
        for key, value in credibility_results.items():
            if isinstance(value, dict):
                summary_dicts[key] = value
            elif isinstance(value, pd.DataFrame):
                detail_dfs[key] = value
            else:
                summary_dicts[key] = {'结果': str(value)}
        
        summary_rows = []
        for test_name, test_dict in summary_dicts.items():
            for metric, val in test_dict.items():
                summary_rows.append({
                    '检验项目': test_name,
                    '指标': metric,
                    '值': val
                })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            dfs_dict['可信度检验'] = summary_df
        
        for detail_name, detail_df in detail_dfs.items():
            sheet_name = f'可信度检验_{detail_name}'
            dfs_dict[sheet_name] = detail_df
    
    # 警告信息
    warnings_list = []
    if constant_vars:
        warnings_list.append(f'已排除常数变量：{", ".join(constant_vars)}')
    if excluded_sep_vars:
        warnings_list.append(f'因完全分离排除的变量：{", ".join(excluded_sep_vars)}')
    warnings_list.append(f'稀疏阈值：累计频数 < {SPARSE_THRESHOLD}')
    warnings_list.append(f'年龄分组阈值：{AGE_THRESHOLD}岁及以上为高龄')
    if multi_df is not None and not multi_df.empty and fit_stats:
        warnings_list.append(f'多因素回归有效样本量：{fit_stats["有效N"]}')
        warnings_list.append(f'多因素回归纳入变量数：{fit_stats["纳入变量数"]}')
    if credibility_results and '可信度综合评级' in credibility_results:
        warnings_list.append(
            f'模型可信度综合评级：{credibility_results["可信度综合评级"]["可信度综合评级"]}'
        )
    if warnings_list:
        dfs_dict['警告信息'] = pd.DataFrame({'警告信息': warnings_list})
    
    output_dir = os.path.dirname(FILE_PATH)
    export_to_excel(output_dir, dfs_dict)


if __name__ == '__main__':
    main()


