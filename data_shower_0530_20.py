import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 生成时间戳 ====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = r'D:/data/data_arrangement/Output'

# ==================== 数据加载和预处理 ====================
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    df = pd.read_excel(file_path)
    
    # ===== 安全的缺失值处理 =====
    print("\n原始数据中'缺失值'字符串分布:")
    for col in df.columns:
        if df[col].dtype == 'object':
            missing_count = (df[col] == '缺失值').sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count}个")
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    df.replace('缺失值', np.nan, inplace=True)
    
    # ===== 数值变量转换 =====
    numeric_cols = ['L-SBP_Crit', 'L-DBP_Crit', 'R-SBP_Crit', 'R-DBP_Crit',
                    'AGE_Crit', 'BMI_Crit', 'HbA1c_Crit']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ===== 服药习惯处理（核心分组变量） =====
    print(f"\n服药习惯原始类别分布:")
    print(df['服药习惯_category'].value_counts(dropna=False))
    
    # 将非标准类别归入"非典型服药习惯"
    if '服药习惯_category' in df.columns:
        df['服药习惯_category'] = df['服药习惯_category'].apply(
            lambda x: '非典型服药习惯' if pd.isna(x) or x in ['不规律服药', '睡前服药', '不服药'] 
                      else x
        )
    df['服药习惯_category'] = df['服药习惯_category'].fillna('非典型服药习惯')
    
    print(f"\n服药习惯处理后类别分布:")
    print(df['服药习惯_category'].value_counts())
    
    
    # 创建二分类因变量：空腹服药=1，餐后服药=0
    df['med_fasting'] = np.where(df['服药习惯_category'] == '空腹服药', 1, 
                                  np.where(df['服药习惯_category'] == '餐后服药', 0, np.nan))
    
    fasting_n = df['med_fasting'].sum()
    postmeal_n = (df['med_fasting'] == 0).sum()
    print(f"\n主要分析样本量（空腹服药={int(fasting_n)}例 + 餐后服药={int(postmeal_n)}例）: {int(fasting_n + postmeal_n)}")
    print(f"非典型服药习惯样本量（排除）: {(df['服药习惯_category'] == '非典型服药习惯').sum()}")
    
    # ===== 性别变量处理 =====
    print(f"\n性别列(Gender_category)唯一值: {df['Gender_category'].dropna().unique()}")
    df['Gender_male'] = np.where(df['Gender_category'] == '男', 1, 
                                  np.where(df['Gender_category'] == '女', 0, np.nan))
    male_count = df['Gender_male'].sum()
    valid_gender = df['Gender_male'].notna().sum()
    print(f"性别: 男性={int(male_count)}例, 女性={int(valid_gender - male_count)}例, 缺失={int(df['Gender_male'].isna().sum())}例")
    
    # ===== 用药记忆情况处理 =====
    print(f"\n用药记忆情况原始类别分布:")
    print(df['用药记忆情况_category'].value_counts(dropna=False))
    
    # 处理缺失值
    df['用药记忆情况_category'] = df['用药记忆情况_category'].fillna('缺失值')
    
    print(f"\n用药记忆情况处理后类别分布:")
    print(df['用药记忆情况_category'].value_counts())
    
    memory_counts = df['用药记忆情况_category'].value_counts()
    for cat in ['可提供具体用药记录', '仅可提供服药种类', '完全不清楚用药情况']:
        count = memory_counts.get(cat, 0)
        print(f"  {cat}: {count}例")
    
    # ===== BMI分类 =====
    df['BMI_group'] = pd.cut(df['BMI_Crit'], 
                              bins=[0, 18.5, 24, 28, 100],
                              labels=['<18.5', '18.5~<24', '24~<28', '≥28'])

    # ===== HbA1c分类（新增） =====
    df['HbA1c_group'] = pd.cut(df['HbA1c_Crit'],
                                bins=[0, 7, 9, 100],
                                labels=['<7%', '7%~<9%', '≥9%'])
   
    # ===== 年龄分组（每5岁为一个分析单元） =====
    age_min = df['AGE_Crit'].min()
    age_max = df['AGE_Crit'].max()
    age_bins = list(range(int(np.floor(age_min / 5) * 5), int(np.ceil(age_max / 5) * 5) + 5, 5))
    age_labels = [f'{int(b)}~<{int(b+5)}' for b in age_bins[:-1]]
    df['AGE_group'] = pd.cut(df['AGE_Crit'], bins=age_bins, labels=age_labels, right=False)

    # ===== 缺失值统计 =====
    print("\n" + "="*60)
    print("关键变量缺失值统计:")
    print("="*60)
    key_cols = ['AGE_Crit', 'Gender_category', 'BMI_Crit', 'HbA1c_Crit',
                'L-SBP_Crit', 'L-DBP_Crit', 'R-SBP_Crit', 'R-DBP_Crit',
                'TC_category', 'TG_category', 'LDL_category', 
                '用药记忆情况_category']

    for col in key_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            total = len(df)
            print(f"  {col}: 缺失 {missing}/{total} ({missing/total*100:.1f}%)")
    
    return df


# ==================== 表1：服药习惯分组描述 ====================
def generate_table1(df):
    """按服药习惯分组展示各变量分布"""
    
    print("\n" + "="*60)
    print("表1：不同服药习惯调查对象的基本特征")
    print("="*60)
    
    # 筛选主要分析人群
    df_main = df[df['服药习惯_category'].isin(['空腹服药', '餐后服药'])].copy()
    
    table1_data = []
    
    groups = ['空腹服药', '餐后服药']
    for group in groups:
        group_df = df_main[df_main['服药习惯_category'] == group]
        
        # 年龄分组（每5岁）
        age_valid = group_df['AGE_group'].dropna()
        age_dist = age_valid.value_counts().sort_index()
        age_str = '; '.join([f"{k}:{v}" for k, v in age_dist.items()])
        age_n = len(age_valid)

        
        # 性别
        male_valid = group_df['Gender_male'].dropna()
        male_count = male_valid.sum()
        male_pct = male_count / len(male_valid) * 100 if len(male_valid) > 0 else 0
        gender_str = f"{int(male_count)} ({male_pct:.1f}%)"
        gender_n = len(male_valid)
        
        # BMI
        bmi_valid = group_df['BMI_Crit'].dropna()
        bmi_mean = bmi_valid.mean()
        bmi_std = bmi_valid.std()
        bmi_n = len(bmi_valid)
        
        # HbA1c
        hba1c_valid = group_df['HbA1c_Crit'].dropna()
        hba1c_mean = hba1c_valid.mean()
        hba1c_std = hba1c_valid.std()
        hba1c_n = len(hba1c_valid)
        
        # 血压
        lsbp_valid = group_df['L-SBP_Crit'].dropna()
        lsbp_mean = lsbp_valid.mean() if len(lsbp_valid) > 0 else np.nan
        lsbp_std = lsbp_valid.std() if len(lsbp_valid) > 0 else np.nan
        lsbp_n = len(lsbp_valid)
        
        ldbp_valid = group_df['L-DBP_Crit'].dropna()
        ldbp_mean = ldbp_valid.mean() if len(ldbp_valid) > 0 else np.nan
        ldbp_std = ldbp_valid.std() if len(ldbp_valid) > 0 else np.nan
        ldbp_n = len(ldbp_valid)
        
        rsbp_valid = group_df['R-SBP_Crit'].dropna()
        rsbp_mean = rsbp_valid.mean() if len(rsbp_valid) > 0 else np.nan
        rsbp_std = rsbp_valid.std() if len(rsbp_valid) > 0 else np.nan
        rsbp_n = len(rsbp_valid)
        
        rdbp_valid = group_df['R-DBP_Crit'].dropna()
        rdbp_mean = rdbp_valid.mean() if len(rdbp_valid) > 0 else np.nan
        rdbp_std = rdbp_valid.std() if len(rdbp_valid) > 0 else np.nan
        rdbp_n = len(rdbp_valid)
        
        # 分类变量分布
        tc_dist = group_df['TC_category'].dropna().value_counts().to_dict()
        tc_str = '; '.join([f"{k}:{v}" for k, v in tc_dist.items()])
        tg_dist = group_df['TG_category'].dropna().value_counts().to_dict()
        tg_str = '; '.join([f"{k}:{v}" for k, v in tg_dist.items()])
        ldl_dist = group_df['LDL_category'].dropna().value_counts().to_dict()
        ldl_str = '; '.join([f"{k}:{v}" for k, v in ldl_dist.items()])
        # 用药记忆情况
        memory_dist = group_df['用药记忆情况_category'].dropna().value_counts().to_dict()
        memory_str = '; '.join([f"{k}:{v}" for k, v in memory_dist.items()])

        
        table1_data.append({
            '服药习惯': group,
            '样本量': len(group_df),
            '年龄（每5岁分组）': age_str + f" (n={age_n})",
            '性别（男性比例）': gender_str + f" (n={gender_n})",
            'BMI': f"{bmi_mean:.1f} ± {bmi_std:.1f} (n={bmi_n})",
            'HbA1c': f"{hba1c_mean:.1f} ± {hba1c_std:.1f} (n={hba1c_n})",
            'L-SBP': f"{lsbp_mean:.1f} ± {lsbp_std:.1f} (n={lsbp_n})" if not np.isnan(lsbp_mean) else 'N/A',
            'L-DBP': f"{ldbp_mean:.1f} ± {ldbp_std:.1f} (n={ldbp_n})" if not np.isnan(ldbp_mean) else 'N/A',
            'R-SBP': f"{rsbp_mean:.1f} ± {rsbp_std:.1f} (n={rsbp_n})" if not np.isnan(rsbp_mean) else 'N/A',
            'R-DBP': f"{rdbp_mean:.1f} ± {rdbp_std:.1f} (n={rdbp_n})" if not np.isnan(rdbp_mean) else 'N/A',
            'TC_category': tc_str,
            'TG_category': tg_str,
            'LDL_category': ldl_str,
            '用药记忆情况': memory_str
        })

    # 总体血压
    lsbp_valid_t = df_main['L-SBP_Crit'].dropna()
    lsbp_mean_t = lsbp_valid_t.mean() if len(lsbp_valid_t) > 0 else np.nan
    lsbp_std_t = lsbp_valid_t.std() if len(lsbp_valid_t) > 0 else np.nan
    lsbp_n_t = len(lsbp_valid_t)

    ldbp_valid_t = df_main['L-DBP_Crit'].dropna()
    ldbp_mean_t = ldbp_valid_t.mean() if len(ldbp_valid_t) > 0 else np.nan
    ldbp_std_t = ldbp_valid_t.std() if len(ldbp_valid_t) > 0 else np.nan
    ldbp_n_t = len(ldbp_valid_t)

    rsbp_valid_t = df_main['R-SBP_Crit'].dropna()
    rsbp_mean_t = rsbp_valid_t.mean() if len(rsbp_valid_t) > 0 else np.nan
    rsbp_std_t = rsbp_valid_t.std() if len(rsbp_valid_t) > 0 else np.nan
    rsbp_n_t = len(rsbp_valid_t)

    rdbp_valid_t = df_main['R-DBP_Crit'].dropna()
    rdbp_mean_t = rdbp_valid_t.mean() if len(rdbp_valid_t) > 0 else np.nan
    rdbp_std_t = rdbp_valid_t.std() if len(rdbp_valid_t) > 0 else np.nan
    rdbp_n_t = len(rdbp_valid_t)

    # 总体
    age_valid_t = df_main['AGE_group'].dropna()
    male_valid_t = df_main['Gender_male'].dropna()
    bmi_valid_t = df_main['BMI_Crit'].dropna()
    hba1c_valid_t = df_main['HbA1c_Crit'].dropna()
    
    tc_dist_t = df_main['TC_category'].dropna().value_counts().to_dict()
    tg_dist_t = df_main['TG_category'].dropna().value_counts().to_dict()
    ldl_dist_t = df_main['LDL_category'].dropna().value_counts().to_dict()
    memory_dist_t = df_main['用药记忆情况_category'].dropna().value_counts().to_dict()



    
    table1_data.insert(0, {
        '服药习惯': '总体',
        '样本量': len(df_main),
        '年龄（每5岁分组）': '; '.join([f"{k}:{v}" for k, v in age_valid_t.value_counts().sort_index().items()]) + f" (n={len(age_valid_t)})",
        '性别（男性比例）': f"{int(male_valid_t.sum())} ({male_valid_t.sum()/len(male_valid_t)*100:.1f}%) (n={len(male_valid_t)})",
        'BMI': f"{bmi_valid_t.mean():.1f} ± {bmi_valid_t.std():.1f} (n={len(bmi_valid_t)})",
        'HbA1c': f"{hba1c_valid_t.mean():.1f} ± {hba1c_valid_t.std():.1f} (n={len(hba1c_valid_t)})",
        'L-SBP': f"{lsbp_mean_t:.1f} ± {lsbp_std_t:.1f} (n={lsbp_n_t})" if not np.isnan(lsbp_mean_t) else 'N/A',
        'L-DBP': f"{ldbp_mean_t:.1f} ± {ldbp_std_t:.1f} (n={ldbp_n_t})" if not np.isnan(ldbp_mean_t) else 'N/A',
        'R-SBP': f"{rsbp_mean_t:.1f} ± {rsbp_std_t:.1f} (n={rsbp_n_t})" if not np.isnan(rsbp_mean_t) else 'N/A',
        'R-DBP': f"{rdbp_mean_t:.1f} ± {rdbp_std_t:.1f} (n={rdbp_n_t})" if not np.isnan(rdbp_mean_t) else 'N/A',
        'TC_category': '; '.join([f"{k}:{v}" for k, v in tc_dist_t.items()]),
        'TG_category': '; '.join([f"{k}:{v}" for k, v in tg_dist_t.items()]),
        'LDL_category': '; '.join([f"{k}:{v}" for k, v in ldl_dist_t.items()]),
        '用药记忆情况': '; '.join([f"{k}:{v}" for k, v in memory_dist_t.items()])
    })

    
    table1_df = pd.DataFrame(table1_data)
    
    # 保存
    table1_path = os.path.join(output_dir, f'表1_服药习惯分组特征_{timestamp}.csv')
    table1_df.to_csv(table1_path, index=False, encoding='utf-8-sig')
    print(f"\n表1已保存到: {table1_path}")
    print("\n表1预览:")
    print(table1_df.to_string(index=False))
    
    return table1_df


# ==================== 核心分析：服药习惯的多因素Logistic回归 ====================
def generate_multivariable_logistic(df):
    """
    以服药习惯（空腹服药=1，餐后服药=0）为因变量
    分析各因素与空腹服药的关联
    """
    
    print("\n" + "="*60)
    print("核心分析：服药习惯的多因素Logistic回归")
    print("（因变量：空腹服药=1，餐后服药=0）")
    print("="*60)
    
    # 筛选主要分析人群
    df_analysis = df[df['服药习惯_category'].isin(['空腹服药', '餐后服药'])].copy()
    
    # 因变量
    y = df_analysis['med_fasting']
    
    results_data = []
    
    # ===== 模型1：单因素分析（每个变量单独分析） =====
    print("\n--- 单因素Logistic回归 ---")
    
    univariate_vars = [
        ('AGE_group', '年龄（每5岁分组）'),
        ('Gender_male', '性别（男性 vs 女性）'),
        ('BMI_Crit', 'BMI（连续）'),
        ('HbA1c_Crit', 'HbA1c（连续）'),
        ('L-SBP_Crit', '左收缩压（每10mmHg）'),
        ('L-DBP_Crit', '左舒张压（每10mmHg）'),
        ('R-SBP_Crit', '右收缩压（每10mmHg）'),
        ('R-DBP_Crit', '右舒张压（每10mmHg）'),
        ('用药记忆情况_category', '用药记忆情况（分类）')
    ]
    
    for var, label in univariate_vars:
        if var not in df_analysis.columns:
            continue
        
        valid = y.notna() & df_analysis[var].notna()
        y_clean = y[valid]
        
        # ===== 总样本量阈值10 =====
        if len(y_clean) < 10:
            print(f"  {label}: 样本量={len(y_clean)}<10，跳过")
            continue
        
        try:
            if df_analysis[var].dtype == 'object' or df_analysis[var].dtype.name == 'category':
                dummies = pd.get_dummies(df_analysis.loc[valid, var], prefix=label[:4])
                dummies = dummies.drop([c for c in dummies.columns if 'nan' in c.lower()], axis=1, errors='ignore')
                # ===== 修改点2：增加事件数为0的检测 =====
                # 对每个类别检查：该类别中因变量=1（空腹服药）的计数是否为0
                for col in list(dummies.columns):
                    # 获取该类别对应的因变量值
                    mask = dummies[col] == 1
                    event_count = y_clean[mask].sum()        # 空腹服药（事件=1）的计数
                    non_event_count = (~y_clean[mask].astype(bool)).sum() if mask.any() else 0  # 实际上更准确
                    # 简化：检查该类别中事件（med_fasting=1）的数量
                    if mask.any():
                        events_in_cat = y_clean[mask].sum()  # 空腹服药人数
                        non_events_in_cat = (~y_clean[mask].astype(bool)).sum()  # 餐后服药人数
                    else:
                        events_in_cat = 0
                        non_events_in_cat = 0
                    
                    # 如果该类别中事件数为0或非事件数为0（完全分离），跳过该类别
                    if events_in_cat == 0 or non_events_in_cat == 0:
                        print(f"  {label} ({col}): 事件数为0或非事件数为0，跳过该类别（完全分离）")
                        dummies = dummies.drop(col, axis=1)
                        continue
                    
                    # 样本量<10的类别也跳过
                    if dummies[col].sum() < 10:
                        print(f"  {label} ({col}): 样本量={int(dummies[col].sum())}<10，跳过该类别")
                        dummies = dummies.drop(col, axis=1)
                
                if dummies.shape[1] > 0:
                    # 以样本量最大的类别为参照
                    cat_counts = dummies.sum().sort_values(ascending=False)
                    ref_cat = cat_counts.index[0]
                    dummies = dummies.drop(ref_cat, axis=1)
                
                if dummies.shape[1] == 0:
                    print(f"  {label}: 虚拟变量转换后无有效列")
                    continue
                
                x_clean = sm.add_constant(dummies.astype(float))
                model = sm.Logit(y_clean, x_clean)
                result = model.fit(disp=0)
                
                for param in result.params.index:
                    if param == 'const':
                        continue
                    coef = result.params[param]
                    se = result.bse[param]
                    p_val = result.pvalues[param]
                    or_val = np.exp(coef)
                    ci_lower = np.exp(coef - 1.96 * se)
                    ci_upper = np.exp(coef + 1.96 * se)
                    
                    print(f"  {label} ({param}): OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.3f}, n={len(y_clean)}")
                    
                    results_data.append({
                        '分析类型': '单因素分析',
                        '变量': f"{label} ({param})",
                        'OR值': f"{or_val:.3f}",
                        '95%CI': f"({ci_lower:.3f}~{ci_upper:.3f})",
                        'P值': f"{p_val:.3f}",
                        '样本量': len(y_clean)
                    })
            else:
                # 连续变量：直接处理
                x_data = df_analysis.loc[valid, var]
                # 血压变量：转换为每10mmHg为单位
                if var in ['L-SBP_Crit', 'L-DBP_Crit', 'R-SBP_Crit', 'R-DBP_Crit']:
                    x_data = x_data / 10
                x_clean = sm.add_constant(x_data)
                model = sm.Logit(y_clean, x_clean)
                result = model.fit(disp=0)
                
                coef = result.params.iloc[1]
                se = result.bse.iloc[1]
                p_val = result.pvalues.iloc[1]
                or_val = np.exp(coef)
                ci_lower = np.exp(coef - 1.96 * se)
                ci_upper = np.exp(coef + 1.96 * se)
                
                print(f"  {label}: OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.3f}, n={len(y_clean)}")
                
                results_data.append({
                    '分析类型': '单因素分析',
                    '变量': label,
                    'OR值': f"{or_val:.3f}",
                    '95%CI': f"({ci_lower:.3f}~{ci_upper:.3f})",
                    'P值': f"{p_val:.3f}",
                    '样本量': len(y_clean)
                })
        except Exception as e:
            print(f"  {label}: 分析失败 - {e}")

    
    # ===== 模型2：多因素模型（调整年龄、性别） =====
    print("\n--- 多因素模型1：调整年龄和性别 ---")
    
    X_multi1 = pd.DataFrame()
    # 年龄分组虚拟变量（每5岁）
    age_dummies = pd.get_dummies(df_analysis['AGE_group'], prefix='年龄')
    age_dummies = age_dummies.drop([c for c in age_dummies.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    # ===== 修改点：阈值从5改为50，与模型2保持一致 =====
    for col in list(age_dummies.columns):
        if age_dummies[col].sum() < 50:
            age_dummies = age_dummies.drop(col, axis=1)

    if age_dummies.shape[1] > 0:
        # 以样本量最大的年龄组为参照
        cat_counts = age_dummies.sum().sort_values(ascending=False)
        ref_cat = cat_counts.index[0]
        age_dummies = age_dummies.drop(ref_cat, axis=1)
    for col in age_dummies.columns:
        X_multi1[col] = age_dummies[col].astype(int)
    X_multi1['性别_男性'] = df_analysis['Gender_male']

    
    valid_m1 = y.notna() & X_multi1.notna().all(axis=1)
    y_m1 = y[valid_m1]
    X_m1 = sm.add_constant(X_multi1[valid_m1])
    
    if len(y_m1) >= 20:
        try:
            model_m1 = sm.Logit(y_m1, X_m1)
            result_m1 = model_m1.fit(disp=0)
            
            for param in result_m1.params.index:
                if param == 'const':
                    continue
                coef = result_m1.params[param]
                se = result_m1.bse[param]
                p_val = result_m1.pvalues[param]
                or_val = np.exp(coef)
                ci_lower = np.exp(coef - 1.96 * se)
                ci_upper = np.exp(coef + 1.96 * se)
                
                print(f"  {param}: OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.3f}")
                
                results_data.append({
                    '分析类型': '多因素模型1（调整年龄+性别）',
                    '变量': param,
                    'OR值': f"{or_val:.3f}",
                    '95%CI': f"({ci_lower:.3f}~{ci_upper:.3f})",
                    'P值': f"{p_val:.3f}",
                    '样本量': len(y_m1)
                })
            
            results_data.append({
                '分析类型': '多因素模型1（调整年龄+性别）',
                '变量': 'AIC',
                'OR值': f"{result_m1.aic:.1f}",
                '95%CI': '',
                'P值': '',
                '样本量': len(y_m1)
            })
        except Exception as e:
            print(f"  多因素模型1拟合失败: {e}")
    
    # ===== 模型2：多因素模型（调整年龄、性别、BMI分组、HbA1c分组、TG、LDL） =====
    print("\n--- 多因素模型2：调整年龄、性别、BMI分组、HbA1c分组、TG、LDL ---")
    
    X_multi2 = pd.DataFrame()
    
    # ---- 年龄分组虚拟变量（每5岁），阈值改为50 ----
    age_dummies2 = pd.get_dummies(df_analysis['AGE_group'], prefix='年龄')
    age_dummies2 = age_dummies2.drop([c for c in age_dummies2.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    # 删除样本量<50的年龄组
    for col in list(age_dummies2.columns):
        if age_dummies2[col].sum() < 50:
            age_dummies2 = age_dummies2.drop(col, axis=1)
    # 以样本量最大的年龄组为参照
    if age_dummies2.shape[1] > 0:
        cat_counts2 = age_dummies2.sum().sort_values(ascending=False)
        ref_cat2 = cat_counts2.index[0]
        age_dummies2 = age_dummies2.drop(ref_cat2, axis=1)
    for col in age_dummies2.columns:
        X_multi2[col] = age_dummies2[col].astype(int)
    
    # ---- 性别 ----
    X_multi2['性别_男性'] = df_analysis['Gender_male']
    
    # ---- BMI分组（分类变量） ----
    bmi_dummies = pd.get_dummies(df_analysis['BMI_group'], prefix='BMI')
    bmi_dummies = bmi_dummies.drop([c for c in bmi_dummies.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    for col in list(bmi_dummies.columns):
        if bmi_dummies[col].sum() < 50:
            bmi_dummies = bmi_dummies.drop(col, axis=1)
    if bmi_dummies.shape[1] > 0:
        cat_counts_bmi = bmi_dummies.sum().sort_values(ascending=False)
        ref_cat_bmi = cat_counts_bmi.index[0]
        bmi_dummies = bmi_dummies.drop(ref_cat_bmi, axis=1)
    for col in bmi_dummies.columns:
        X_multi2[col] = bmi_dummies[col].astype(int)
    
    # ---- HbA1c分组（分类变量） ----
    hba1c_dummies = pd.get_dummies(df_analysis['HbA1c_group'], prefix='HbA1c')
    hba1c_dummies = hba1c_dummies.drop([c for c in hba1c_dummies.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    for col in list(hba1c_dummies.columns):
        if hba1c_dummies[col].sum() < 50:
            hba1c_dummies = hba1c_dummies.drop(col, axis=1)
    if hba1c_dummies.shape[1] > 0:
        cat_counts_hba1c = hba1c_dummies.sum().sort_values(ascending=False)
        ref_cat_hba1c = cat_counts_hba1c.index[0]
        hba1c_dummies = hba1c_dummies.drop(ref_cat_hba1c, axis=1)
    for col in hba1c_dummies.columns:
        X_multi2[col] = hba1c_dummies[col].astype(int)
    
    # ---- TG和LDL分类变量（阈值改为50） ----
    for cat_col, prefix in [('TG_category', 'TG'), 
                             ('LDL_category', 'LDL')]:
        if cat_col in df_analysis.columns:
            dummies = pd.get_dummies(df_analysis[cat_col], prefix=prefix)
            # 删除NaN列
            dummies = dummies.drop([c for c in dummies.columns if 'nan' in c.lower()], axis=1, errors='ignore')
            # 删除样本量<50的稀疏类别（避免奇异矩阵）
            for col in list(dummies.columns):
                if dummies[col].sum() < 50:
                    dummies = dummies.drop(col, axis=1)
            # 以样本量最大的类别为参照
            if dummies.shape[1] > 0:
                cat_counts_dum = dummies.sum().sort_values(ascending=False)
                ref_cat_dum = cat_counts_dum.index[0]
                dummies = dummies.drop(ref_cat_dum, axis=1)
            for col in dummies.columns:
                X_multi2[col] = dummies[col].astype(int)
    
    # ---- 检查是否有足够变量进入模型 ----
    if X_multi2.shape[1] == 0:
        print("  警告：所有分类变量均因样本量<50被排除，模型2无法运行")
    else:
        valid_m2 = y.notna() & X_multi2.notna().all(axis=1)
        y_m2 = y[valid_m2]
        X_m2 = sm.add_constant(X_multi2[valid_m2])
        
        print(f"  有效样本量: {len(y_m2)}")
        print(f"  自变量: {list(X_m2.columns)}")
        
        if len(y_m2) >= 20 and X_m2.shape[1] < len(y_m2):
            try:
                model_m2 = sm.Logit(y_m2, X_m2)
                result_m2 = model_m2.fit(disp=0)
                
                for param in result_m2.params.index:
                    if param == 'const':
                        continue
                    coef = result_m2.params[param]
                    se = result_m2.bse[param]
                    p_val = result_m2.pvalues[param]
                    or_val = np.exp(coef)
                    ci_lower = np.exp(coef - 1.96 * se)
                    ci_upper = np.exp(coef + 1.96 * se)
                    
                    print(f"  {param}: OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.3f}")
                    
                    results_data.append({
                        '分析类型': '多因素模型2（调整年龄+性别+BMI分组+HbA1c分组+TG+LDL）',
                        '变量': param,
                        'OR值': f"{or_val:.3f}",
                        '95%CI': f"({ci_lower:.3f}~{ci_upper:.3f})",
                        'P值': f"{p_val:.3f}",
                        '样本量': len(y_m2)
                    })
                
                results_data.append({
                    '分析类型': '多因素模型2（调整年龄+性别+BMI分组+HbA1c分组+TG+LDL）',
                    '变量': 'AIC',
                    'OR值': f"{result_m2.aic:.1f}",
                    '95%CI': '',
                    'P值': '',
                    '样本量': len(y_m2)
                })
            except Exception as e:
                print(f"  多因素模型2拟合失败: {e}")
        else:
            print(f"  样本量不足或变量过多，跳过模型2拟合")
    
    # 保存结果
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(output_dir, f'服药习惯Logistic回归_{timestamp}.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\nLogistic回归结果已保存到: {results_path}")
        return results_df
    
    return None

def check_complete_separation(y, X, min_samples=10):
    problematic_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            continue
        # ===== 新增：虚拟变量样本量<10直接排除 =====
        if X[col].nunique() == 2:  # 0/1虚拟变量
            n_ones = X[col].sum()
            if n_ones < min_samples:
                problematic_cols.append(col)
                print(f"  准完全分离检测: {col} 样本量={int(n_ones)}<{min_samples}，排除")
                continue
        # 检查严格完全分离
        mask_1 = X[col] == 1
        if mask_1.any():
            y_when_1 = y[mask_1]
            if y_when_1.nunique() == 1:
                if col not in problematic_cols:
                    problematic_cols.append(col)
                    print(f"  完全分离检测: {col} 在取值=1时因变量全部为{y_when_1.iloc[0]}")
        mask_0 = X[col] == 0
        if mask_0.any():
            y_when_0 = y[mask_0]
            if y_when_0.nunique() == 1 and col not in problematic_cols:
                problematic_cols.append(col)
                print(f"  完全分离检测: {col} 在取值=0时因变量全部为{y_when_0.iloc[0]}")
    return problematic_cols

# ==================== 亚组分析 ====================
def generate_subgroup_analysis(df):
    """按性别分层，分析各因素与服药习惯的关联"""
    
    print("\n" + "="*60)
    print("亚组分析：按性别分层")
    print("="*60)
    
    df_main = df[df['服药习惯_category'].isin(['空腹服药', '餐后服药'])].copy()
    y = df_main['med_fasting']
    
    subgroup_results = []
    
    # 男性亚组
    df_male = df_main[df_main['Gender_male'] == 1]
    y_male = y[df_main['Gender_male'] == 1]
    print(f"\n男性亚组样本量: {len(df_male)}")
    
    if len(df_male) >= 30:
        X_male = pd.DataFrame()
        age_dummies_male = pd.get_dummies(df_male['AGE_group'], prefix='年龄')
        age_dummies_male = age_dummies_male.drop([c for c in age_dummies_male.columns if 'nan' in c.lower()], axis=1, errors='ignore')
        # ===== 修改点：年龄组阈值统一为50 =====
        for col in list(age_dummies_male.columns):
            if age_dummies_male[col].sum() < 50:
                age_dummies_male = age_dummies_male.drop(col, axis=1)
        if age_dummies_male.shape[1] > 0:
            cat_counts_male = age_dummies_male.sum().sort_values(ascending=False)
            ref_cat_male = cat_counts_male.index[0]
            age_dummies_male = age_dummies_male.drop(ref_cat_male, axis=1)
        for col in age_dummies_male.columns:
            X_male[col] = age_dummies_male[col].astype(int)
        X_male['HbA1c'] = df_male['HbA1c_Crit']
        X_male['BMI'] = df_male['BMI_Crit']
        
        valid_male = y_male.notna() & X_male.notna().all(axis=1)
        y_male_clean = y_male[valid_male]
        X_male_clean_df = X_male[valid_male]  # 先不加const，用于检测
        
        # ===== 完全分离检测 =====
        problematic_cols = check_complete_separation(y_male_clean, X_male_clean_df)
        if problematic_cols:
            print(f"  男性亚组: 发现完全分离变量 {problematic_cols}，已自动排除")
            X_male_clean_df = X_male_clean_df.drop(columns=problematic_cols)
        
        X_male_clean = sm.add_constant(X_male_clean_df)
        
        if len(y_male_clean) >= 20 and X_male_clean.shape[1] >= 2:  # 至少const+1个变量
            try:
                model_male = sm.Logit(y_male_clean, X_male_clean)
                result_male = model_male.fit(disp=0)
                
                for param in result_male.params.index:
                    if param == 'const':
                        continue
                    coef = result_male.params[param]
                    se = result_male.bse[param]
                    p_val = result_male.pvalues[param]
                    or_val = np.exp(coef)
                    ci_lower = np.exp(coef - 1.96 * se)
                    ci_upper = np.exp(coef + 1.96 * se)
                    
                    print(f"  男性-{param}: OR={or_val:.3f} ({ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.3f}")
                    
                    subgroup_results.append({
                        '亚组': '男性',
                        '变量': param,
                        'OR值': f"{or_val:.3f}",
                        '95%CI': f"({ci_lower:.3f}~{ci_upper:.3f})",
                        'P值': f"{p_val:.3f}",
                        '样本量': len(y_male_clean)
                    })
            except Exception as e:
                print(f"  男性亚组分析失败: {e}")
    
    # 女性亚组
    df_female = df_main[df_main['Gender_male'] == 0]
    y_female = y[df_main['Gender_male'] == 0]
    print(f"\n女性亚组样本量: {len(df_female)}")
    
    if len(df_female) >= 30:
        X_female = pd.DataFrame()
        # 年龄分组虚拟变量（每5岁）
        age_dummies_female = pd.get_dummies(df_female['AGE_group'], prefix='年龄')
        age_dummies_female = age_dummies_female.drop([c for c in age_dummies_female.columns if 'nan' in c.lower()], axis=1, errors='ignore')
        # ===== 排除样本量<10的年龄组（防止准完全分离）=====
        for col in list(age_dummies_female.columns):
            if age_dummies_female[col].sum() < 10:
                print(f"  女性亚组: 年龄组 {col} 样本量={int(age_dummies_female[col].sum())}<10，排除（准完全分离）")
                age_dummies_female = age_dummies_female.drop(col, axis=1)
        # 以样本量最大的年龄组为参照
        if age_dummies_female.shape[1] > 0:
            cat_counts_female = age_dummies_female.sum().sort_values(ascending=False)
            ref_cat_female = cat_counts_female.index[0]
            age_dummies_female = age_dummies_female.drop(ref_cat_female, axis=1)
        for col in age_dummies_female.columns:
            X_female[col] = age_dummies_female[col].astype(int)
        # 女性亚组中HbA1c缺失较多，去掉HbA1c，只用年龄组+BMI
        # X_female['HbA1c'] = df_female['HbA1c_Crit']
        X_female['BMI'] = df_female['BMI_Crit']
        
        
        # 注意：不加入Gender_male（在女性亚组中全部为0，是常数）
        # 注意：不加入 Gender_male（在亚组中为常数）
        #女性亚组因可纳入统计的糖化血红蛋白例数低于20，已该组不纳入亚组分析
        valid_female = y_female.notna() & X_female.notna().all(axis=1)
        y_female_clean = y_female[valid_female]
        X_female_clean_df = X_female[valid_female]
        # 完全分离检测
        problematic_cols = check_complete_separation(y_female_clean, X_female_clean_df)
        if problematic_cols:
            print(f"  女性亚组: 发现完全分离变量 {problematic_cols}，已自动排除")
            X_female_clean_df = X_female_clean_df.drop(columns=problematic_cols)
        X_female_clean = sm.add_constant(X_female_clean_df)
        # ===== 检查剩余变量数是否足够 =====
        if len(y_female_clean) >= 20 and X_female_clean.shape[1] >= 2:
            try:
                model_female = sm.Logit(y_female_clean, X_female_clean)
                result_female = model_female.fit(disp=0)
                
                for param in result_female.params.index:
                    if param == 'const':
                        continue
                    coef = result_female.params[param]
                    se = result_female.bse[param]
                    p_val = result_female.pvalues[param]
                    or_val = np.exp(coef)
                    ci_lower = np.exp(coef - 1.96 * se)
                    ci_upper = np.exp(coef + 1.96 * se)
                    
                    print(f"  女性-{param}: OR={or_val:.3f} ({ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.3f}")
                    
                    subgroup_results.append({
                        '亚组': '女性',
                        '变量': param,
                        'OR值': f"{or_val:.3f}",
                        '95%CI': f"({ci_lower:.3f}~{ci_upper:.3f})",
                        'P值': f"{p_val:.3f}",
                        '样本量': len(y_female_clean)
                    })
            except Exception as e:
                print(f"  女性亚组分析失败: {e}")
                print(f"  女性亚组: 排除完全分离变量后，剩余变量仍存在共线性问题，跳过分析")
        else:
            print(f"  女性亚组: 排除完全分离变量后，剩余自变量={X_female_clean.shape[1]-1}个，不足以进行分析，跳过")
    
    # 保存
    if subgroup_results:
        subgroup_df = pd.DataFrame(subgroup_results)
        subgroup_path = os.path.join(output_dir, f'亚组分析_按性别分层_{timestamp}.csv')
        subgroup_df.to_csv(subgroup_path, index=False, encoding='utf-8-sig')
        print(f"\n亚组分析结果已保存到: {subgroup_path}")
        return subgroup_df
    
    return None


# ==================== 左右侧血压分析 ====================
def generate_bp_numeric_analysis(df):
    """左右侧血压数值变量统计分析"""
    
    print("\n" + "="*60)
    print("左右侧血压数值变量统计分析")
    print("="*60)
    
    df_main = df[df['服药习惯_category'].isin(['空腹服药', '餐后服药'])].copy()
    bp_cols = ['L-SBP_Crit', 'L-DBP_Crit', 'R-SBP_Crit', 'R-DBP_Crit']
    
    # 描述性统计
    desc = df_main[bp_cols].describe().round(1)
    print("\n描述性统计:")
    print(desc)
    
    # 左右侧配对t检验
    pairs = [
        ('L-SBP_Crit', 'R-SBP_Crit', '收缩压(SBP)左右侧比较'),
        ('L-DBP_Crit', 'R-DBP_Crit', '舒张压(DBP)左右侧比较')
    ]
    
    ttest_results = []
    for col1, col2, label in pairs:
        pair_df = df_main[[col1, col2]].dropna()
        if len(pair_df) > 10:
            t_stat, p_val = stats.ttest_rel(pair_df[col1], pair_df[col2])
            mean_diff = (pair_df[col1] - pair_df[col2]).mean()
            std_diff = (pair_df[col1] - pair_df[col2]).std()
            ttest_results.append({
                '比较项目': label,
                '左侧均值±标准差': f"{pair_df[col1].mean():.1f} ± {pair_df[col1].std():.1f}",
                '右侧均值±标准差': f"{pair_df[col2].mean():.1f} ± {pair_df[col2].std():.1f}",
                '均值差值(左-右)': f"{mean_diff:.1f}",
                '差值标准差': f"{std_diff:.1f}",
                't值': f"{t_stat:.3f}",
                'P值': f"{p_val:.3f}",
                '有效样本量': len(pair_df)
            })
            print(f"\n{label}:")
            print(f"  左侧: {pair_df[col1].mean():.1f} ± {pair_df[col1].std():.1f}")
            print(f"  右侧: {pair_df[col2].mean():.1f} ± {pair_df[col2].std():.1f}")
            print(f"  配对t检验: t={t_stat:.3f}, P={p_val:.3f}")
    
    # 保存
    desc_path = os.path.join(output_dir, f'左右侧血压描述统计_{timestamp}.csv')
    desc.to_csv(desc_path, encoding='utf-8-sig')
    
    if ttest_results:
        ttest_df = pd.DataFrame(ttest_results)
        ttest_path = os.path.join(output_dir, f'左右侧血压配对t检验_{timestamp}.csv')
        ttest_df.to_csv(ttest_path, index=False, encoding='utf-8-sig')
        print(f"\n配对t检验结果已保存到: {ttest_path}")
        return desc, ttest_df
    
    return desc, None

# ==================== 服药习惯与血压控制程度分析 ====================
def sbp_control_level(sbp):
    """SBP控制程度分类"""
    if pd.isna(sbp):
        return np.nan
    if sbp <= 159:
        return '轻度控制不良'
    elif sbp <= 179:
        return '中度控制不良'
    else:
        return '重度控制不良'

def dbp_control_level(dbp):
    """DBP控制程度分类"""
    if pd.isna(dbp):
        return np.nan
    if dbp <= 99:
        return '轻度控制不良'
    elif dbp <= 109:
        return '中度控制不良'
    else:
        return '重度控制不良'

def generate_bp_control_analysis(df):
    """
    分析服药习惯（空腹服药 vs 餐后服药）与四列血压控制程度的关系
    血压控制程度分为：轻度控制不良、中度控制不良、重度控制不良
    """
    print("\n" + "="*60)
    print("服药习惯与血压控制程度分析")
    print("="*60)
    
    # 筛选主要分析人群
    df_main = df[df['服药习惯_category'].isin(['空腹服药', '餐后服药'])].copy()
    
    # 为四列血压创建控制程度分类
    bp_cols_info = [
        ('L-SBP_Crit', 'L-SBP_control', '左侧收缩压(SBP)', sbp_control_level),
        ('L-DBP_Crit', 'L-DBP_control', '左侧舒张压(DBP)', dbp_control_level),
        ('R-SBP_Crit', 'R-SBP_control', '右侧收缩压(SBP)', sbp_control_level),  # ✅ 修复
        ('R-DBP_Crit', 'R-DBP_control', '右侧舒张压(DBP)', dbp_control_level)
    ]
    
    for orig_col, new_col, label, func in bp_cols_info:
        if orig_col in df_main.columns:
            df_main[new_col] = df_main[orig_col].apply(func)
    
    # 定义控制程度的顺序
    control_order = ['轻度控制不良', '中度控制不良', '重度控制不良']
    
    all_results = []
    
    for orig_col, new_col, label, func in bp_cols_info:
        if new_col not in df_main.columns:
            continue
        
        print(f"\n--- {label} ---")
        
        # 交叉表：服药习惯 × 血压控制程度
        cross_tab = pd.crosstab(
            df_main['服药习惯_category'],
            df_main[new_col],
            margins=True,
            margins_name='合计'
        )
        # 重新排列列顺序
        available_cols = [c for c in control_order if c in cross_tab.columns]
        other_cols = [c for c in cross_tab.columns if c not in control_order and c != '合计']
        col_order = available_cols + other_cols + (['合计'] if '合计' in cross_tab.columns else [])
        cross_tab = cross_tab[col_order]
        
        print("\n交叉表（频数）:")
        print(cross_tab)
        
        # 百分比交叉表（行百分比）
        cross_tab_pct = pd.crosstab(
            df_main['服药习惯_category'],
            df_main[new_col],
            margins=True,
            margins_name='合计',
            normalize='index'
        ) * 100
        
        available_cols_pct = [c for c in control_order if c in cross_tab_pct.columns]
        other_cols_pct = [c for c in cross_tab_pct.columns if c not in control_order and c != '合计']
        col_order_pct = available_cols_pct + other_cols_pct + (['合计'] if '合计' in cross_tab_pct.columns else [])
        cross_tab_pct = cross_tab_pct[col_order_pct]
        
        print("\n交叉表（行百分比 %）:")
        print(cross_tab_pct.round(1))
        
        
        # 构建列联表（只保留有数据的列）
        cross_tab = pd.crosstab(df_main['服药习惯_category'], df_main[new_col])
        cross_tab = cross_tab[[c for c in control_order if c in cross_tab.columns]]

        # ===== 新增：如果"重度控制不良"样本量太少（期望频数<5），合并"中度"和"重度" =====
        if '重度控制不良' in cross_tab.columns and '中度控制不良' in cross_tab.columns:
            # 检查重度控制不良的期望频数
            row_sums_check = cross_tab.sum(axis=1).values
            col_sums_check = cross_tab.sum(axis=0).values
            total_check = cross_tab.values.sum()
            expected_check = np.outer(row_sums_check, col_sums_check) / total_check
            # 找到重度控制不良列的索引
            severe_idx = list(cross_tab.columns).index('重度控制不良')
            severe_expected = expected_check[:, severe_idx]
            if any(severe_expected < 5):
                print(f"  (注意：重度控制不良期望频数<5，已与中度控制不良合并)")
                cross_tab['中重度控制不良'] = cross_tab['中度控制不良'] + cross_tab['重度控制不良']
                cross_tab = cross_tab.drop(columns=['中度控制不良', '重度控制不良'])


        # 卡方检验
        cross_tab_for_chi = pd.crosstab(
            df_main['服药习惯_category'],
            df_main[new_col]
        )
        # 只保留有数据的列
        cross_tab_for_chi = cross_tab_for_chi[[c for c in control_order if c in cross_tab_for_chi.columns]]
        
        if cross_tab_for_chi.shape[0] >= 2 and cross_tab_for_chi.shape[1] >= 2:
            chi2, p_val, dof, expected = stats.chi2_contingency(cross_tab_for_chi)
            print(f"\n卡方检验: χ²={chi2:.3f}, P={p_val:.3f}, 自由度={dof}")
            
            # 检查期望频数
            expected_lt5 = (expected < 5).sum()
            total_cells = expected.size
            pct_lt5 = expected_lt5 / total_cells * 100
            
            if pct_lt5 > 20:
                # 超过20%的单元格期望频数<5，使用蒙特卡洛模拟计算精确p值
                print(f"  (注意：有{expected_lt5}个单元格期望频数<5（占总单元格的{pct_lt5:.1f}%），超过20%，卡方检验结果可能不可靠)")
                
                # 蒙特卡洛模拟计算精确p值（随机置换检验）
                n_simulations = 10000
                observed_stat = chi2
                
                # 获取原始分类数据用于置换检验
                med_fasting_series = df_main['服药习惯_category']
                bp_control_series = df_main[new_col]
                
                # 只保留有数据的行
                valid_mask = med_fasting_series.notna() & bp_control_series.notna()
                med_fasting_valid = med_fasting_series[valid_mask].values
                bp_control_valid = bp_control_series[valid_mask].values
                
                simulated_stats = []
                for _ in range(n_simulations):
                    # 随机置换服药习惯标签（保持行和列合计不变）
                    shuffled_med = np.random.permutation(med_fasting_valid)
                    # 重新计算列联表
                    sim_table = pd.crosstab(shuffled_med, bp_control_valid)
                    # 只保留有数据的列
                    sim_table = sim_table[[c for c in control_order if c in sim_table.columns]]
                    
                    if sim_table.shape[0] >= 2 and sim_table.shape[1] >= 2:
                        try:
                            sim_chi2, _, _, _ = stats.chi2_contingency(sim_table)
                            simulated_stats.append(sim_chi2)
                        except:
                            continue
                
                if len(simulated_stats) > 0:
                    p_sim = (np.sum(np.array(simulated_stats) >= observed_stat) + 1) / (len(simulated_stats) + 1)
                    print(f"  蒙特卡洛模拟精确检验: P={p_sim:.3f}（基于{len(simulated_stats)}次模拟）")
                else:
                    p_sim = np.nan
                    print(f"  蒙特卡洛模拟精确检验: 无法计算（数据不足）")
            elif expected_lt5 > 0:
                print(f"  (注意：有{expected_lt5}个单元格期望频数<5（占总单元格的{pct_lt5:.1f}%），卡方检验结果需谨慎解读)")
            else:
                print(f"  所有单元格期望频数均≥5，卡方检验结果可靠")

        else:
            chi2, p_val = np.nan, np.nan
            print("\n卡方检验：数据不足")

        
        # 保存结果
        for group in ['空腹服药', '餐后服药']:
            if group not in cross_tab.index:
                continue
            for level in control_order:
                if level not in cross_tab.columns:
                    continue
                count = cross_tab.loc[group, level]
                pct_val = cross_tab_pct.loc[group, level] if level in cross_tab_pct.columns else np.nan
                
                all_results.append({
                    '血压指标': label,
                    '服药习惯': group,
                    '控制程度': level,
                    '例数': int(count),
                    '百分比(%)': round(pct_val, 1) if not pd.isna(pct_val) else '',
                    '卡方值': round(chi2, 3) if not pd.isna(chi2) else '',
                    'P值': round(p_val, 4) if not pd.isna(p_val) else ''
                })
        
        # 总体行
        if '合计' in cross_tab.index:
            for level in control_order:
                if level not in cross_tab.columns:
                    continue
                count_total = cross_tab.loc['合计', level]
                all_results.append({
                    '血压指标': label,
                    '服药习惯': '合计',
                    '控制程度': level,
                    '例数': int(count_total),
                    '百分比(%)': '',
                    '卡方值': round(chi2, 3) if not pd.isna(chi2) else '',
                    'P值': round(p_val, 4) if not pd.isna(p_val) else ''
                })
    
    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(output_dir, f'服药习惯与血压控制程度分析_{timestamp}.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n血压控制程度分析结果已保存到: {results_path}")
        print("\n结果预览:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    return None

# ==================== 用药记忆情况分析 ====================
def generate_medication_memory_analysis(df):
    """
    分析用药记忆情况与服药习惯的关系
    仅纳入"可提供具体用药记录"和"仅可提供服药种类"两类进行对比
    1. 2×2交叉表分析（卡方检验）
    2. 用药记忆情况+性别+年龄的多因素Logistic回归（以服药习惯为因变量）
       ——以"仅可提供服药种类"为参照
    """
    
    print("\n" + "="*60)
    print("用药记忆情况与服药习惯分析")
    print("（仅对比：可提供具体用药记录 vs 仅可提供服药种类）")
    print("="*60)
    
    # 筛选主要分析人群（空腹服药 vs 餐后服药）
    df_main = df[df['服药习惯_category'].isin(['空腹服药', '餐后服药'])].copy()
    
    # ★★★ 关键修改：仅保留"可提供具体用药记录"和"仅可提供服药种类" ★★★
    target_memory_cats = ['可提供具体用药记录', '仅可提供服药种类']
    df_main = df_main[df_main['用药记忆情况_category'].isin(target_memory_cats)].copy()
    
    n_total = len(df_main)
    n_record = (df_main['用药记忆情况_category'] == '可提供具体用药记录').sum()
    n_type = (df_main['用药记忆情况_category'] == '仅可提供服药种类').sum()
    print(f"\n纳入样本量: {n_total}例")
    print(f"  可提供具体用药记录: {n_record}例")
    print(f"  仅可提供服药种类: {n_type}例")
    print(f"  排除: 完全不清楚用药情况 + 缺失值")
    
    all_results = []
    
    # ===== 1. 2×2交叉表分析 =====
    print("\n--- 2×2交叉表：用药记忆情况 × 服药习惯 ---")
    
    cross_tab = pd.crosstab(
        df_main['用药记忆情况_category'],
        df_main['服药习惯_category'],
        margins=True,
        margins_name='合计'
    )
    print("\n交叉表（频数）:")
    print(cross_tab)
    
    # 行百分比
    cross_tab_pct = pd.crosstab(
        df_main['用药记忆情况_category'],
        df_main['服药习惯_category'],
        margins=True,
        margins_name='合计',
        normalize='index'
    ) * 100
    print("\n交叉表（行百分比 %）:")
    print(cross_tab_pct.round(1))
    
    # 列百分比
    cross_tab_col_pct = pd.crosstab(
        df_main['用药记忆情况_category'],
        df_main['服药习惯_category'],
        margins=True,
        margins_name='合计',
        normalize='columns'
    ) * 100
    print("\n交叉表（列百分比 %）:")
    print(cross_tab_col_pct.round(1))
    
    # 卡方检验（2×2表，自动使用Yates校正）
    cross_tab_for_chi = pd.crosstab(
        df_main['用药记忆情况_category'],
        df_main['服药习惯_category']
    )
    
    if cross_tab_for_chi.shape[0] >= 2 and cross_tab_for_chi.shape[1] >= 2:
        # 2×2表使用Yates连续性校正
        chi2, p_val, dof, expected = stats.chi2_contingency(cross_tab_for_chi, correction=True)
        print(f"\n卡方检验（Yates连续性校正）: χ²={chi2:.3f}, P={p_val:.4f}, 自由度={dof}")
        
        # 同时报告未校正的卡方值
        chi2_uncorrected, p_uncorrected, _, _ = stats.chi2_contingency(cross_tab_for_chi, correction=False)
        print(f"卡方检验（Pearson未校正）: χ²={chi2_uncorrected:.3f}, P={p_uncorrected:.4f}")
        
        # Fisher精确检验（2×2表推荐）
        odds_ratio, p_fisher = stats.fisher_exact(cross_tab_for_chi)
        print(f"Fisher精确检验: OR={odds_ratio:.3f}, P={p_fisher:.4f}")
        
        # 检查期望频数
        expected_lt5 = (expected < 5).sum()
        if expected_lt5 > 0:
            print(f"  (注意：有{expected_lt5}个单元格期望频数<5，建议以Fisher精确检验结果为准)")
        else:
            print(f"  所有单元格期望频数均≥5，卡方检验结果可靠")
        
        # 保存交叉表结果
        for memory_cat in cross_tab.index:
            if memory_cat == '合计':
                continue
            for med_cat in ['空腹服药', '餐后服药']:
                if med_cat not in cross_tab.columns:
                    continue
                count = cross_tab.loc[memory_cat, med_cat]
                row_pct = cross_tab_pct.loc[memory_cat, med_cat] if med_cat in cross_tab_pct.columns else np.nan
                col_pct = cross_tab_col_pct.loc[memory_cat, med_cat] if med_cat in cross_tab_col_pct.columns else np.nan
                
                all_results.append({
                    '分析类型': '交叉表分析（可提供具体用药记录 vs 仅可提供服药种类）',
                    '用药记忆情况': memory_cat,
                    '服药习惯': med_cat,
                    '例数': int(count),
                    '行百分比(%)': round(row_pct, 1) if not pd.isna(row_pct) else '',
                    '列百分比(%)': round(col_pct, 1) if not pd.isna(col_pct) else '',
                    '卡方值': f"{chi2:.3f}(Yates)",
                    'P值': round(p_fisher, 4)
                })
    else:
        print("\n卡方检验：数据不足")
    
    # ===== 2. 多因素Logistic回归：用药记忆情况+性别+年龄 → 服药习惯 =====
    print("\n--- 多因素Logistic回归：用药记忆情况+性别+年龄 → 服药习惯 ---")
    print("（因变量：空腹服药=1，餐后服药=0）")
    print("（用药记忆情况：以'仅可提供服药种类'为参照）")
    
    y = df_main['med_fasting']
    
    # 构建自变量
    X_memory = pd.DataFrame()
    # 年龄分组虚拟变量（每5岁）
    age_dummies_mem = pd.get_dummies(df_main['AGE_group'], prefix='年龄')
    age_dummies_mem = age_dummies_mem.drop([c for c in age_dummies_mem.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    for col in list(age_dummies_mem.columns):
        if age_dummies_mem[col].sum() < 50:
            age_dummies_mem = age_dummies_mem.drop(col, axis=1)
    if age_dummies_mem.shape[1] > 0:
        cat_counts = age_dummies_mem.sum().sort_values(ascending=False)
        ref_cat = cat_counts.index[0]
        age_dummies_mem = age_dummies_mem.drop(ref_cat, axis=1)

    for col in age_dummies_mem.columns:
        X_memory[col] = age_dummies_mem[col].astype(int)
    X_memory['性别_男性'] = df_main['Gender_male']

    # ★★★ 关键修改：用药记忆情况——创建二分类变量，"仅可提供服药种类"=0（参照），"可提供具体用药记录"=1 ★★★
    df_main['memory_record'] = np.where(
        df_main['用药记忆情况_category'] == '可提供具体用药记录', 1, 0
    )
    X_memory['用药记忆_可提供具体记录(vs仅可提供服药种类)'] = df_main['memory_record'].astype(int)
    
    print(f"  用药记忆情况变量: 可提供具体用药记录=1, 仅可提供服药种类=0（参照）")
    
    # 有效样本
    valid_memory = y.notna() & X_memory.notna().all(axis=1)
    y_memory = y[valid_memory]
    X_memory_clean = sm.add_constant(X_memory[valid_memory])
    
    print(f"  有效样本量: {len(y_memory)}")
    print(f"  自变量: {list(X_memory_clean.columns)}")
    
    if len(y_memory) >= 20:
        try:
            model_memory = sm.Logit(y_memory, X_memory_clean)
            result_memory = model_memory.fit(disp=0)
            
            print(f"\n  模型整体检验:")
            print(f"  伪R² (McFadden): {result_memory.prsquared:.4f}")
            print(f"  AIC: {result_memory.aic:.1f}")
            
            print(f"\n  各自变量结果:")
            for param in result_memory.params.index:
                if param == 'const':
                    continue
                coef = result_memory.params[param]
                se = result_memory.bse[param]
                p_val = result_memory.pvalues[param]
                or_val = np.exp(coef)
                ci_lower = np.exp(coef - 1.96 * se)
                ci_upper = np.exp(coef + 1.96 * se)
                
                print(f"  {param}: OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.4f}")
                
                all_results.append({
                    '分析类型': '多因素Logistic回归（调整年龄+性别；参照=仅可提供服药种类）',
                    '用药记忆情况': '',
                    '服药习惯': param,
                    '例数': '',
                    '行百分比(%)': '',
                    '列百分比(%)': '',
                    '卡方值': '',
                    'OR值': f"{or_val:.3f}",                              # ✅ 新增
                    '95%CI': f"({ci_lower:.3f}~{ci_upper:.3f})",          # ✅ 新增
                    'P值': round(p_val, 4)
                })
            
            # AIC 行也补充空字段以保持列对齐
            all_results.append({
                '分析类型': '多因素Logistic回归（调整年龄+性别；参照=仅可提供服药种类）',
                '用药记忆情况': '',
                '服药习惯': 'AIC',
                '例数': '',
                '行百分比(%)': '',
                '列百分比(%)': '',
                '卡方值': f"{result_memory.aic:.1f}",
                'OR值': '',                                               # ✅ 新增
                '95%CI': '',                                              # ✅ 新增
                'P值': ''
            })
            
        except Exception as e:
            print(f"  多因素Logistic回归拟合失败: {e}")
    
    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(output_dir, f'用药记忆情况分析_{timestamp}.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n用药记忆情况分析结果已保存到: {results_path}")
        return results_df
    
    return None


# ==================== 左收缩压混杂因素深入分析 ====================
def generate_bp_confounding_analysis(df):
    """
    针对"左收缩压单因素显著、多因素不显著"的混杂原因进行深入分析
    包括：PSM、IPW、E-value、350例全调整模型
    左收缩压以每升高10mmHg为一个分析单元
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.linear_model import LogisticRegression
    
    print("\n" + "="*70)
    print("左收缩压与服药习惯：混杂因素深入分析")
    print("（左收缩压以每升高10mmHg为一个分析单元）")
    print("="*70)
    
    # 筛选主要分析人群
    df_main = df[df['服药习惯_category'].isin(['空腹服药', '餐后服药'])].copy()
    df_main['med_fasting'] = np.where(df_main['服药习惯_category'] == '空腹服药', 1, 0)
    
    # 创建每10mmHg的变量
    df_main['L-SBP_per10'] = df_main['L-SBP_Crit'] / 10.0
    
    # ===== 方法一：仅用有左收缩压数据的样本做多因素Logistic回归 =====
    print("\n" + "-"*70)
    print("方法一：仅用有左收缩压数据的样本做多因素Logistic回归")
    print("-"*70)
    
    # 筛选有左收缩压数据的样本
    bp_valid = df_main['L-SBP_Crit'].notna()
    df_bp = df_main.loc[bp_valid, ['med_fasting', 'L-SBP_per10', 'L-SBP_Crit', 'AGE_group', 'Gender_male', 'BMI_Crit', 'HbA1c_Crit']].copy()

    print(f"有左收缩压数据的样本量: {len(df_bp)}")
    print(f"  空腹服药: {(df_bp['med_fasting']==1).sum()}例")
    print(f"  餐后服药: {(df_bp['med_fasting']==0).sum()}例")
    
    # 模型A：单因素（左收缩压每10mmHg）
    y_bp = df_bp['med_fasting']
    X_bp1 = sm.add_constant(df_bp[['L-SBP_per10']])

    # 删除缺失值
    valid_idx = y_bp.notna() & X_bp1.notna().all(axis=1)
    y_bp_clean = y_bp[valid_idx]
    X_bp1_clean = X_bp1.loc[valid_idx]

    print(f"  有效样本量（删除缺失后）: {len(y_bp_clean)}")

    model_bp1 = sm.Logit(y_bp_clean, X_bp1_clean).fit(disp=0)  # ✅ 只拟合一次


    print(f"\n模型A（单因素-左收缩压每10mmHg）:")
    coef = model_bp1.params['L-SBP_per10']
    se = model_bp1.bse['L-SBP_per10']
    p_val = model_bp1.pvalues['L-SBP_per10']
    or_val = np.exp(coef)
    ci_lower = np.exp(coef - 1.96 * se)
    ci_upper = np.exp(coef + 1.96 * se)
    print(f"  左收缩压（每升高10mmHg）: OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.3f}, n={len(y_bp)}")
    
    # 模型B：调整年龄分组（每5岁）+性别
    # 构建年龄分组虚拟变量，并删除在子样本中样本量过少的年龄组
    age_dummies_b = pd.get_dummies(df_bp['AGE_group'], prefix='年龄')
    age_dummies_b = age_dummies_b.drop([c for c in age_dummies_b.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    # 删除样本量<5的年龄组（避免完全分离）
    for col in list(age_dummies_b.columns):
        if age_dummies_b[col].sum() < 5:
            age_dummies_b = age_dummies_b.drop(col, axis=1)
    # 以样本量最大的年龄组为参照
    if age_dummies_b.shape[1] > 0:
        cat_counts_b = age_dummies_b.sum().sort_values(ascending=False)
        ref_cat_b = cat_counts_b.index[0]
        age_dummies_b = age_dummies_b.drop(ref_cat_b, axis=1)

    X_b_extra = pd.DataFrame()
    X_b_extra['L-SBP_per10'] = df_bp['L-SBP_per10']
    for col in age_dummies_b.columns:
        X_b_extra[col] = age_dummies_b[col].astype(int)
    X_b_extra['Gender_male'] = df_bp['Gender_male']
    vars_b = list(X_b_extra.columns)
    valid_b = y_bp.notna() & X_b_extra.notna().all(axis=1)
    y_b = y_bp[valid_b]
    X_b = sm.add_constant(X_b_extra.loc[valid_b])
    # 再次检查：删除valid筛选后全为0的年龄组虚拟变量列（避免Singular matrix）
    age_cols_b = [c for c in X_b.columns if c.startswith('年龄_')]
    for col in age_cols_b:
        if X_b[col].nunique() <= 1 or X_b[col].sum() < 5:
            X_b = X_b.drop(col, axis=1)
    print(f"  模型B: n={len(y_b)}, 变量数={X_b.shape[1]}")
    print(f"  (参照组: {ref_cat_b}，即该年龄组的样本量最大)")
    
    if len(y_b) >= 20 and X_b.shape[1] <= len(y_b) - 2:
        try:
            model_b = sm.Logit(y_b, X_b).fit(disp=0)
        except Exception as e:
            model_b = None
            print(f"  模型B拟合失败: {e}")
    else:
        model_b = None
        print("  模型B：样本量不足或变量过多，跳过")



    # 模型C：调整年龄分组（每5岁）+性别+BMI+HbA1c
    # 先构建不含年龄组的基础数据，dropna后再创建年龄组虚拟变量
    X_c_base = pd.DataFrame()
    X_c_base['L-SBP_per10'] = df_bp['L-SBP_per10']
    X_c_base['Gender_male'] = df_bp['Gender_male']
    X_c_base['BMI_Crit'] = df_bp['BMI_Crit']
    X_c_base['HbA1c_Crit'] = df_bp['HbA1c_Crit']
    
    # 先dropna，再创建年龄组虚拟变量
    valid_c = y_bp.notna() & X_c_base.notna().all(axis=1)
    y_c = y_bp[valid_c]
    df_c_valid = df_bp.loc[valid_c]
    
    # 在有效样本中重新创建年龄组虚拟变量
    age_dummies_c = pd.get_dummies(df_c_valid['AGE_group'], prefix='年龄')
    age_dummies_c = age_dummies_c.drop([c for c in age_dummies_c.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    
    # 删除样本量<5的年龄组（避免完全分离）
    for col in list(age_dummies_c.columns):
        if age_dummies_c[col].sum() < 5:
            age_dummies_c = age_dummies_c.drop(col, axis=1)
    
    # 以样本量最大的年龄组为参照
    if age_dummies_c.shape[1] > 0:
        cat_counts_c = age_dummies_c.sum().sort_values(ascending=False)
        ref_cat_c = cat_counts_c.index[0]
        age_dummies_c = age_dummies_c.drop(ref_cat_c, axis=1)
    
    # 组装X_c
    X_c = pd.DataFrame()
    X_c['L-SBP_per10'] = df_c_valid['L-SBP_per10']
    for col in age_dummies_c.columns:
        X_c[col] = age_dummies_c[col].astype(int)
    X_c['Gender_male'] = df_c_valid['Gender_male']
    X_c['BMI_Crit'] = df_c_valid['BMI_Crit']
    X_c['HbA1c_Crit'] = df_c_valid['HbA1c_Crit']
    X_c = sm.add_constant(X_c)
    
    print(f"\n模型C（调整年龄分组+性别+BMI+HbA1c）: n={len(y_c)}, 变量数={X_c.shape[1]}")
    
    if len(y_c) >= 20 and X_c.shape[1] <= len(y_c) - 2:
        try:
            model_c = sm.Logit(y_c, X_c).fit(disp=0)
        except Exception as e:
            model_c = None
            print(f"  模型C拟合失败: {e}")
    else:
        model_c = None
        print("  模型C：样本量不足或变量过多，跳过")


    # 收集模型A-C的结果
    results_models = []
    for label, model_obj, n in [('模型A（单因素）', model_bp1, len(y_bp)),
                                  ('模型B（+年龄分组+性别）', model_b, len(y_b) if model_b is not None else 0),
                                  ('模型C（+年龄分组+性别+BMI+HbA1c）', model_c, len(y_c) if model_c is not None else 0)]:
        if model_obj is None:
            continue
        for param in model_obj.params.index:

            if param == 'const':
                continue
            coef = model_obj.params[param]
            se = model_obj.bse[param]
            p_val = model_obj.pvalues[param]
            or_val = np.exp(coef)
            ci_lower = np.exp(coef - 1.96 * se)
            ci_upper = np.exp(coef + 1.96 * se)
            display_param = '左收缩压（每升高10mmHg）' if param == 'L-SBP_per10' else param
            results_models.append({
                '分析方法': '350例全调整模型',
                '模型': label,
                '样本量': n,
                '变量': display_param,
                'OR': f"{or_val:.3f}",
                '95%CI下限': f"{ci_lower:.3f}",
                '95%CI上限': f"{ci_upper:.3f}",
                'P值': f"{p_val:.3f}"
            })
    
    # ===== 方法二：倾向性评分匹配（PSM） =====
    print("\n" + "-"*70)
    print("方法二：倾向性评分匹配（PSM）")
    print("-"*70)

    # 构建倾向性评分模型：以服药习惯为因变量，年龄分组+性别为协变量
    psm_vars_base = ['Gender_male']
    age_dummies_psm = pd.get_dummies(df_main['AGE_group'], prefix='年龄')
    age_dummies_psm = age_dummies_psm.drop([c for c in age_dummies_psm.columns if 'nan' in c.lower()], axis=1, errors='ignore')
    # 删除样本量<5的年龄组
    for col in list(age_dummies_psm.columns):
        if age_dummies_psm[col].sum() < 5:
            age_dummies_psm = age_dummies_psm.drop(col, axis=1)
    # 以样本量最大的年龄组为参照
    if age_dummies_psm.shape[1] > 0:
        cat_counts_psm = age_dummies_psm.sum().sort_values(ascending=False)
        ref_cat_psm = cat_counts_psm.index[0]
        age_dummies_psm = age_dummies_psm.drop(ref_cat_psm, axis=1)
    
    # 使用临时DataFrame，不污染df_main
    df_psm_temp = df_main[['med_fasting', 'L-SBP_Crit', 'L-SBP_per10', 'AGE_group', 'Gender_male']].copy()
    for col in age_dummies_psm.columns:
        df_psm_temp[col] = age_dummies_psm[col].astype(int)
    psm_vars = psm_vars_base + list(age_dummies_psm.columns)

    valid_psm = df_psm_temp['med_fasting'].notna() & df_psm_temp[psm_vars].notna().all(axis=1)
    df_psm = df_psm_temp.loc[valid_psm].copy()

    print(f"PSM总样本量: {len(df_psm)}")
    print(f"  空腹服药（处理组）: {(df_psm['med_fasting']==1).sum()}例")
    print(f"  餐后服药（对照组）: {(df_psm['med_fasting']==0).sum()}例")
    
    # 计算倾向性评分
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(df_psm[psm_vars], df_psm['med_fasting'])
    df_psm['propensity_score'] = ps_model.predict_proba(df_psm[psm_vars])[:, 1]
    
    print(f"\n倾向性评分分布:")
    print(f"  空腹服药组: 均值={df_psm[df_psm['med_fasting']==1]['propensity_score'].mean():.3f} ± {df_psm[df_psm['med_fasting']==1]['propensity_score'].std():.3f}")
    print(f"  餐后服药组: 均值={df_psm[df_psm['med_fasting']==0]['propensity_score'].mean():.3f} ± {df_psm[df_psm['med_fasting']==0]['propensity_score'].std():.3f}")
    
    
    ps_treatment = df_psm[df_psm['med_fasting']==1]['propensity_score']
    ps_control = df_psm[df_psm['med_fasting']==0]['propensity_score']
    # 计算两组PS的Cohen's d
    mean_diff = ps_treatment.mean() - ps_control.mean()
    pooled_std = np.sqrt((ps_treatment.var() + ps_control.var()) / 2)
    cohens_d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
    print(f"倾向性评分组间差异: Cohen's d = {cohens_d:.3f}")
    # ===== 新增：当Cohen's d < 0.3时给出警告 =====
    if cohens_d < 0.3:
        print(f"  ⚠️ 警告：两组倾向性评分差异很小（Cohen's d={cohens_d:.3f}）")
        print(f"  ⚠️ 说明年龄和性别对服药习惯的预测能力很弱，PSM在此数据中可能不适用")
        print(f"  ⚠️ 建议以多因素回归或IPW结果为主要结论依据")
    # 1:1最近邻匹配（卡钳值0.05）
    treatment = df_psm[df_psm['med_fasting']==1].copy()
    control = df_psm[df_psm['med_fasting']==0].copy()
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treatment[['propensity_score']])
    
    matched_indices = []
    used_controls = set()
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        control_original_index = control.index[idx[0]]
        if dist[0] <= 0.05 and control_original_index not in used_controls:
            matched_indices.append((treatment.index[i], control_original_index))
            used_controls.add(control_original_index)

    matched_treatment = [m[0] for m in matched_indices]
    matched_control = [m[1] for m in matched_indices]
    df_matched = pd.concat([
        df_psm.loc[matched_treatment],
        df_psm.loc[matched_control]
    ])
    
    print(f"\n匹配后样本量: {len(df_matched)}")
    print(f"  空腹服药: {len(matched_treatment)}例")
    print(f"  餐后服药: {len(matched_control)}例")
    
    if len(df_matched) < 50:
        print(f"\n  ⚠️ 警告：匹配后样本量={len(df_matched)}<50，结果可能不可靠！")
        print(f"  ⚠️ 建议：仅进行单因素分析，不进行多因素Logistic回归")
        
        # 仅做单因素分析
        bp_treatment = df_matched[df_matched['med_fasting']==1]['L-SBP_Crit'].dropna()
        bp_control = df_matched[df_matched['med_fasting']==0]['L-SBP_Crit'].dropna()
        
        print(f"\n  匹配后左收缩压比较（单因素分析）:")
        print(f"    空腹服药: {bp_treatment.mean():.1f} ± {bp_treatment.std():.1f} (n={len(bp_treatment)})")
        print(f"    餐后服药: {bp_control.mean():.1f} ± {bp_control.std():.1f} (n={len(bp_control)})")
        
        if len(bp_treatment) >= 5 and len(bp_control) >= 5:
            t_stat, p_val_psm = stats.ttest_ind(bp_treatment, bp_control)
            print(f"    t检验: t={t_stat:.3f}, P={p_val_psm:.4f}")
            print(f"    ⚠️ 注意：此结果仅基于单因素比较，未调整混杂因素")
        else:
            p_val_psm = np.nan
            print(f"    t检验: 样本量不足")
        
        # 跳过后续多因素Logistic回归
        model_psm = None
    else:
        # 匹配后比较左收缩压（原始代码）
        bp_treatment = df_matched[df_matched['med_fasting']==1]['L-SBP_Crit'].dropna()
        bp_control = df_matched[df_matched['med_fasting']==0]['L-SBP_Crit'].dropna()
        print(f"  空腹服药: {bp_treatment.mean():.1f} ± {bp_treatment.std():.1f} (n={len(bp_treatment)})")
        print(f"  餐后服药: {bp_control.mean():.1f} ± {bp_control.std():.1f} (n={len(bp_control)})")
        
        if len(bp_treatment) >= 5 and len(bp_control) >= 5:
            t_stat, p_val_psm = stats.ttest_ind(bp_treatment, bp_control)
            print(f"  t检验: t={t_stat:.3f}, P={p_val_psm:.4f}")
        else:
            p_val_psm = np.nan
            print(f"  t检验: 样本量不足")
        
        # PSM后的Logistic回归（原始代码）
        psm_match_vars = ['med_fasting', 'L-SBP_per10', 'AGE_group', 'Gender_male'] + [c for c in df_matched.columns if c.startswith('年龄_')]
        df_matched_model = df_matched[psm_match_vars].dropna()
        if len(df_matched_model) >= 20:
            y_psm = df_matched_model['med_fasting']
            # 在匹配后样本中重新创建年龄分组虚拟变量
            age_dummies_psm2 = pd.get_dummies(df_matched_model['AGE_group'], prefix='年龄')
            age_dummies_psm2 = age_dummies_psm2.drop([c for c in age_dummies_psm2.columns if 'nan' in c.lower()], axis=1, errors='ignore')
            # 删除样本量<5的年龄组
            for col in list(age_dummies_psm2.columns):
                if age_dummies_psm2[col].sum() < 5:
                    age_dummies_psm2 = age_dummies_psm2.drop(col, axis=1)
            # 以样本量最大的年龄组为参照
            if age_dummies_psm2.shape[1] > 0:
                cat_counts_psm2 = age_dummies_psm2.sum().sort_values(ascending=False)
                ref_cat_psm2 = cat_counts_psm2.index[0]
                age_dummies_psm2 = age_dummies_psm2.drop(ref_cat_psm2, axis=1)
            
            X_psm = pd.DataFrame()
            X_psm['L-SBP_per10'] = df_matched_model['L-SBP_per10']
            for col in age_dummies_psm2.columns:
                X_psm[col] = age_dummies_psm2[col].astype(int)
            X_psm['Gender_male'] = df_matched_model['Gender_male']
            X_psm = sm.add_constant(X_psm)
            
            print(f"\nPSM匹配后Logistic回归（调整年龄+性别）: n={len(y_psm)}, 变量数={X_psm.shape[1]}")

            try:
                model_psm = sm.Logit(y_psm, X_psm).fit(disp=0)
            except Exception as e:
                model_psm = None
                print(f"  PSM模型拟合失败: {e}")

            
            if model_psm is not None:
                for param in model_psm.params.index:
                    if param == 'const':
                        continue
                    coef = model_psm.params[param]
                    se = model_psm.bse[param]
                    p_val = model_psm.pvalues[param]
                    or_val = np.exp(coef)
                    ci_lower = np.exp(coef - 1.96 * se)
                    ci_upper = np.exp(coef + 1.96 * se)
                    label = '左收缩压（每升高10mmHg）' if param == 'L-SBP_per10' else param
                    print(f"  {label}: OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.4f}")
                    
                    # 保存到results_models
                    results_models.append({
                        '分析方法': 'PSM匹配后回归',
                        '模型': f'PSM匹配(n={len(y_psm)})',
                        '样本量': len(y_psm),
                        '变量': label,
                        'OR': f"{or_val:.3f}",
                        '95%CI下限': f"{ci_lower:.3f}",
                        '95%CI上限': f"{ci_upper:.3f}",
                        'P值': f"{p_val:.4f}"
                    })

    # ===== 方法三：逆概率加权（IPW） =====
    print("\n" + "-"*70)
    print("方法三：逆概率加权（IPW）")
    print("-"*70)
    
    df_ipw = df_psm.copy()
    df_ipw['ipw_weight'] = np.where(
        df_ipw['med_fasting'] == 1,
        1.0 / df_ipw['propensity_score'],
        1.0 / (1.0 - df_ipw['propensity_score'])
    )
    lower_cap = df_ipw['ipw_weight'].quantile(0.01)
    upper_cap = df_ipw['ipw_weight'].quantile(0.99)
    df_ipw['ipw_weight_capped'] = df_ipw['ipw_weight'].clip(lower_cap, upper_cap)
    
    print(f"IPW权重统计:")
    print(f"  权重范围: {df_ipw['ipw_weight_capped'].min():.3f} ~ {df_ipw['ipw_weight_capped'].max():.3f}")
    print(f"  权重均值: {df_ipw['ipw_weight_capped'].mean():.3f}")
    
    # IPW加权Logistic回归（使用每10mmHg）

    # 关键：在选取列时，必须包含 'AGE_group' 列
    ipw_model_vars = ['med_fasting', 'L-SBP_per10', 'AGE_group', 'Gender_male', 'ipw_weight_capped'] + [c for c in df_ipw.columns if c.startswith('年龄_')]

    df_ipw_model = df_ipw[ipw_model_vars].dropna()
    if len(df_ipw_model) >= 20:
        y_ipw = df_ipw_model['med_fasting']
        # 在IPW样本中重新创建年龄分组虚拟变量
        age_dummies_ipw = pd.get_dummies(df_ipw_model['AGE_group'], prefix='年龄')  # ✅ 现在有'AGE_group'列了

        # 删除样本量<2的年龄组
        for col in list(age_dummies_ipw.columns):
            if age_dummies_ipw[col].sum() < 2:
                age_dummies_ipw = age_dummies_ipw.drop(col, axis=1)
        # 第一个年龄组为参照
        valid_age_cats_ipw = sorted([c for c in age_dummies_ipw.columns])
        if valid_age_cats_ipw:
            age_dummies_ipw = age_dummies_ipw.drop(valid_age_cats_ipw[0], axis=1)
        
        X_ipw = pd.DataFrame()
        X_ipw['L-SBP_per10'] = df_ipw_model['L-SBP_per10']
        for col in age_dummies_ipw.columns:
            X_ipw[col] = age_dummies_ipw[col].astype(int)
        X_ipw['Gender_male'] = df_ipw_model['Gender_male']
        X_ipw = sm.add_constant(X_ipw)
        
        print(f"\nIPW加权Logistic回归（调整年龄+性别）: n={len(df_ipw_model)}, 变量数={X_ipw.shape[1]}")

        try:
            model_ipw = sm.GLM(y_ipw, X_ipw, 
                              family=sm.families.Binomial(),
                              freq_weights=df_ipw_model['ipw_weight_capped']).fit()
        except Exception as e:
            model_ipw = None
            print(f"  IPW模型拟合失败: {e}")

        
    if model_ipw is not None:
        for param in model_ipw.params.index:
            if param == 'const':
                continue
            coef = model_ipw.params[param]
            se = model_ipw.bse[param]
            p_val = model_ipw.pvalues[param]
            or_val = np.exp(coef)
            ci_lower = np.exp(coef - 1.96 * se)
            ci_upper = np.exp(coef + 1.96 * se)
            label = '左收缩压（每升高10mmHg）' if param == 'L-SBP_per10' else param
            print(f"  {label}: OR={or_val:.3f} (95%CI: {ci_lower:.3f}~{ci_upper:.3f}), P={p_val:.4f}")
            
            # ✅ 立即保存到results_models
            results_models.append({
                '分析方法': 'IPW加权回归',
                '模型': f'IPW(n={len(df_ipw_model)})',
                '样本量': len(df_ipw_model),
                '变量': label,
                'OR': f"{or_val:.3f}",
                '95%CI下限': f"{ci_lower:.3f}",
                '95%CI上限': f"{ci_upper:.3f}",
                'P值': f"{p_val:.4f}"
            })

    # ===== 方法四：E-value敏感性分析 =====
    print("\n" + "-"*70)
    print("方法四：E-value敏感性分析")
    print("-"*70)

    def compute_evalue(or_val, ci_lower=None, ci_upper=None):
        """计算E-value"""
        def _evalue(rr):
            if rr < 1:
                rr = 1 / rr
            return rr + np.sqrt(rr * (rr - 1))
        
        evalue_point = _evalue(or_val)
        
        evalue_ci = None
        if ci_lower is not None and ci_upper is not None:
            if or_val < 1:
                evalue_ci = _evalue(ci_upper) if ci_upper < 1 else 1.0
            else:
                evalue_ci = _evalue(ci_lower) if ci_lower > 1 else 1.0
        
        return evalue_point, evalue_ci
    
    # 优先使用多因素模型B的调整后OR，如果失败则回退到单因素
    if model_b is not None:
        or_evalue = np.exp(model_b.params['L-SBP_per10'])
        ci_lower_evalue = np.exp(model_b.params['L-SBP_per10'] - 1.96 * model_b.bse['L-SBP_per10'])
        ci_upper_evalue = np.exp(model_b.params['L-SBP_per10'] + 1.96 * model_b.bse['L-SBP_per10'])
        print(f"\n多因素调整后结果（左收缩压每升高10mmHg）:")
    else:
        or_evalue = np.exp(model_bp1.params['L-SBP_per10'])
        ci_lower_evalue = np.exp(model_bp1.params['L-SBP_per10'] - 1.96 * model_bp1.bse['L-SBP_per10'])
        ci_upper_evalue = np.exp(model_bp1.params['L-SBP_per10'] + 1.96 * model_bp1.bse['L-SBP_per10'])
        print(f"\n单因素结果（左收缩压每升高10mmHg）:")
    
    print(f"  OR={or_evalue:.3f} (95%CI: {ci_lower_evalue:.3f}~{ci_upper_evalue:.3f})")
    
    evalue_point, evalue_ci = compute_evalue(or_evalue, ci_lower_evalue, ci_upper_evalue)
    
    print(f"\nE-value分析:")
    print(f"  点估计E-value = {evalue_point:.3f}")
    print(f"  95%CI E-value = {evalue_ci:.3f}" if evalue_ci is not None else "")
    print(f"\n解释: 要推翻左收缩压（每升高10mmHg）与空腹服药的关联,")
    print(f"  未测量的混杂因素与左收缩压和空腹服药的关联强度需均达到OR={evalue_point:.3f}以上")
    print(f"  才能使观察到的关联被完全解释")
    
    # 对比已知混杂因素的强度
    print(f"\n已知混杂因素强度对比（来自多因素模型B）:")
    if model_b is not None and 'Gender_male' in model_b.params.index:
        gender_or_adjusted = np.exp(model_b.params['Gender_male'])
        known_factors = {
            '性别（男性）': gender_or_adjusted
        }
    else:
        known_factors = {
            '性别（男性）': 1.567
        }
    
    for factor, or_factor in known_factors.items():
        if or_factor < 1:
            or_factor_inv = 1 / or_factor
            print(f"  {factor}: OR={or_factor:.3f} (逆OR={or_factor_inv:.3f})")
        else:
            print(f"  {factor}: OR={or_factor:.3f}")
    
    print(f"\n结论: 若E-value > 已知混杂因素的OR值，则未测量混杂不太可能完全解释观察到的关联")
    print(f"  当前E-value={evalue_point:.3f}，而最强已知混杂（性别）OR={list(known_factors.values())[0]:.3f}")
    if evalue_point > list(known_factors.values())[0]:
        print(f"  → 未测量混杂需要比性别更强的关联才能推翻结论，结果较为稳健")
    else:
        print(f"  → 未测量混杂强度要求不高，结果可能不够稳健")

    
    # ===== 汇总结果 =====
    print("\n" + "="*70)
    print("汇总：左收缩压（每升高10mmHg）与空腹服药的关联分析")
    print("="*70)
    
    summary_data = []
    
    # 方法一结果
    for r in results_models:
        summary_data.append(r)
    
    # # PSM结果
    # if len(df_matched_model) >= 20:
    #     for param in model_psm.params.index:
    #         if param == 'const':
    #             continue
    #         coef = model_psm.params[param]
    #         se = model_psm.bse[param]
    #         p_val = model_psm.pvalues[param]
    #         or_val = np.exp(coef)
    #         ci_lower = np.exp(coef - 1.96 * se)
    #         ci_upper = np.exp(coef + 1.96 * se)
    #         display_param = '左收缩压（每升高10mmHg）' if param == 'L-SBP_per10' else param
    #         summary_data.append({
    #             '分析方法': 'PSM匹配后回归',
    #             '模型': f'PSM匹配(n={len(df_matched_model)})',
    #             '样本量': len(df_matched_model),
    #             '变量': display_param,
    #             'OR': f"{or_val:.3f}",
    #             '95%CI下限': f"{ci_lower:.3f}",
    #             '95%CI上限': f"{ci_upper:.3f}",
    #             'P值': f"{p_val:.3f}"
    #         })
    
    # # IPW结果
    # if len(df_ipw_model) >= 20:
    #     for param in model_ipw.params.index:
    #         if param == 'const':
    #             continue
    #         coef = model_ipw.params[param]
    #         se = model_ipw.bse[param]
    #         p_val = model_ipw.pvalues[param]
    #         or_val = np.exp(coef)
    #         ci_lower = np.exp(coef - 1.96 * se)
    #         ci_upper = np.exp(coef + 1.96 * se)
    #         display_param = '左收缩压（每升高10mmHg）' if param == 'L-SBP_per10' else param
    #         summary_data.append({
    #             '分析方法': 'IPW加权回归',
    #             '模型': f'IPW(n={len(df_ipw_model)})',
    #             '样本量': len(df_ipw_model),
    #             '变量': display_param,
    #             'OR': f"{or_val:.3f}",
    #             '95%CI下限': f"{ci_lower:.3f}",
    #             '95%CI上限': f"{ci_upper:.3f}",
    #             'P值': f"{p_val:.3f}"
    #         })
    
    # E-value结果
    summary_data.append({
        '分析方法': 'E-value敏感性分析',
        '模型': 'E-value',
        '样本量': '',
        '变量': '左收缩压（每10mmHg）E-value',
        'OR': f"{evalue_point:.3f}",
        '95%CI下限': '',
        '95%CI上限': '',
        'P值': f"95%CI E-value={evalue_ci:.3f}" if evalue_ci is not None else ''
    })
    
    summary_df = pd.DataFrame(summary_data)
    results_path = os.path.join(output_dir, f'左收缩压混杂因素分析_{timestamp}.csv')
    summary_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n分析结果已保存到: {results_path}")
    print("\n结果汇总:")
    print(summary_df.to_string(index=False))
    
    return summary_df


# ==================== 主函数 ====================
def main():
    # 文件路径
    file_path = r'D:/data/data_arrangement/Output/Enroll_2026_处理_0520.xlsx'  # 请修改为实际路径
    
    # 加载数据
    print("正在加载数据...")
    df = load_and_preprocess_data(file_path)
    
    # 表1：服药习惯分组描述
    table1 = generate_table1(df)
    
    # 核心分析：服药习惯的多因素Logistic回归
    logistic_results = generate_multivariable_logistic(df)
    
    # 亚组分析
    subgroup_results = generate_subgroup_analysis(df)
    
    # 左右侧血压分析
    bp_desc, bp_ttest = generate_bp_numeric_analysis(df)
    
    # 用药记忆情况分析
    memory_results = generate_medication_memory_analysis(df)
    
    # 左收缩压混杂因素深入分析
    bp_confounding_results = generate_bp_confounding_analysis(df)
    generate_bp_control_analysis_results = generate_bp_control_analysis(df)

    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    
    return {
        'table1': table1,
        'logistic_results': logistic_results,
        'subgroup_results': subgroup_results,
        'bp_desc': bp_desc,
        'bp_ttest': bp_ttest,
        'generate_bp_control_analysis_results':generate_bp_control_analysis_results,
        'bp_confounding_results': bp_confounding_results
    }

if __name__ == '__main__':
    results = main()
    