import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates
# 修正导入：是 MinMaxScaler
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 0. 顶刊绘图风格设置
# ==========================================
def set_pub_style():
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.rcParams['font.family'] = 'sans-serif'
    # 兼容多种中文字体，防止乱码
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


set_pub_style()


# ==========================================
# 1. 数据加载模块
# ==========================================
class CityDataLoader:
    def __init__(self, base_path, city_name="city_A"):
        self.city_path = os.path.join(base_path, city_name)

    def normalize_date(self, df, col='date'):
        """将日期标准化为相对天数和伪日期对象"""
        if df.empty or col not in df.columns: return df

        # 强制转数值 (处理 '20200101' 或 '0, 1, 2')
        # errors='coerce' 会把无法转换的变成 NaN，fillna(0) 填充
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 如果数值很大(如20200101)，说明是绝对日期，需要转换成相对天数
        if df[col].max() > 10000:
            temp = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
            min_date = temp.min()
            # 覆盖原列为相对天数 (int)
            df[col] = (temp - min_date).dt.days.fillna(0).astype(int)
        else:
            # 已经是相对天数
            df[col] = df[col].astype(int)

        # 构造一个伪时间用于绘图 (Base: 2020-01-01)
        df['dt'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df[col], unit='D')
        return df

    def load_temporal_data(self):
        print("Loading data...")

        # 1. Infection (Daily)
        f_inf = os.path.join(self.city_path, 'infection.csv')
        if os.path.exists(f_inf):
            df_inf = pd.read_csv(f_inf, names=['date', 'region', 'cnt'], header=None)
            df_inf = self.normalize_date(df_inf)
        else:
            df_inf = pd.DataFrame()

        # 2. Migration (Daily)
        f_mig = os.path.join(self.city_path, 'migration.csv')
        if os.path.exists(f_mig):
            df_mig = pd.read_csv(f_mig, names=['date', 'f', 't', 'idx'], header=None)
            df_mig = self.normalize_date(df_mig)
        else:
            df_mig = pd.DataFrame()

        # 3. Density (Hourly) - 限制行数防止内存溢出
        f_den = os.path.join(self.city_path, 'density.csv')
        if os.path.exists(f_den):
            df_den = pd.read_csv(f_den, names=['date', 'hour', 'lon', 'lat', 'idx'], header=None, nrows=500000)
            df_den = self.normalize_date(df_den)
        else:
            df_den = pd.DataFrame()

        return df_inf, df_mig, df_den


# ==========================================
# 2. 高级时序可视化模块
# ==========================================

def plot_hourly_pulse(df_den):
    """图1: 城市脉搏 (小时级变化)"""
    print("Plotting Hourly Pulse...")
    if df_den.empty:
        print("Skipping: Density data is empty.")
        return

    # 聚合：计算每一天、每一小时的平均人流密度
    hourly_trend = df_den.groupby(['date', 'hour'])['idx'].mean().reset_index()

    plt.figure(figsize=(10, 6))

    # 绘制带置信区间的折线图
    sns.lineplot(data=hourly_trend, x='hour', y='idx',
                 color='#2A9D8F', linewidth=2.5, marker='o')

    plt.title('Intraday Mobility Pattern: The "Pulse" of the City', fontsize=14, pad=15)
    plt.xlabel('Hour of Day (0-23)', fontweight='bold')
    plt.ylabel('Average Flow Index', fontweight='bold')
    plt.xticks(range(0, 24, 2))

    # 标记白天黑夜
    plt.axvspan(0, 6, color='gray', alpha=0.1, label='Night')
    plt.axvspan(18, 23, color='gray', alpha=0.1)

    plt.tight_layout()
    plt.show()


def plot_normalized_evolution(df_inf, df_mig, df_den):
    """图2: 归一化协同演变"""
    print("Plotting Normalized Evolution...")
    if df_inf.empty or df_mig.empty:
        print("Skipping: Inf or Mig data is empty.")
        return

    # 聚合每日数据
    d_inf = df_inf.groupby('date')['cnt'].sum().reset_index().rename(columns={'cnt': 'Infection'})
    d_mig = df_mig.groupby('date')['idx'].sum().reset_index().rename(columns={'idx': 'Migration'})

    if not df_den.empty:
        d_den = df_den.groupby('date')['idx'].mean().reset_index().rename(columns={'idx': 'Density'})
    else:
        d_den = pd.DataFrame(columns=['date', 'Density'])

    # 合并所有数据
    merged = pd.merge(d_inf, d_mig, on='date', how='outer')
    if not d_den.empty:
        merged = pd.merge(merged, d_den, on='date', how='outer')

    merged = merged.sort_values('date').fillna(0)

    # 归一化处理 (Min-Max Scaling)
    scaler = MinMaxScaler()
    cols_to_norm = [c for c in ['Infection', 'Migration', 'Density'] if c in merged.columns]

    # 防止全0列导致除以0警告
    if not merged.empty:
        merged[cols_to_norm] = scaler.fit_transform(merged[cols_to_norm])

    # 转换为长格式以便绘图
    melted = merged.melt(id_vars='date', value_vars=cols_to_norm, var_name='Type', value_name='Normalized Value')

    plt.figure(figsize=(12, 6))

    palette = {'Infection': '#E63946', 'Migration': '#457B9D', 'Density': '#2A9D8F'}

    sns.lineplot(data=melted, x='date', y='Normalized Value', hue='Type',
                 palette=palette, linewidth=2.5, style='Type', markers=True, dashes=False)

    plt.title('Co-evolution of Epidemic and Mobility (Normalized [0-1])', fontsize=14)
    plt.xlabel('Days (Relative)', fontweight='bold')
    plt.ylabel('Normalized Intensity', fontweight='bold')

    # 标注最高点
    try:
        peak_inf = merged.loc[merged['Infection'].idxmax()]
        plt.annotate(f"Infection Peak (Day {int(peak_inf['date'])})",
                     xy=(peak_inf['date'], peak_inf['Infection']),
                     xytext=(peak_inf['date'] + 2, peak_inf['Infection']),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    except:
        pass

    plt.legend(title='Metric')
    plt.tight_layout()
    plt.show()


def plot_time_lag_correlation(df_inf, df_mig):
    """图3: 时滞相关性分析"""
    print("Plotting Time-Lag Analysis...")
    if df_inf.empty or df_mig.empty: return

    # 准备序列
    ts_inf = df_inf.groupby('date')['cnt'].sum()
    ts_mig = df_mig.groupby('date')['idx'].sum()

    # 索引对齐（补0）
    idx = range(int(min(ts_inf.index.min(), ts_mig.index.min())),
                int(max(ts_inf.index.max(), ts_mig.index.max())) + 1)
    ts_inf = ts_inf.reindex(idx, fill_value=0)
    ts_mig = ts_mig.reindex(idx, fill_value=0)

    lags = range(-10, 16)  # 滞后 -10 到 +15 天
    corrs = []

    for k in lags:
        # 计算 Migration(t) 和 Infection(t+k) 的相关性
        # shift(-k) 将 Infection 数据向上移动 k 天（即 t 时刻对应 t+k 的数据）
        shifted_inf = ts_inf.shift(-k)

        valid_mask = ~np.isnan(shifted_inf) & ~np.isnan(ts_mig)
        if valid_mask.sum() > 5:
            c = np.corrcoef(ts_mig[valid_mask], shifted_inf[valid_mask])[0, 1]
        else:
            c = 0
        corrs.append(c)

    plt.figure(figsize=(10, 6))

    colors = ['#E63946' if c > 0 else '#457B9D' for c in corrs]
    plt.bar(lags, corrs, color=colors, alpha=0.8)

    best_lag = lags[np.argmax(corrs)]
    max_corr = max(corrs)

    plt.axvline(best_lag, color='black', linestyle='--', alpha=0.5, label=f'Max Corr (Lag={best_lag})')

    plt.title('Time-Lag Correlation: Migration(t) vs Infection(t + lag)', fontsize=14)
    plt.xlabel('Lag (Days) [Positive: Migration leads Infection]', fontweight='bold')
    plt.ylabel('Pearson Correlation', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Max Correlation: {max_corr:.2f} at Lag: {best_lag} days.")


# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 请确保路径正确
    BASE_DIR = r"C:\Users\21626\Downloads\train_data\train_data"
    TARGET_CITY = "city_A"

    if os.path.exists(os.path.join(BASE_DIR, TARGET_CITY)):
        loader = CityDataLoader(BASE_DIR, TARGET_CITY)
        df_inf, df_mig, df_den = loader.load_temporal_data()

        plot_hourly_pulse(df_den)
        plot_normalized_evolution(df_inf, df_mig, df_den)
        plot_time_lag_correlation(df_inf, df_mig)

        print("Analysis Complete.")
    else:
        print(f"Path Error: {os.path.join(BASE_DIR, TARGET_CITY)} not found.")