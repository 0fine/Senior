import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


# ==========================================
# 0. CS顶刊绘图风格设置
# ==========================================
def set_pub_style():
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "axes.edgecolor": "0.15"})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False


set_pub_style()


# ==========================================
# 1. 多城市数据加载器 (Multi-City Loader)
# ==========================================
class MultiCityLoader:
    def __init__(self, base_path, cities):
        self.base_path = base_path
        self.cities = cities  # List of city names ['city_A', 'city_B'...]

    def normalize_date(self, df, col='date'):
        """将日期标准化为相对天数 (0, 1, 2...)"""
        if df.empty or col not in df.columns: return df

        # 强制转数值
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 如果是绝对日期格式 (如 20200101)，转为相对天数
        if df[col].max() > 10000:
            temp = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
            min_date = temp.min()
            df['day'] = (temp - min_date).dt.days.fillna(0).astype(int)
        else:
            df['day'] = df[col].astype(int)

        return df

    def load_combined_data(self):
        print(f"Aggregating data across {len(self.cities)} cities...")

        all_inf = []
        all_mig = []
        all_wea = []
        city_stats = []

        for city in self.cities:
            city_path = os.path.join(self.base_path, city)
            if not os.path.exists(city_path):
                print(f"Warning: {city} not found.")
                continue

            # 1. Infection
            f_inf = os.path.join(city_path, 'infection.csv')
            if os.path.exists(f_inf):
                df = pd.read_csv(f_inf, names=['date', 'region', 'cnt'], header=None)
                df = self.normalize_date(df)
                df['city'] = city
                all_inf.append(df)

            # 2. Migration
            f_mig = os.path.join(city_path, 'migration.csv')
            if os.path.exists(f_mig):
                df = pd.read_csv(f_mig, names=['date', 'f', 't', 'idx'], header=None)
                df = self.normalize_date(df)
                df['city'] = city
                all_mig.append(df)

            # 3. Weather
            f_wea = os.path.join(city_path, 'weather.csv')
            if os.path.exists(f_wea):
                df = pd.read_csv(f_wea,
                                 names=['date', 'hour', 'temp', 'humidity', 'wind_dir', 'wind_spd', 'force', 'type'],
                                 header=None)
                df = self.normalize_date(df)
                df['city'] = city
                all_wea.append(df)

        # 合并所有城市数据
        master_inf = pd.concat(all_inf, ignore_index=True) if all_inf else pd.DataFrame()
        master_mig = pd.concat(all_mig, ignore_index=True) if all_mig else pd.DataFrame()
        master_wea = pd.concat(all_wea, ignore_index=True) if all_wea else pd.DataFrame()

        return master_inf, master_mig, master_wea


# ==========================================
# 2. 联合分析可视化模块
# ==========================================

def plot_comparative_epidemic_curves(df_inf):
    """
    图1: 疫情爆发曲线对比 (Epidemic Curves Comparison)
    分析不同城市的爆发规模和峰值到达时间。
    """
    print("Plotting Comparative Curves...")
    if df_inf.empty: return

    # 按城市和天数聚合
    daily_trend = df_inf.groupby(['city', 'day'])['cnt'].sum().reset_index()

    plt.figure(figsize=(10, 6))

    # 绘制曲线
    sns.lineplot(data=daily_trend, x='day', y='cnt', hue='city',
                 palette='bright', linewidth=2.5, alpha=0.8)

    plt.title('Heterogeneity in Epidemic Propagation Across Cities', fontsize=14, pad=20)
    plt.xlabel('Relative Day (Since Start)', fontweight='bold')
    plt.ylabel('Daily New Infections', fontweight='bold')
    plt.legend(title='City ID', frameon=True)
    plt.tight_layout()
    plt.show()


def plot_gravity_law_verification(df_inf, df_mig):
    """
    图2: 验证重力模型 (Migration vs Infection Scale)
    散点图：X轴为总迁徙规模，Y轴为总感染人数。
    验证假设：流动性越强的城市，疫情规模越大。
    """
    print("Plotting Gravity Law Verification...")
    if df_inf.empty or df_mig.empty: return

    # 计算每个城市的指标
    # 1. 总感染人数
    total_inf = df_inf.groupby('city')['cnt'].sum().reset_index(name='Total Infections')

    # 2. 总迁徙强度 (流入+流出)
    total_mig = df_mig.groupby('city')['idx'].sum().reset_index(name='Total Migration Index')

    # 3. 感染峰值天数 (Peak Day)
    peak_day = df_inf.groupby(['city', 'day'])['cnt'].sum().reset_index()
    peak_day = peak_day.loc[peak_day.groupby('city')['cnt'].idxmax()][['city', 'day']]
    peak_day.columns = ['city', 'Peak Day']

    # 合并
    merged = pd.merge(total_inf, total_mig, on='city')
    merged = pd.merge(merged, peak_day, on='city')

    # 绘图
    plt.figure(figsize=(8, 7))

    # 散点大小代表峰值出现的早晚 (点越大，峰值出现越晚 -> 其实点越大代表某种权重更好，这里用固定大小)
    sns.scatterplot(data=merged, x='Total Migration Index', y='Total Infections',
                    hue='city', s=300, palette='bright', alpha=0.9, edgecolor='black')

    # 添加线性回归趋势线 (Linear Fit)
    sns.regplot(data=merged, x='Total Migration Index', y='Total Infections',
                scatter=False, color='gray', line_kws={'linestyle': '--', 'alpha': 0.5})

    # 计算相关系数
    r, p = stats.pearsonr(merged['Total Migration Index'], merged['Total Infections'])

    # 添加文字标注
    for i in range(merged.shape[0]):
        plt.text(merged['Total Migration Index'][i] + 0.02 * merged['Total Migration Index'].max(),
                 merged['Total Infections'][i],
                 merged['city'][i], fontsize=11, fontweight='bold')

    plt.title(f'The "Gravity Law": Mobility vs Epidemic Scale\n(Pearson r={r:.2f}, p={p:.3f})', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_weather_robustness_heatmap(df_inf, df_wea):
    """
    图3: 环境因素影响的鲁棒性分析 (Heatmap)
    行：不同的城市
    列：天气特征 (温度, 湿度, 风速)
    颜色：Pearson相关系数
    分析：是否在所有城市中，气温都和疫情呈负相关？
    """
    print("Plotting Weather Robustness...")
    if df_inf.empty or df_wea.empty: return

    # 1. 数据预处理
    weather_cols = ['temp', 'humidity', 'wind_spd']
    # 确保数值类型
    for c in weather_cols:
        if c in df_wea.columns:
            df_wea[c] = pd.to_numeric(df_wea[c], errors='coerce')

    # 存储相关系数矩阵
    corr_data = []
    cities = df_inf['city'].unique()

    for city in cities:
        # 获取该城市数据
        sub_inf = df_inf[df_inf['city'] == city].groupby('day')['cnt'].sum().reset_index()
        sub_wea = df_wea[df_wea['city'] == city].groupby('day')[weather_cols].mean().reset_index()

        # 合并
        merged = pd.merge(sub_inf, sub_wea, on='day')
        if len(merged) < 5: continue

        # 计算该城市的相关系数
        corrs = merged.corr()['cnt'][weather_cols]  # 只取与感染数的相关性
        corrs['city'] = city
        corr_data.append(corrs)

    if not corr_data: return

    corr_df = pd.DataFrame(corr_data).set_index('city')

    # 绘图
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, vmin=-0.8, vmax=0.8,
                linewidths=1, linecolor='white', fmt='.2f')

    plt.title('Robustness Check: Weather-Infection Correlation Across Cities', fontsize=14)
    plt.ylabel('City')
    plt.xlabel('Environmental Factors')
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置路径
    BASE_DIR = r"C:\Users\21626\Downloads\train_data\train_data"

    # 定义要分析的城市列表 (假设有5个)
    # 如果实际文件名是 city_A, city_B... 请确保这里一致
    TARGET_CITIES = ['city_A', 'city_B', 'city_C', 'city_D', 'city_E']

    if os.path.exists(BASE_DIR):
        # 1. 加载数据
        loader = MultiCityLoader(BASE_DIR, TARGET_CITIES)
        df_inf, df_mig, df_wea = loader.load_combined_data()

        if not df_inf.empty:
            # 2. 联合分析图表

            # 图1: 各城市疫情曲线对比
            plot_comparative_epidemic_curves(df_inf)

            # 图2: 迁徙规模与疫情规模的回归分析
            plot_gravity_law_verification(df_inf, df_mig)

            # 图3: 环境影响的一致性热力图
            plot_weather_robustness_heatmap(df_inf, df_wea)

            print("Multi-city analysis complete.")
        else:
            print("No infection data loaded. Check city directory names.")
    else:
        print(f"Base path not found: {BASE_DIR}")