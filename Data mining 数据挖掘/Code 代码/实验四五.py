import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.preprocessing import MinMaxScaler
# 【关键修复】将 Line2D 放在顶部导入，防止作用域错误
from matplotlib.lines import Line2D


# ==========================================
# 0. 顶刊绘图风格设置
# ==========================================
def set_pub_style():
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white", {"axes.spines.left": True, "axes.spines.bottom": True,
                            "axes.spines.right": False, "axes.spines.top": False})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


set_pub_style()


# ==========================================
# 1. 多城市数据加载器
# ==========================================
class MultiCityLoader:
    def __init__(self, base_path, cities):
        self.base_path = base_path
        self.cities = cities

    def normalize_date(self, df, col='date'):
        if df.empty or col not in df.columns: return df
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if df[col].max() > 10000:
            temp = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
            min_date = temp.min()
            df['day'] = (temp - min_date).dt.days.fillna(0).astype(int)
        else:
            df['day'] = df[col].astype(int)
        return df

    def load_data(self):
        print("Loading data...")
        all_inf, all_mig = [], []

        for city in self.cities:
            city_path = os.path.join(self.base_path, city)
            if not os.path.exists(city_path): continue

            # Infection
            f_inf = os.path.join(city_path, 'infection.csv')
            if os.path.exists(f_inf):
                df = pd.read_csv(f_inf, names=['date', 'region', 'cnt'], header=None)
                df = self.normalize_date(df)
                daily = df.groupby('day')['cnt'].sum().reset_index()
                daily['city'] = city
                daily['cnt_smooth'] = daily['cnt'].rolling(window=3, min_periods=1).mean()
                all_inf.append(daily)

            # Migration
            f_mig = os.path.join(city_path, 'migration.csv')
            if os.path.exists(f_mig):
                df = pd.read_csv(f_mig, names=['date', 'f', 't', 'idx'], header=None)
                df = self.normalize_date(df)
                daily_m = df.groupby('day')['idx'].sum().reset_index()
                daily_m['city'] = city
                daily_m['idx_smooth'] = daily_m['idx'].rolling(window=3, min_periods=1).mean()
                all_mig.append(daily_m)

        df_inf = pd.concat(all_inf, ignore_index=True) if all_inf else pd.DataFrame()
        df_mig = pd.concat(all_mig, ignore_index=True) if all_mig else pd.DataFrame()
        return df_inf, df_mig


# ==========================================
# 2. 可视化模块
# ==========================================

def plot_ridge_joyplot(df_inf):
    """图1: 山脊图"""
    print("Plotting Ridge Plot...")
    if df_inf.empty: return

    cities = df_inf['city'].unique()
    n_cities = len(cities)
    fig, axes = plt.subplots(n_cities, 1, figsize=(10, 8), sharex=True)
    if n_cities == 1: axes = [axes]
    fig.subplots_adjust(hspace=-0.4)

    palette = sns.color_palette("coolwarm", n_cities)
    max_day = df_inf['day'].max()

    for i, city in enumerate(cities):
        ax = axes[i]
        subset = df_inf[df_inf['city'] == city]
        ax.fill_between(subset['day'], 0, subset['cnt_smooth'], color=palette[i], alpha=0.85)
        ax.plot(subset['day'], subset['cnt_smooth'], color='white', lw=1.5)
        ax.text(0.01, 0.4, city, transform=ax.transAxes, fontweight='bold', fontsize=12, ha='left')
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.patch.set_alpha(0)

    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('Relative Day', fontweight='bold')
    axes[-1].set_xlim(0, max_day)
    fig.suptitle('Epidemic Peaks Across Cities', y=0.92, fontsize=16, fontweight='bold')
    plt.show()


def plot_phase_trajectory(df_inf, df_mig):
    """
    图2: 相空间轨迹图 (Log Scale 版)
    """
    print("Plotting Phase Trajectory (Log Scale)...")
    if df_inf.empty or df_mig.empty: return

    merged = pd.merge(df_inf, df_mig, on=['city', 'day'], how='inner')
    if merged.empty: return

    cities = merged['city'].unique()

    fig, ax = plt.subplots(figsize=(11, 9))

    legend_elements = []
    cmap = plt.get_cmap('viridis_r')

    for i, city in enumerate(cities):
        # 增加平滑度
        subset = merged[merged['city'] == city].sort_values('day').copy()
        subset['cnt_smooth'] = subset['cnt'].rolling(window=5, min_periods=1).mean()
        subset['idx_smooth'] = subset['idx'].rolling(window=5, min_periods=1).mean()

        if len(subset) < 2: continue

        x = subset['idx_smooth'].values
        y = subset['cnt_smooth'].values
        t = subset['day'].values

        # 绘制彩色轨迹
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(t.min(), t.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.85)
        lc.set_array(t)
        lc.set_linewidth(2.5)
        lc.set_capstyle('round')

        ax.add_collection(lc)

        # 绘制起点和终点
        ax.scatter(x[0], y[0], color=cmap(0.0), s=60, edgecolors='black', zorder=10, marker='o', label='_nolegend_')
        ax.scatter(x[-1], y[-1], color=cmap(1.0), s=100, edgecolors='black', zorder=10, marker='*', label='_nolegend_')

        # 自定义图例句柄 (现在 Line2D 已经正确导入)
        legend_elements.append(Line2D([0], [0], color='gray', lw=2, label=city))

    # 设置 Y 轴为对称对数坐标，解决量级差异
    ax.set_yscale('symlog', linthresh=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    ax.autoscale()
    ax.margins(0.05)

    cbar = plt.colorbar(lc, ax=ax, label='Time Evolution (Days)', pad=0.02)

    # 城市图例
    ax.legend(handles=legend_elements, loc='upper right', title='City ID', frameon=True)

    # 起点终点说明图例
    legend_markers = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.0), markersize=8, markeredgecolor='k',
               label='Start (Day 0)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=cmap(1.0), markersize=10, markeredgecolor='k',
               label='Current (Latest)')
    ]
    # 添加第二个图例
    first_legend = ax.get_legend()
    ax.add_artist(first_legend)
    ax.legend(handles=legend_markers, loc='upper left', frameon=True)

    ax.set_xlabel('Migration Index (Mobility)', fontweight='bold', fontsize=12)
    ax.set_ylabel('New Infections (Log Scale)', fontweight='bold', fontsize=12)
    ax.set_title('Phase Space Trajectories: Hysteresis Loops', fontsize=14, pad=15)

    plt.tight_layout()
    plt.show()


def plot_standardized_heatmap(df_inf):
    """图3: 热力图"""
    print("Plotting Normalized Heatmap...")
    if df_inf.empty: return

    pivot = df_inf.pivot_table(index='city', columns='day', values='cnt_smooth', aggfunc='sum').fillna(0)
    # Z-Score
    pivot_norm = pivot.apply(lambda x: (x - x.mean()) / (x.std() + 1e-5), axis=1)

    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot_norm, cmap='RdBu_r', center=0, vmin=-2, vmax=3,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Z-Score'})

    plt.title('Spatiotemporal Synchronization (Standardized)', fontsize=14, pad=15)
    plt.xlabel('Relative Day', fontweight='bold')
    plt.ylabel('City ID', fontweight='bold')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_bubble_matrix(df_inf, df_mig):
    """
    图2 (新版): 时序气泡矩阵图 (Temporal Bubble Matrix)

    维度编码:
    - X轴: 时间 (Day)
    - Y轴: 城市 (City)
    - 气泡大小: 感染人数 (Infection) -> 重点突出发病规模
    - 气泡颜色: 迁徙指数 (Migration) -> 展示流动性强弱
    """
    print("Plotting Bubble Matrix...")
    if df_inf.empty or df_mig.empty: return

    # 1. 数据合并
    merged = pd.merge(df_inf, df_mig, on=['city', 'day'], how='inner')
    if merged.empty: return

    # 2. 数据平滑 (让视觉效果更好)
    # 对每个城市分别计算平滑
    merged['cnt_smooth'] = merged.groupby('city')['cnt'].transform(lambda x: x.rolling(3, 1).mean())
    merged['idx_smooth'] = merged.groupby('city')['idx'].transform(lambda x: x.rolling(3, 1).mean())

    # 3. 准备绘图
    fig, ax = plt.subplots(figsize=(13, 7))

    cities = merged['city'].unique()
    # 将城市转为数字索引以便绘图: 0, 1, 2...
    city_map = {city: i for i, city in enumerate(cities)}
    merged['city_y'] = merged['city'].map(city_map)

    # 4. 定义气泡大小 (Size Scaling)
    # 由于City A可能有10万，City E只有100，线性映射会导致City E看不见
    # 这里使用 power scale (0.5次方 即开根号) 或者 log scale 来压缩差距
    # 也可以简单地除以最大值 * 因子
    max_inf = merged['cnt_smooth'].max()
    # s = (val / max) * scale
    # 为了让小城市也能被看见，加一个基础大小 base_size
    base_size = 30
    scale_factor = 800
    # 使用平方根缩放，既能体现差异，又不会让大球大得离谱
    merged['size'] = (np.sqrt(merged['cnt_smooth']) / np.sqrt(max_inf)) * scale_factor + base_size

    # 5. 绘制散点 (Scatter)
    # c=idx_smooth (颜色映射迁徙), s=size (大小映射感染)
    scatter = ax.scatter(
        x=merged['day'],
        y=merged['city_y'],
        s=merged['size'],
        c=merged['idx_smooth'],
        cmap='YlGnBu',  # 浅黄到深蓝，蓝色代表高迁徙
        alpha=0.85,
        edgecolors='gray',
        linewidth=0.5
    )

    # 6. 美化图表
    # 设置Y轴标签为城市名
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities, fontweight='bold', fontsize=12)

    # 设置X轴
    ax.set_xlabel('Time Evolution (Relative Day)', fontweight='bold', fontsize=12)
    ax.set_xlim(merged['day'].min() - 1, merged['day'].max() + 3)  # 留点边距

    # 添加网格线 (只显示X轴方向的虚线，方便看时间对齐)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    # 去除多余边框
    sns.despine(left=True, bottom=False)

    # --- 添加图例 (Legends) ---

    # 7.1 颜色条 (Colorbar) -> 代表迁徙指数
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Migration Index (Color Depth)', fontweight='bold')

    # 7.2 大小图例 (Size Legend) -> 代表感染人数
    # 制作几个示例大小：最大值的10%, 50%, 100%
    # 逆运算还原真实数值
    legend_sizes = [0.1, 0.5, 1.0]
    legend_labels = [int(max_inf * r) for r in legend_sizes]
    # 计算对应的气泡显示大小
    legend_area_sizes = [(np.sqrt(val) / np.sqrt(max_inf)) * scale_factor + base_size for val in legend_labels]

    # 手动创建图例句柄
    legend_elements = []
    for area, label in zip(legend_area_sizes, legend_labels):
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='gray', label=f'{label}',
                                      markersize=np.sqrt(area), alpha=0.6))  # markersize大概是面积的开方

    # 添加大小图例到上方
    legend1 = ax.legend(handles=legend_elements, title="New Infections (Bubble Size)",
                        loc='upper center', bbox_to_anchor=(0.5, 1.15),
                        ncol=3, frameon=False)

    ax.set_title('Spatio-Temporal Evolution: Migration vs. Infection', fontsize=14, y=1.15, fontweight='bold')

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    BASE_DIR = r"C:\Users\21626\Downloads\train_data\train_data"
    TARGET_CITIES = ['city_A', 'city_B', 'city_C', 'city_D', 'city_E']

    if os.path.exists(BASE_DIR):
        loader = MultiCityLoader(BASE_DIR, TARGET_CITIES)
        df_inf, df_mig = loader.load_data()

        if not df_inf.empty:
            plot_ridge_joyplot(df_inf)
            plot_phase_trajectory(df_inf, df_mig)
            plot_standardized_heatmap(df_inf)
            plot_bubble_matrix(df_inf, df_mig)
            print("Done.")
        else:
            print("No data loaded.")