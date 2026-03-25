import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


# ==========================================
# 1. 风格配置
# ==========================================
def set_pub_style():
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


set_pub_style()


# ==========================================
# 2. 数据加载模块
# ==========================================
class CityDataLoader:
    def __init__(self, base_path, city_name="city_A"):
        self.city_path = os.path.join(base_path, city_name)
        self.city_name = city_name

    def normalize_to_relative_day(self, df, col_name='date'):
        """将不同格式的日期（YYYYMMDD 或 0,1,2）统一转换为 relative_day"""
        if df.empty or col_name not in df.columns:
            return df

        try:
            # 先强制转为数值，无法转换的变NaN
            numeric_dates = pd.to_numeric(df[col_name], errors='coerce')

            # 判断是否解析成功
            if numeric_dates.isna().all():
                # 如果全是NaN，说明可能是字符串日期 "2020-01-01"
                temp_dt = pd.to_datetime(df[col_name], errors='coerce')
                min_dt = temp_dt.min()
                if pd.isna(min_dt):
                    # 极端的兜底：如果完全无法解析，就生成序列索引
                    df['relative_day'] = range(len(df))
                    df['date_dt'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df['relative_day'], unit='D')
                else:
                    df['relative_day'] = (temp_dt - min_dt).dt.days
                    df['date_dt'] = temp_dt

            elif numeric_dates.max() > 10000:
                # YYYYMMDD 格式 (如 20200101)
                temp_dt = pd.to_datetime(numeric_dates, format='%Y%m%d', errors='coerce')
                min_dt = temp_dt.min()
                df['relative_day'] = (temp_dt - min_dt).dt.days
                df['date_dt'] = temp_dt

            else:
                # 已经是相对天数 (0, 1, 2)
                df['relative_day'] = numeric_dates
                # 构造伪日期用于X轴显示
                df['date_dt'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(numeric_dates, unit='D')

        except Exception as e:
            print(f"Date normalization warning: {e}")
            df['relative_day'] = 0

        return df

    def load_all(self):
        print(f"Loading data for {self.city_name} from {self.city_path}...")

        # 1. Infection
        f_inf = os.path.join(self.city_path, 'infection.csv')
        if os.path.exists(f_inf):
            df_inf = pd.read_csv(f_inf, names=['date', 'region_id', 'new_infections'], header=None)
            df_inf = self.normalize_to_relative_day(df_inf, 'date')
        else:
            print(f"Error: {f_inf} not found.")
            df_inf = pd.DataFrame()

        # 2. Migration
        f_mig = os.path.join(self.city_path, 'migration.csv')
        if os.path.exists(f_mig):
            df_mig = pd.read_csv(f_mig, names=['date', 'from_city', 'to_city', 'mig_index'], header=None)
            df_mig = self.normalize_to_relative_day(df_mig, 'date')
        else:
            df_mig = pd.DataFrame()

        # 3. Density (限制行数)
        f_den = os.path.join(self.city_path, 'density.csv')
        if os.path.exists(f_den):
            df_den = pd.read_csv(f_den, names=['date', 'hour', 'lon', 'lat', 'flow_index'], header=None, nrows=200000)
            df_den = self.normalize_to_relative_day(df_den, 'date')
        else:
            df_den = pd.DataFrame()

        # 4. Grid Attribute
        f_grid = os.path.join(self.city_path, 'grid_attr.csv')
        if os.path.exists(f_grid):
            df_grid = pd.read_csv(f_grid, names=['lon', 'lat', 'region_id'], header=None)
        else:
            df_grid = pd.DataFrame()

        # 5. Weather
        f_wea = os.path.join(self.city_path, 'weather.csv')
        if os.path.exists(f_wea):
            df_wea = pd.read_csv(f_wea, names=['date', 'hour', 'temp', 'humidity', 'wind_dir', 'wind_spd', 'wind_force',
                                               'weather_type'], header=None)
            df_wea = self.normalize_to_relative_day(df_wea, 'date')
        else:
            df_wea = pd.DataFrame()

        print("Data loaded successfully.")
        return df_inf, df_mig, df_den, df_grid, df_wea


# ==========================================
# 3. 可视化绘图模块
# ==========================================

def plot_epidemic_migration_dynamics(df_inf, df_mig, city_name):
    """ 图1: 疫情与迁徙 """
    if 'relative_day' not in df_inf.columns or 'relative_day' not in df_mig.columns:
        return

    daily_inf = df_inf.groupby('relative_day')['new_infections'].sum().reset_index()
    daily_mig = df_mig.groupby('relative_day')['mig_index'].sum().reset_index()

    # 映射日期
    if 'date_dt' in df_inf.columns:
        date_map = df_inf[['relative_day', 'date_dt']].drop_duplicates().set_index('relative_day')
    else:
        # Fallback
        daily_inf['date_dt'] = daily_inf['relative_day']
        date_map = None

    merged = pd.merge(daily_inf, daily_mig, on='relative_day', how='inner')

    if date_map is not None:
        merged['date_dt'] = merged['relative_day'].map(date_map['date_dt'])

    if merged.empty:
        print("Plot 1 Skipped: No overlapping dates.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_inf = '#E63946'
    ax1.bar(merged['date_dt'], merged['new_infections'], color=color_inf, alpha=0.6, label='Infections')
    ax1.set_ylabel('New Infections', color=color_inf, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_inf)

    # 日期格式化
    if date_map is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))

    ax2 = ax1.twinx()
    color_mig = '#1D3557'
    ax2.plot(merged['date_dt'], merged['mig_index'], color=color_mig, marker='o', markersize=4, label='Migration')
    ax2.set_ylabel('Migration Index', color=color_mig, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_mig)

    plt.title(f'Dynamics in {city_name}')
    plt.tight_layout()
    plt.show()


def plot_spatial_heatmap(df_inf):
    """ 图2: 时空热力图 """
    print("Preparing Spatial Heatmap...")
    if df_inf.empty: return

    pivot = df_inf.pivot_table(index='region_id', columns='date_dt', values='new_infections', aggfunc='sum').fillna(0)

    if pivot.size == 0:
        print("Error: Pivot table is empty.")
        return

    plt.figure(figsize=(12, 6))

    x_labels = [d.strftime('%m-%d') if hasattr(d, 'strftime') else str(d) for d in pivot.columns]

    ax = sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.0, xticklabels=x_labels,
                     cbar_kws={'label': 'New Infections'})

    plt.title('Spatiotemporal Distribution of Infections')
    plt.ylabel('Region ID')
    plt.xlabel('Date')

    if len(pivot.columns) > 20:
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        n = 5
        ax.set_xticks(ticks[::n])
        ax.set_xticklabels(labels[::n], rotation=0)

    plt.tight_layout()
    plt.show()


def plot_grid_density_scatter(df_grid, df_den):
    """ 图3: 网格分布 """
    print("Preparing Grid Map...")
    if df_grid.empty or df_den.empty: return

    # 强制转为数值，防止 merge 失败
    for df in [df_grid, df_den]:
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')

    grid_flow = df_den.groupby(['lon', 'lat'])['flow_index'].mean().reset_index()

    df_grid['lon_r'] = df_grid['lon'].round(4)
    df_grid['lat_r'] = df_grid['lat'].round(4)
    grid_flow['lon_r'] = grid_flow['lon'].round(4)
    grid_flow['lat_r'] = grid_flow['lat'].round(4)

    merged = pd.merge(df_grid, grid_flow, on=['lon_r', 'lat_r'], how='inner')

    if merged.empty:
        merged = df_grid
        size_col = None
    else:
        size_col = 'flow_index'

    plt.figure(figsize=(8, 7))
    sns.scatterplot(data=merged, x='lon_x', y='lat_x', hue='region_id',
                    size=size_col, palette='tab20', legend=False, alpha=0.8, edgecolor=None)
    plt.title('Spatial Grid Distribution')
    plt.axis('equal')
    plt.show()


def plot_weather_correlation(df_inf, df_wea):
    """ 图4: 天气相关性 (已彻底修复类型错误) """
    print("Preparing Weather Correlation...")
    if df_inf.empty or df_wea.empty:
        print("Weather or Infection data missing.")
        return

    target_cols = ['temp', 'humidity', 'wind_spd', 'wind_force']
    available_cols = [c for c in target_cols if c in df_wea.columns]

    if not available_cols:
        print("No weather columns found.")
        return

    # ==========================================
    # 核心修复: 强制将天气列转换为数值类型
    # ==========================================
    for col in available_cols:
        # errors='coerce' 会将所有非数字字符变成 NaN，从而避免 groupby 报错
        df_wea[col] = pd.to_numeric(df_wea[col], errors='coerce')

    # 筛选出还有有效数据的列
    valid_cols = [c for c in available_cols if df_wea[c].notna().any()]

    if not valid_cols:
        print("Error: Weather columns contain no numeric data.")
        return

    # 现在列里全是数字，mean() 绝对安全
    daily_wea = df_wea.groupby('relative_day')[valid_cols].mean().reset_index()

    daily_inf = df_inf.groupby('relative_day')['new_infections'].sum().reset_index()

    merged = pd.merge(daily_inf, daily_wea, on='relative_day')

    if merged.shape[0] < 2:
        print("Not enough data points for correlation.")
        return

    corr = merged.drop(columns=['relative_day']).corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", center=0,
                linewidths=0.5, cbar_kws={"shrink": .7})
    plt.title('Correlation: Weather Factors vs. Infection')
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    BASE_DIR = r"C:\Users\21626\Downloads\train_data\train_data"
    TARGET_CITY = "city_A"

    if not os.path.exists(os.path.join(BASE_DIR, TARGET_CITY)):
        print(f"Path Error: {os.path.join(BASE_DIR, TARGET_CITY)} does not exist.")
    else:
        try:
            loader = CityDataLoader(BASE_DIR, TARGET_CITY)
            df_inf, df_mig, df_den, df_grid, df_wea = loader.load_all()

            plot_epidemic_migration_dynamics(df_inf, df_mig, TARGET_CITY)
            plot_spatial_heatmap(df_inf)
            plot_grid_density_scatter(df_grid, df_den)
            plot_weather_correlation(df_inf, df_wea)

            print("All visualizations complete.")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback

            traceback.print_exc()