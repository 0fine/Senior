import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os


# ==========================================
# 1. 顶刊绘图风格设置
# ==========================================
def set_pub_style():
    sns.set_context("paper", font_scale=1.4)
    # 极坐标图不要网格线干扰，后面单独设置
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    # 兼容多种中文字体，防止乱码
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


set_pub_style()


# ==========================================
# 2. 数据加载与清洗
# ==========================================
def load_and_clean_weather(file_path):
    print("Loading and cleaning weather data...")

    # 1. 定义列名
    cols = ['date', 'hour', 'temp', 'humidity', 'wind_dir', 'wind_spd', 'wind_force', 'weather']

    # 读取数据
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path, names=cols, header=None)

    # 2. 生成完整的时间戳
    # 将日期和小时合并。注意：hour 列可能是整数 0-23
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + df['hour'].astype(str).str.zfill(2),
                                    format='%Y%m%d%H', errors='coerce')

    # 3. 清洗湿度 (去除 '%' 并转数字)
    if df['humidity'].dtype == object:
        df['humidity'] = df['humidity'].astype(str).str.replace('%', '', regex=False)
    df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')

    # 4. 清洗风力 (提取数字)
    # 提取 <3 中的 3, 或者 12km/h 中的 12 (如果有的话，这里假设是等级)
    df['wind_force_clean'] = df['wind_force'].astype(str).str.extract(r'(\d+)').astype(float)

    # 5. 清洗风向 (填补空缺)
    df['wind_dir'] = df['wind_dir'].fillna('Unknown')

    # 简单插值填补缺失的气温和湿度
    df['temp'] = df['temp'].interpolate(method='linear')
    df['humidity'] = df['humidity'].interpolate(method='linear')

    return df


# ==========================================
# 3. 可视化绘图模块
# ==========================================

def plot_temp_humidity_dynamics(df):
    """ 图1: 气温湿度双轴图 """
    if df.empty: return

    # 取前72小时，避免太密集
    plot_data = df.iloc[:72].copy()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：气温
    color_temp = '#D62828'
    sns.lineplot(data=plot_data, x='datetime', y='temp', ax=ax1,
                 color=color_temp, linewidth=2.5, label='Temperature')
    ax1.set_ylabel('Temperature (°C)', color=color_temp, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_temp)
    ax1.set_xlabel('Time', fontweight='bold')

    # 右轴：湿度
    ax2 = ax1.twinx()
    color_hum = '#457B9D'
    ax2.fill_between(plot_data['datetime'], 0, plot_data['humidity'],
                     color=color_hum, alpha=0.2, label='Humidity')
    sns.lineplot(data=plot_data, x='datetime', y='humidity', ax=ax2,
                 color=color_hum, linewidth=1.5, linestyle='--')

    ax2.set_ylabel('Humidity (%)', color=color_hum, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_hum)
    ax2.set_ylim(0, 110)

    # 格式化时间轴
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))

    plt.title('Temporal Dynamics: Temperature vs Humidity', fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


def plot_wind_rose(df):
    """ 图2: 风向玫瑰图 (修复 UserWarning) """
    if df.empty: return

    # 1. 定义映射
    # 注意：Quiet (静风) 不参与角度计算
    dir_map = {
        'North': 0, 'Northeast': 45, 'East': 90, 'Southeast': 135,
        'South': 180, 'Southwest': 225, 'West': 270, 'Northwest': 315
    }

    # 2. 统计频次
    wind_counts = df['wind_dir'].value_counts().reset_index()
    wind_counts.columns = ['direction', 'count']

    # 3. 分离静风
    quiet_rows = wind_counts[wind_counts['direction'].str.contains('Quiet', case=False, na=False)]
    quiet_count = quiet_rows['count'].sum()

    # 4. 筛选有向风
    active_winds = wind_counts[wind_counts['direction'].isin(dir_map.keys())].copy()

    # 转换为弧度
    active_winds['angle'] = active_winds['direction'].map(dir_map) * np.pi / 180

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # 设置正北为0度，顺时针
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # 绘制柱状图
    # align='center' 确保柱子中心对准刻度
    bars = ax.bar(active_winds['angle'], active_winds['count'],
                  width=0.5, bottom=0.0, color='#2A9D8F', alpha=0.8, edgecolor='white')

    # ==========================================
    # 核心修复: 先设置 Ticks 位置，再设置 Labels
    # ==========================================
    # 8个方向的弧度位置: 0, 45, 90... 转弧度
    fixed_ticks = np.radians(np.arange(0, 360, 45))
    fixed_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    ax.set_xticks(fixed_ticks)
    ax.set_xticklabels(fixed_labels)

    # 标注静风
    total = df.shape[0]
    quiet_pct = (quiet_count / total) * 100 if total > 0 else 0
    plt.text(0, 0, f'Quiet\n{quiet_pct:.1f}%', ha='center', va='center', fontweight='bold')

    plt.title(f'Wind Direction Distribution (n={total})', fontsize=14, y=1.08)
    plt.tight_layout()
    plt.show()


def plot_weather_correlation_matrix(df):
    """ 图3: 相关性密度图 """
    if df.empty: return

    # 准备数据
    clean_df = df[['temp', 'humidity', 'wind_force_clean']].dropna()
    if clean_df.shape[0] < 10:
        print("Not enough data for correlation plot.")
        return

    # 临时切换风格以适应 JointGrid
    with sns.axes_style("white"):
        g = sns.JointGrid(data=clean_df, x="temp", y="humidity", height=7, space=0)

        # 核心图：六边形密度
        g.plot_joint(plt.hexbin, gridsize=20, cmap="Blues", mincnt=1, edgecolors="white")

        # 边缘图：密度曲线
        sns.kdeplot(data=clean_df, x="temp", ax=g.ax_marg_x, fill=True, color="#457B9D")
        sns.kdeplot(data=clean_df, y="humidity", ax=g.ax_marg_y, fill=True, color="#457B9D")

        # 回归线
        sns.regplot(data=clean_df, x="temp", y="humidity", scatter=False, ax=g.ax_joint,
                    color="#D62828", line_kws={'linestyle': '--', 'linewidth': 2})

        # 标注相关系数
        corr = clean_df['temp'].corr(clean_df['humidity'])
        g.ax_joint.text(0.05, 0.95, f'Pearson r = {corr:.2f}', transform=g.ax_joint.transAxes,
                        fontsize=12, fontweight='bold', va='top')

        g.ax_joint.set_xlabel('Temperature (°C)', fontweight='bold')
        g.ax_joint.set_ylabel('Humidity (%)', fontweight='bold')

    plt.show()


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 请确认这个路径是正确的
    FILE_PATH = r"C:\Users\21626\Downloads\train_data\train_data\city_A\weather.csv"

    if os.path.exists(FILE_PATH):
        df_weather = load_and_clean_weather(FILE_PATH)

        print("Data Snippet (Cleaned):")
        print(df_weather[['datetime', 'temp', 'humidity', 'wind_dir', 'wind_force_clean']].head())

        plot_temp_humidity_dynamics(df_weather)
        plot_wind_rose(df_weather)
        plot_weather_correlation_matrix(df_weather)
    else:
        print(f"File not found: {FILE_PATH}")