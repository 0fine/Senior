import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import os

# ==========================================
# 1. 系统配置与学术风格定义
# ==========================================
st.set_page_config(layout="wide", page_title="城市群疫情高维时空可视分析系统")

# 注入CSS：论文级 Side Note (备注栏) 样式
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    /* 备注栏容器 */
    .caption-box {
        background-color: #1f2937;
        border-left: 4px solid #3b82f6; /* 蓝色科技边框 */
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* 备注标题 */
    .caption-title {
        color: #60a5fa;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    /* 备注正文 */
    .caption-text {
        color: #e5e7eb;
        font-size: 14px;
        line-height: 1.6;
    }
    /* 重点高亮 */
    .highlight { color: #f87171; font-weight: bold; }
    /* 调整布局间距 */
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据路径与工具函数
# ==========================================
BASE_DIR = r"C:\Users\21626\Downloads\train_data\train_data"
CITY_LIST = ['city_A', 'city_B', 'city_C', 'city_D', 'city_E']


def smart_date_parser(date_series):
    """
    智能日期解析：兼容 20200101 (int/str) 和 2020-01-01
    """
    s = date_series.astype(str).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')


# ==========================================
# 3. 鲁棒的数据加载引擎 (关键修复)
# ==========================================

@st.cache_data
def get_city_meta():
    """获取各城市中心坐标"""
    city_coords = {}
    for city in CITY_LIST:
        path = os.path.join(BASE_DIR, city, "grid_attr.csv")
        try:
            if os.path.exists(path):
                # 读少量数据估算中心
                df = pd.read_csv(path, header=None, nrows=2000)
                # 假设前两列是 lon, lat
                lon = pd.to_numeric(df.iloc[:, 0], errors='coerce').mean()
                lat = pd.to_numeric(df.iloc[:, 1], errors='coerce').mean()
                if 70 < lon < 140 and 10 < lat < 60:
                    city_coords[city] = {"lon": lon, "lat": lat}
                else:
                    raise ValueError
            else:
                raise FileNotFoundError
        except:
            # 默认坐标 (Fallback)
            defaults = {
                'city_A': (116.4, 39.9), 'city_B': (121.5, 31.2),
                'city_C': (114.3, 30.6), 'city_D': (113.3, 23.1), 'city_E': (104.1, 30.7)
            }
            pos = defaults.get(city, (114.0, 30.0))
            city_coords[city] = {"lon": pos[0], "lat": pos[1]}
    return city_coords


@st.cache_data
def load_global_data(city_coords):
    """加载 infection 和 migration (宏观)"""
    inf_frames = []
    mig_frames = []

    debug_logs = []  # 记录错误以便排查

    for city in CITY_LIST:
        c_path = os.path.join(BASE_DIR, city)

        # --- Infection 读取修复 ---
        # 目标格式: date, rid, count
        p = os.path.join(c_path, "infection.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, header=None)

                # 🚨 关键修复：根据列数判断列映射
                if df.shape[1] == 4:
                    # 格式可能是: [CityID, RegionID, Date, Count] (基于您的报错信息)
                    # 映射: Date=2, RID=1, Count=3
                    df = df.iloc[:, [2, 1, 3]]
                    df.columns = ['date', 'rid', 'count']
                elif df.shape[1] == 3:
                    # 格式: [Date, RegionID, Count] (标准描述)
                    df.columns = ['date', 'rid', 'count']
                else:
                    # 异常情况，盲猜最后3列
                    df = df.iloc[:, -3:]
                    df.columns = ['date', 'rid', 'count']

                df['date'] = smart_date_parser(df['date'])
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
                valid = df.dropna(subset=['date'])

                if not valid.empty:
                    daily = valid.groupby('date')['count'].sum().reset_index()
                    daily['city'] = city
                    inf_frames.append(daily)
                else:
                    debug_logs.append(f"{city}: 解析后日期全为空，原始列数 {df.shape[1]}")
            except Exception as e:
                debug_logs.append(f"{city} 读取失败: {e}")

        # --- Migration 读取 ---
        p = os.path.join(c_path, "migration.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, header=None)
                if df.shape[1] >= 4:
                    # 取最后4列: Date, Dep, Arr, Index
                    df = df.iloc[:, -4:]
                    df.columns = ['date', 'dep', 'arr', 'idx']
                    df['date'] = smart_date_parser(df['date'])
                    df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
                    mig_frames.append(df.dropna(subset=['date']))
            except:
                pass

    df_inf = pd.concat(inf_frames) if inf_frames else pd.DataFrame()
    df_mig = pd.concat(mig_frames) if mig_frames else pd.DataFrame()

    # 映射坐标
    if not df_mig.empty:
        c_map = {k: (v['lon'], v['lat']) for k, v in city_coords.items()}

        def get_xy(name): return c_map.get(name, (0, 0))

        df_mig['f_xy'] = df_mig['dep'].map(get_xy)
        df_mig['t_xy'] = df_mig['arr'].map(get_xy)
        df_mig[['flon', 'flat']] = pd.DataFrame(df_mig['f_xy'].tolist(), index=df_mig.index)
        df_mig[['tlon', 'tlat']] = pd.DataFrame(df_mig['t_xy'].tolist(), index=df_mig.index)

    return df_inf, df_mig, debug_logs


@st.cache_data
def load_local_data(city, date_filter):
    """加载微观数据 (Density, Transfer, Attr, Weather)"""
    data = {}
    c_path = os.path.join(BASE_DIR, city)
    target_date_str = date_filter.strftime('%Y%m%d')

    # 1. Density (大文件按日期过滤)
    # 格式: date, hour, lon, lat, index (5列)
    try:
        p = os.path.join(c_path, "density.csv")
        if os.path.exists(p):
            # 预读大量行
            df = pd.read_csv(p, header=None, nrows=100000)
            if df.shape[1] >= 5:
                df = df.iloc[:, -5:]
                df.columns = ['date', 'hour', 'lon', 'lat', 'val']
                # 字符串匹配提速
                df = df[df['date'].astype(str) == target_date_str].copy()
                if not df.empty:
                    for c in ['hour', 'lon', 'lat', 'val']:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                    data['density'] = df
    except:
        pass
    if 'density' not in data: data['density'] = pd.DataFrame()

    # 2. Transfer (无日期，通用)
    # 格式: hour, slon, slat, elon, elat, val (6列)
    try:
        p = os.path.join(c_path, "transfer.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, header=None, nrows=20000)
            if df.shape[1] >= 6:
                df = df.iloc[:, -6:]
                df.columns = ['hour', 'slon', 'slat', 'elon', 'elat', 'val']
                for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                data['transfer'] = df
    except:
        pass
    if 'transfer' not in data: data['transfer'] = pd.DataFrame()

    # 3. Grid Attr
    try:
        p = os.path.join(c_path, "grid_attr.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, header=None, nrows=50000)
            if df.shape[1] >= 3:
                df = df.iloc[:, -3:]
                df.columns = ['lon', 'lat', 'rid']
                data['attr'] = df
    except:
        pass
    if 'attr' not in data: data['attr'] = pd.DataFrame()

    # 4. Weather
    try:
        p = os.path.join(c_path, "weather.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, header=None)
            if df.shape[1] >= 3:
                col_d = 0 if df.shape[1] == 8 else 1
                col_t = 2 if df.shape[1] == 8 else 3
                df['date'] = smart_date_parser(df.iloc[:, col_d])
                df['temp'] = pd.to_numeric(df.iloc[:, col_t], errors='coerce')
                data['weather'] = df.groupby('date')[['temp']].mean().reset_index()
    except:
        pass
    if 'weather' not in data: data['weather'] = pd.DataFrame()

    # 5. Infection (Region level)
    try:
        p = os.path.join(c_path, "infection.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, header=None)
            if df.shape[1] == 4:
                df = df.iloc[:, [2, 1, 3]]
            elif df.shape[1] == 3:
                pass  # 默认
            else:
                df = df.iloc[:, -3:]
            df.columns = ['date', 'rid', 'count']
            df['date'] = smart_date_parser(df['date'])
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            data['inf_region'] = df
    except:
        pass
    if 'inf_region' not in data: data['inf_region'] = pd.DataFrame()

    return data


# ==========================================
# 4. 页面主逻辑
# ==========================================

# --- 初始化 ---
meta = get_city_meta()
df_macro_inf, df_macro_mig, debug_logs = load_global_data(meta)

# 错误处理
if df_macro_inf.empty:
    st.error("❌ 数据加载失败。以下是调试信息：")
    for log in debug_logs: st.text(log)
    st.stop()

# --- 侧边栏 ---
st.sidebar.title("🎛️ 控制面板")
st.sidebar.markdown("---")

min_date = df_macro_inf['date'].min().date()
max_date = df_macro_inf['date'].max().date()
sel_date = st.sidebar.slider("📅 日期选择 (Global Date)", min_date, max_date, min_date)
sel_date_ts = pd.Timestamp(sel_date)

sel_city = st.sidebar.selectbox("🏙️ 目标城市 (Target City)", CITY_LIST)

# --- 数据切片 ---
daily_inf = df_macro_inf[df_macro_inf['date'] == sel_date_ts]
daily_mig = df_macro_mig[df_macro_mig['date'] == sel_date_ts]
local_data = load_local_data(sel_city, sel_date)

st.title("🦠 城市群疫情高维时空可视分析系统")
st.markdown("---")

# ==========================================
# VIEW 1 & 2: 宏观态势 (保持不变)
# ==========================================
st.subheader("1. 宏观态势感知 (Macro-Awareness)")
c1, c2 = st.columns([3, 1])

with c1:
    # Map
    nodes = []
    for city in CITY_LIST:
        coords = meta.get(city, {'lon': 0, 'lat': 0})
        val = daily_inf[daily_inf['city'] == city]['count'].sum()
        nodes.append({'name': city, 'lon': coords['lon'], 'lat': coords['lat'], 'val': int(val)})

    layers = [
        pdk.Layer(
            "ScatterplotLayer", pd.DataFrame(nodes),
            get_position=["lon", "lat"], get_radius="val * 50 + 5000",
            get_fill_color=[255, 50, 50, 180], stroked=True, get_line_color=[255, 255, 255],
            pickable=True
        )
    ]
    if not daily_mig.empty:
        layers.append(pdk.Layer(
            "ArcLayer", daily_mig,
            get_source_position=["flon", "flat"], get_target_position=["tlon", "tlat"],
            get_width="idx / 10", get_source_color=[0, 255, 255, 150], get_target_color=[255, 0, 255, 150]
        ))

    st.pydeck_chart(pdk.Deck(
        layers=layers, initial_view_state=pdk.ViewState(latitude=35, longitude=110, zoom=3.5),
        map_style="mapbox://styles/mapbox/dark-v10", tooltip={"text": "{name}: {val}"}
    ))

with c2:
    # Trend
    trend = df_macro_inf[df_macro_inf['city'] == sel_city].sort_values('date')
    w = local_data['weather']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=trend['date'], y=trend['count'], name='新增病例', fill='tozeroy', line=dict(color='#ff4b4b')))
    if not w.empty:
        merged = pd.merge(trend, w, on='date')
        fig.add_trace(go.Scatter(x=merged['date'], y=merged['temp'], name='气温', yaxis='y2',
                                 line=dict(color='#3b82f6', dash='dot')))
    fig.update_layout(template='plotly_dark', height=250, margin=dict(l=0, r=0, t=20, b=0),
                      legend=dict(orientation="h", y=1.2))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="caption-box">
        <div class="caption-title">🗺️ 宏观概览 (Macro View)</div>
        <div class="caption-text">
        展示了<b>城市间传播网络</b>与<b>时序演化</b>。
        <ul>
            <li><b>左图:</b> OD流向与疫情分布。弧线代表人口流动，圆点大小代表当日确诊量。</li>
            <li><b>右图:</b> 气温与疫情的双变量分析，探索环境相关性。</li>
        </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# VIEW 3: 城市脉搏 (Urban Pulse) - 重绘
# 重点：24小时动态节律
# ==========================================
st.subheader(f"2. 微观：城市脉搏与人群节律 ({sel_city})")
c1, c2 = st.columns([3, 1])

df_den = local_data['density']

with c1:
    if not df_den.empty and 'hour' in df_den.columns:
        hours = sorted(df_den['hour'].unique())
        if hours:
            # 24H 时间滑块
            cur_hour = st.select_slider("⏱️ 拖动滑块查看城市24小时“呼吸”节奏 (Time Slider)", options=hours)

            # 过滤
            layer_data = df_den[df_den['hour'] == cur_hour]

            # 使用 HeatmapLayer
            layers_v3 = [
                pdk.Layer(
                    "HeatmapLayer", layer_data,
                    get_position=["lon", "lat"], get_weight="val",
                    radius_pixels=60, intensity=1.5, threshold=0.1, opacity=0.8
                )
            ]

            center_lat = layer_data['lat'].mean()
            center_lon = layer_data['lon'].mean()

            st.pydeck_chart(pdk.Deck(
                layers=layers_v3,
                initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10, pitch=0),
                map_style="mapbox://styles/mapbox/dark-v10",
                tooltip={"text": "密度: {val}"}
            ))
        else:
            st.warning("数据中缺少小时信息。")
    else:
        st.info(f"{sel_date} 暂无密度数据，请更换日期尝试。")

with c2:
    peak = df_den['val'].max() if not df_den.empty else 0
    st.markdown(f"""
    <div class="caption-box">
        <div class="caption-title">🔥 城市脉搏 (Urban Pulse)</div>
        <div class="caption-text">
        <span class="highlight">时间维度分析：</span> 从“天”深入到“小时”。
        <br><br>
        通过上方滑块，您可以像播放电影一样观察城市内部的人群流动：
        <ul>
            <li><b>早高峰 (7-9点):</b> 居住区向工作区汇聚。</li>
            <li><b>晚高峰 (17-19点):</b> 逆向流动。</li>
        </ul>
        <br>
        <b>当前最高密度:</b> {peak}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# VIEW 4: 传播骨架 (Transfer) - 重绘
# 重点：网络结构
# ==========================================
st.subheader("3. 介观：城市内部传播骨架网")
c1, c2 = st.columns([1, 3])

df_trans = local_data['transfer']

with c1:
    st.markdown("""
    <div class="caption-box">
        <div class="caption-title">🕸️ 传播骨架 (Mobility Skeleton)</div>
        <div class="caption-text">
        基于 <b>Graph Visualization</b> 技术展示城市内部的 OD (Origin-Destination) 轨迹。
        <br><br>
        <b>视觉编码:</b>
        <ul>
            <li><b>3D 弧线:</b> 连接出发与到达网格。</li>
            <li><b>颜色:</b> 青色(出发) -> 黄色(到达)。</li>
            <li><b>高度:</b> 映射迁移强度。</li>
        </ul>
        <br>
        <span class="highlight">分析任务:</span> 识别连接不同高危区域的“超级传播走廊”。
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    if not df_trans.empty:
        # 过滤 Top 1000 强度的连接
        top_trans = df_trans.nlargest(1000, 'val')

        layer_arc_micro = pdk.Layer(
            "ArcLayer", top_trans,
            get_source_position=["slon", "slat"],
            get_target_position=["elon", "elat"],
            get_width="val / 2",
            get_tilt=15,
            get_source_color=[0, 255, 200, 100],
            get_target_color=[255, 200, 0, 100],
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer_arc_micro],
            initial_view_state=pdk.ViewState(latitude=meta[sel_city]['lat'], longitude=meta[sel_city]['lon'], zoom=10.5,
                                             pitch=50),
            map_style="mapbox://styles/mapbox/dark-v10",
        ))
    else:
        st.info("暂无城市内部轨迹数据。")

# ==========================================
# VIEW 5: 区域风险演化 (Risk Evolution) - 重绘
# 重点：时序变化 + 气泡图动画
# ==========================================
st.markdown("---")
st.subheader("4. 深度洞察：区域风险的时空演化")
c1, c2 = st.columns(2)

df_inf_reg = local_data['inf_region']

# 准备 View 5 数据：聚合各区域的 [日期, 累计确诊, 当日新增]
if not df_inf_reg.empty:
    # 累计确诊
    df_inf_reg = df_inf_reg.sort_values(['rid', 'date'])
    df_inf_reg['cum_cases'] = df_inf_reg.groupby('rid')['count'].cumsum()
    # 截取最近7天数据用于展示
    end_date = sel_date_ts
    start_date = end_date - pd.Timedelta(days=14)
    mask = (df_inf_reg['date'] >= start_date) & (df_inf_reg['date'] <= end_date)
    df_evo = df_inf_reg[mask].copy()
    df_evo['rid'] = df_evo['rid'].astype(str)
    df_evo['date_str'] = df_evo['date'].dt.strftime('%Y-%m-%d')
else:
    df_evo = pd.DataFrame()

with c1:
    if not df_evo.empty:
        # 动画气泡图 (Animated Bubble Chart)
        fig_anim = px.scatter(
            df_evo, x="count", y="cum_cases",
            animation_frame="date_str", animation_group="rid",
            size="cum_cases", color="rid",
            hover_name="rid",
            log_x=False, size_max=40,
            range_x=[0, df_evo['count'].max() * 1.1],
            range_y=[0, df_evo['cum_cases'].max() * 1.1],
            labels={'count': '当日新增', 'cum_cases': '累计确诊', 'date_str': '日期'}
        )
        fig_anim.update_layout(
            template='plotly_dark',
            title="区域疫情爆发轨迹 (点击播放)",
            height=400
        )
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info("暂无区域级时序数据。")

    st.markdown("""
    <div class="caption-box">
        <div class="caption-title">🎬 动态演化 (Animated Evolution)</div>
        <div class="caption-text">
        这是一个<b>动态气泡图</b> (点击 Play 播放)。
        <ul>
            <li><b>X轴:</b> 当日新增 (爆发强度)。</li>
            <li><b>Y轴:</b> 累计确诊 (历史负担)。</li>
            <li><b>气泡:</b> 代表不同区域，随时间移动。</li>
        </ul>
        <span class="highlight">变化特征:</span> 气泡向右移动代表爆发，向上移动代表积累。观察哪些区域在短时间内快速向右上方冲刺。
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    # 辅助视图：Ridge Plot (山脊图) 变体，展示多区域波峰
    if not df_evo.empty:
        # 选出 Top 5 区域
        top_r = df_evo.groupby('rid')['count'].sum().nlargest(5).index
        df_top = df_evo[df_evo['rid'].isin(top_r)]

        fig_area = px.area(
            df_top, x="date", y="count", color="rid",
            line_group="rid",
            labels={'count': '新增病例', 'rid': 'Top区域'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_area.update_layout(template='plotly_dark', title="核心疫区波峰对比", height=400)
        st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("""
    <div class="caption-box">
        <div class="caption-title">🌊 波峰对比 (Peak Comparison)</div>
        <div class="caption-text">
        对比核心疫区的爆发时间差。
        <br>
        <span class="highlight">时序分析:</span> 
        不同区域的波峰是否错峰出现？这能揭示病毒在城市内部的<b>扩散路径</b>（例如：从中心区 Region A 扩散到郊区 Region B）。
        </div>
    </div>
    """, unsafe_allow_html=True)