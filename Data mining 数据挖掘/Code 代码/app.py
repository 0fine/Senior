import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import os

# ==========================================
# 1. 系统配置与样式
# ==========================================
st.set_page_config(layout="wide", page_title="城市群疫情时空可视分析系统")

# 注入 CSS：创建类似论文的 "Side Note" 备注栏
st.markdown("""
<style>
    .reportview-container .main .block-container {max-width: 1400px;}
    .caption-box {
        background-color: #262730;
        border-left: 5px solid #ff4b4b; /* 红色高亮条 */
        padding: 15px;
        border-radius: 5px;
        font-size: 14px;
        color: #e0e0e0;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .caption-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 8px;
        color: #ff9f4b; /* 橙色标题 */
        display: flex;
        align-items: center;
    }
    .caption-icon { margin-right: 8px; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = r"C:\Users\21626\Downloads\train_data\train_data"
CITY_LIST = ['city_A', 'city_B', 'city_C', 'city_D', 'city_E']


# ==========================================
# 2. 数据处理引擎 (增强版)
# ==========================================

def smart_date_parser(date_series):
    """鲁棒的日期解析，处理 20200101 或 2020-01-01"""
    s = date_series.astype(str).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')


@st.cache_data
def get_city_coordinates():
    city_coords = {}
    for city in CITY_LIST:
        path = os.path.join(BASE_DIR, city, "grid_attr.csv")
        try:
            if os.path.exists(path):
                # 读取少量数据估算中心点
                df = pd.read_csv(path, header=None, nrows=2000)
                # 假设前两列是经纬度
                lon = pd.to_numeric(df.iloc[:, 0], errors='coerce').mean()
                lat = pd.to_numeric(df.iloc[:, 1], errors='coerce').mean()

                # 坐标范围校验 (中国大致范围)
                if 70 < lon < 140 and 15 < lat < 55:
                    city_coords[city] = {"name": city, "lon": lon, "lat": lat}
                else:
                    raise ValueError("坐标异常")
            else:
                raise FileNotFoundError
        except:
            # Fallback 默认坐标
            defaults = {'city_A': (116.4, 39.9), 'city_B': (121.5, 31.2), 'city_C': (114.3, 30.6)}
            pos = defaults.get(city, (114.0, 30.0))
            city_coords[city] = {"name": city, "lon": pos[0], "lat": pos[1]}
    return city_coords


@st.cache_data
def load_macro_data(city_coords):
    all_infections = []
    all_migrations = []

    for city in CITY_LIST:
        city_path = os.path.join(BASE_DIR, city)

        # --- Infection ---
        inf_path = os.path.join(city_path, "infection.csv")
        if os.path.exists(inf_path):
            try:
                df = pd.read_csv(inf_path, header=None)
                # 自动判定列数
                if df.shape[1] == 4:
                    df.columns = ['cid', 'rid', 'date', 'val']
                elif df.shape[1] == 3:
                    df.columns = ['date', 'rid', 'val']
                else:
                    df = df.iloc[:, -3:]; df.columns = ['date', 'rid', 'val']

                df['val'] = pd.to_numeric(df['val'], errors='coerce').fillna(0)
                df['date'] = smart_date_parser(df['date'])

                valid = df.dropna(subset=['date'])
                if not valid.empty:
                    daily = valid.groupby('date')['val'].sum().reset_index()
                    daily['city_name'] = city
                    all_infections.append(daily)
            except:
                pass

        # --- Migration ---
        mig_path = os.path.join(city_path, "migration.csv")
        if os.path.exists(mig_path):
            try:
                df = pd.read_csv(mig_path, header=None)
                if df.shape[1] >= 4:
                    df = df.iloc[:, -4:]
                    df.columns = ['date', 'dep', 'arr', 'idx']
                    df['idx'] = pd.to_numeric(df['idx'], errors='coerce')
                    df['date'] = smart_date_parser(df['date'])
                    all_migrations.append(df.dropna(subset=['date']))
            except:
                pass

    df_inf = pd.concat(all_infections) if all_infections else pd.DataFrame()
    df_mig = pd.concat(all_migrations) if all_migrations else pd.DataFrame()

    # 映射坐标
    if not df_mig.empty:
        c_map = {k: (v['lon'], v['lat']) for k, v in city_coords.items()}

        def get_pos(n): return c_map.get(n, (0, 0))

        df_mig['f_pos'] = df_mig['dep'].map(get_pos)
        df_mig['t_pos'] = df_mig['arr'].map(get_pos)
        df_mig['flon'] = df_mig['f_pos'].apply(lambda x: x[0])
        df_mig['flat'] = df_mig['f_pos'].apply(lambda x: x[1])
        df_mig['tlon'] = df_mig['t_pos'].apply(lambda x: x[0])
        df_mig['tlat'] = df_mig['t_pos'].apply(lambda x: x[1])

    return df_inf, df_mig


@st.cache_data
def load_micro_data(city_name, selected_date_obj):
    """
    加载微观数据：网格密度、转移流向、区域属性、天气
    """
    city_path = os.path.join(BASE_DIR, city_name)
    data_pack = {
        'density': pd.DataFrame(),
        'transfer': pd.DataFrame(),  # 无日期，一般是小时模式
        'grid_attr': pd.DataFrame(),
        'weather': pd.DataFrame()
    }

    # 1. Density (大文件优化读取)
    try:
        p = os.path.join(city_path, "density.csv")
        if os.path.exists(p):
            # 增加读取行数以确保能覆盖到选定日期，实际部署建议用数据库
            df = pd.read_csv(p, header=None, nrows=200000)
            if df.shape[1] >= 5:
                df = df.iloc[:, -5:]
                df.columns = ['date', 'hour', 'lon', 'lat', 'idx']
                # 格式化日期字符串进行匹配 (YYYYMMDD)
                target_str = selected_date_obj.strftime('%Y%m%d')
                df['date_str'] = df['date'].astype(str).str.strip()

                # 过滤
                df_filtered = df[df['date_str'] == target_str].copy()
                if not df_filtered.empty:
                    df_filtered['idx'] = pd.to_numeric(df_filtered['idx'], errors='coerce')
                    df_filtered['hour'] = pd.to_numeric(df_filtered['hour'], errors='coerce')
                    df_filtered['lon'] = pd.to_numeric(df_filtered['lon'], errors='coerce')
                    df_filtered['lat'] = pd.to_numeric(df_filtered['lat'], errors='coerce')
                    data_pack['density'] = df_filtered
    except:
        pass

    # 2. Transfer (无日期列，假设通用模式)
    try:
        p = os.path.join(city_path, "transfer.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, header=None, nrows=50000)
            # hour, slon, slat, elon, elat, intensity
            if df.shape[1] >= 6:
                df = df.iloc[:, -6:]
                df.columns = ['hour', 'slon', 'slat', 'elon', 'elat', 'val']
                # 转换
                for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                data_pack['transfer'] = df.dropna()
    except:
        pass

    # 3. Grid Attr
    try:
        p = os.path.join(city_path, "grid_attr.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, header=None, nrows=50000)
            if df.shape[1] >= 3:
                df = df.iloc[:, -3:]
                df.columns = ['lon', 'lat', 'region_id']
                df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
                df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
                data_pack['grid_attr'] = df
    except:
        pass

    # 4. Weather
    try:
        p = os.path.join(city_path, "weather.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, header=None)
            if df.shape[1] >= 3:
                col_d = 0 if df.shape[1] == 8 else 1
                col_t = 2 if df.shape[1] == 8 else 3
                df['date'] = smart_date_parser(df.iloc[:, col_d])
                df['temp'] = pd.to_numeric(df.iloc[:, col_t], errors='coerce')
                data_pack['weather'] = df.groupby('date')[['temp']].mean().reset_index()
    except:
        pass

    return data_pack


# ==========================================
# 3. 页面主逻辑
# ==========================================

# 初始化
city_coords = get_city_coordinates()
df_macro_inf, df_macro_mig = load_macro_data(city_coords)

if df_macro_inf.empty:
    st.error("数据初始化失败。请检查 infection.csv。")
    st.stop()

# 侧边栏
st.sidebar.title("🎛️ 控制面板")
st.sidebar.markdown("---")
min_date = df_macro_inf['date'].min().date()
max_date = df_macro_inf['date'].max().date()

sel_date = st.sidebar.slider("📅 选择日期 (全域联动)", min_value=min_date, max_value=max_date, value=min_date)
sel_date_ts = pd.Timestamp(sel_date)
sel_city = st.sidebar.selectbox("🏙️ 选择目标城市 (微观分析)", CITY_LIST)

# 数据切片
daily_inf = df_macro_inf[df_macro_inf['date'] == sel_date_ts]
daily_mig = df_macro_mig[df_macro_mig['date'] == sel_date_ts]
micro_data = load_micro_data(sel_city, sel_date)

st.title("🦠 城市群疫情时空态势可视分析系统")
st.markdown("---")

# ==========================================
# View 1: 宏观态势
# ==========================================
st.subheader("1. 宏观视图：城市间传播网络与疫情分布")
c1, c2 = st.columns([3, 1])

with c1:
    nodes = []
    for city in CITY_LIST:
        coords = city_coords.get(city, {'lon': 0, 'lat': 0})
        v = daily_inf[daily_inf['city_name'] == city]['val'].sum()
        nodes.append({'name': city, 'lon': coords['lon'], 'lat': coords['lat'], 'val': int(v)})

    layers = []
    # 散点
    layers.append(pdk.Layer(
        "ScatterplotLayer", pd.DataFrame(nodes),
        get_position=["lon", "lat"], get_radius="val * 50 + 5000",
        get_fill_color=[255, 50, 50, 180], stroked=True,
        get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True
    ))
    # 迁徙线
    if not daily_mig.empty:
        layers.append(pdk.Layer(
            "ArcLayer", daily_mig,
            get_source_position=["flon", "flat"], get_target_position=["tlon", "tlat"],
            get_width="idx / 10",
            get_source_color=[0, 255, 255, 150], get_target_color=[255, 0, 255, 150]
        ))

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=35, longitude=110, zoom=3.5),
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "{name}\n新增确诊: {val}"}
    ))

with c2:
    st.markdown(f"""
    <div class="caption-box">
        <div class="caption-title">🗺️ 视图解读 (View Analysis)</div>
        本视图展示了<b>宏观尺度的疫情-人口交互网络</b>：
        <ul>
            <li><b>节点 (红圆):</b> 代表各城市。圆圈大小编码了 <b>{sel_date}</b> 当日的新增感染人数。</li>
            <li><b>连线 (弧线):</b> 代表城市间的人口迁徙流 (OD Flow)。青色端为起点，紫色端为终点，线条粗细代表迁徙规模指数。</li>
        </ul>
        <hr style="border-top: 1px solid #555;">
        <b>分析目标:</b> 识别输入性病例风险（高迁徙入度）与本地暴发（大红圆）的时空滞后关系。
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# View 2: 时序环境关联
# ==========================================
st.subheader(f"2. 时序分析：环境因素与传播关联 ({sel_city})")
c1, c2 = st.columns([3, 1])

with c1:
    trend = df_macro_inf[df_macro_inf['city_name'] == sel_city].sort_values('date')
    w = micro_data['weather']

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=trend['date'], y=trend['val'], name='新增病例', fill='tozeroy', line=dict(color='#ff4b4b')))
    if not w.empty:
        merged = pd.merge(trend, w, on='date')
        fig.add_trace(go.Scatter(x=merged['date'], y=merged['temp'], name='气温 (°C)', yaxis='y2',
                                 line=dict(color='#00d4ff', dash='dot')))

    fig.update_layout(
        template='plotly_dark', height=300, margin=dict(t=20, l=0, r=0, b=0),
        yaxis=dict(title="新增病例数"), yaxis2=dict(title="气温", overlaying='y', side='right'),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("""
    <div class="caption-box">
        <div class="caption-title">📈 关联分析 (Correlation)</div>
        该双轴图用于探索环境因子对R0值的影响：
        <ul>
            <li><b>红色区域:</b> 每日新增确诊趋势。</li>
            <li><b>蓝色虚线:</b> 每日平均气温。</li>
        </ul>
        <br>
        <b>研究假设:</b> 观察气温骤降是否伴随随后的感染率上升（负相关性），辅助判断病毒的环境耐受性。
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# View 3: 微观动态热力 (Density)
# ==========================================
st.markdown("---")
st.subheader(f"3. 微观分析 A：城市人流动态热力图")
c1, c2 = st.columns([3, 1])

# 获取当前城市数据
df_den = micro_data['density']
center = city_coords.get(sel_city, {'lon': 114, 'lat': 30})

with c1:
    if not df_den.empty:
        # 小时滑块
        hours = sorted(df_den['hour'].unique())
        if len(hours) > 0:
            cur_hour = st.slider("⏱️ 选择时段 (24H Dynamics)", min(hours), max(hours), hours[0])
            data_layer = df_den[df_den['hour'] == cur_hour]
        else:
            cur_hour = "N/A"
            data_layer = df_den

        # Heatmap Layer
        layer = pdk.Layer(
            "HeatmapLayer", data_layer,
            get_position=["lon", "lat"], get_weight="idx",
            opacity=0.8, radius_pixels=40, intensity=1.5, threshold=0.1
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=center['lat'], longitude=center['lon'], zoom=10, pitch=0),
            map_style="mapbox://styles/mapbox/dark-v10",
            tooltip={"text": "密度指数: {weight}"}
        ))
    else:
        st.warning(f"⚠️ {sel_city} 在 {sel_date} 暂无密度数据，可能是日期未覆盖或文件读取限制。建议更换日期尝试。")

with c2:
    peak_val = df_den['idx'].max() if not df_den.empty else 0
    st.markdown(f"""
    <div class="caption-box">
        <div class="caption-title">🔥 时变热力图 (Heatmap)</div>
        从静态统计转向<b>高时空分辨率的动态热场</b>：
        <ul>
            <li><b>视觉编码:</b> 颜色深浅代表网格人流密度 (Grid Density)。红色区域为高风险人群聚集点。</li>
            <li><b>交互功能:</b> 拖动上方滑块，观察城市人流的“潮汐效应”（如早晚高峰）。</li>
        </ul>
        <br>
        <b>关键指标:</b><br>
        当前最高密度指数: <span style="color:#ff4b4b;font-weight:bold">{peak_val}</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# View 4 & 5: 进阶分析 (Transfer & Attr)
# ==========================================
st.markdown("---")
st.subheader("4. 进阶分析：城市内部轨迹与功能区画像")
row2_c1, row2_c2 = st.columns(2)

# --- View 4: Intra-city Trajectory ---
with row2_c1:
    st.markdown("**4.1 城市毛细血管：内部出行轨迹 (Intra-city Mobility)**")
    df_trans = micro_data['transfer']

    if not df_trans.empty:
        # 取 Top 路径以防渲染过重
        df_top = df_trans.nlargest(1000, 'val')

        layer_lines = pdk.Layer(
            "LineLayer", df_top,
            get_source_position=["slon", "slat"],
            get_target_position=["elon", "elat"],
            get_color=[255, 255, 255, 50],  # 半透明白线
            get_width=1,
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer_lines],
            initial_view_state=pdk.ViewState(latitude=center['lat'], longitude=center['lon'], zoom=10, pitch=0),
            map_style="mapbox://styles/mapbox/dark-v10",
        ))

        st.markdown(f"""
        <div class="caption-box" style="font-size:12px; padding:10px;">
            <div class="caption-title" style="font-size:14px">🕸️ 轨迹网络 (Trajectory Net)</div>
            展示了城市内部的微观移动轨迹（取强度Top 1000）。
            <br><b>作用：</b> 识别连接不同高危区域的“传播走廊”。
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("该城市暂无内部轨迹(Transfer)数据。")

# --- View 5: Functional Region Analysis ---
with row2_c2:
    st.markdown("**4.2 风险画像：功能区人流统计 (Functional Region Risk)**")
    df_attr = micro_data['grid_attr']
    df_den_sub = micro_data['density']

    if not df_attr.empty and not df_den_sub.empty:
        # 将密度数据与区域属性合并（空间连接的简化版：按经纬度近似匹配）
        # 注意：实际中经纬度可能有浮点误差，这里演示逻辑。
        # 更严谨的做法是将 lat/lon round 到 4位小数后 merge
        df_attr['lon_r'] = df_attr['lon'].round(3)
        df_attr['lat_r'] = df_attr['lat'].round(3)
        df_den_sub['lon_r'] = df_den_sub['lon'].round(3)
        df_den_sub['lat_r'] = df_den_sub['lat'].round(3)

        merged = pd.merge(df_den_sub, df_attr, on=['lon_r', 'lat_r'])

        if not merged.empty:
            # 按区域ID统计总人流
            stats = merged.groupby('region_id')['idx'].mean().reset_index()
            stats['region_id'] = stats['region_id'].astype(str)  # 转字符方便绘图

            fig_bar = px.bar(
                stats, x='region_id', y='idx',
                color='idx',
                labels={'idx': '平均人流密度', 'region_id': '区域功能ID'},
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(template='plotly_dark', height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(f"""
            <div class="caption-box" style="font-size:12px; padding:10px;">
                <div class="caption-title" style="font-size:14px">📊 区域风险分布</div>
                统计不同功能区（Region ID）的平均人流密度。
                <br><b>作用：</b> 区分是居住区（假定ID=0）还是商业区（假定ID=1）风险更高。
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("网格属性无法与密度数据匹配（坐标精度不一致）。")
    else:
        st.info("缺少网格属性(grid_attr)数据。")