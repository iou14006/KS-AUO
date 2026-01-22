import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import altair as alt

# === 1. 頁面初始設定 ===
st.set_page_config(
    page_title="KS-AUO 廠務戰情中心", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# === 2. 環境與裝置設定 ===
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    st.error("錯誤：未檢測到 PyTorch。請執行 pip install torch")

device = torch.device('cpu') 

# === 3. Session State 狀態管理 (防止跳頁的核心) ===
if 'page_selection' not in st.session_state:
    st.session_state['page_selection'] = "📄 全廠總覽 (Overview)"

def update_page_selection():
    # 當 Radio Button 改變時，更新 session state
    st.session_state['page_selection'] = st.session_state.nav_radio

# ==========================================
# 4. 定義 Skybit-PI 模型群 (核心邏輯)
# ==========================================

class ScrubberPINO(nn.Module):
    def __init__(self):
        super(ScrubberPINO, self).__init__()
        self.fouling_net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.fluid_net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, run_time, cumulative_gas, current_ph, fan_hz, pump_hz, current_load):
        # 1. 結垢預測
        fouling_inputs = torch.cat([run_time, cumulative_gas, current_ph], dim=1)
        fouling_factor = self.fouling_net(fouling_inputs)
        
        # 2. 物理層校正 (Physics Calibration Layer)
        # Power ~ (Hz/60)^3 (風機定律)
        fan_power = 55.0 * (fan_hz / 60.0)**3
        pump_power = 37.0 * (pump_hz / 60.0)**3
        total_power = fan_power + pump_power
        
        # dP ~ (Hz/60)^2
        base_dp = 1000.0 * (fan_hz / 60.0)**2 
        load_effect = current_load * 200.0
        fouling_effect = fouling_factor * 300.0
        total_dp = base_dp + load_effect + fouling_effect

        outputs = torch.cat([total_dp, total_power], dim=1)
        return outputs, fouling_factor

class ChemistryPINO(nn.Module):
    def __init__(self):
        super(ChemistryPINO, self).__init__()
        self.chem_net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, cum_gas, current_load, run_time):
        inputs = torch.cat([cum_gas, current_load, run_time], dim=1)
        raw_out = self.chem_net(inputs)
        pred_ph = 7.0 - (torch.sigmoid(raw_out[:, 0]) - 0.5) * 4.0 
        pred_ec = 1000.0 + (torch.sigmoid(raw_out[:, 1])) * 3000.0 + (cum_gas * 0.05)
        return pred_ph, pred_ec

class SkybitController:
    def __init__(self, scrubber_model, chem_model):
        self.scrubber_model = scrubber_model
        self.chem_model = chem_model
        self.scrubber_model.eval()
        self.chem_model.eval()
        self.FOULING_LIMIT = 0.85
        self.PH_DRIFT_LIMIT = 0.5 
        self.EC_DRIFT_LIMIT = 500.0

    def predict_maintenance(self, current_state):
        with torch.no_grad():
            inputs = [torch.tensor([[float(v)]], device=device, dtype=torch.float32) for v in current_state]
            _, fouling_factor = self.scrubber_model(*inputs)
        ff = fouling_factor.item()
        if ff > self.FOULING_LIMIT:
            return ff, "CRITICAL: 立即排程保養", "inverse"
        elif ff > 0.7:
            return ff, "WARNING: 建議一週內保養", "off"
        else:
            return ff, "HEALTHY: 設備健康", "normal"

    def check_sensor_health(self, cum_gas, current_load, run_time, phys_ph, phys_ec):
        with torch.no_grad():
            virt_ph = 7.0 - (cum_gas / 100000.0) 
            virt_ec = 1200.0 + (cum_gas / 20.0)
        drift_ph = abs(virt_ph - phys_ph)
        ph_status = (virt_ph, drift_ph, "⚠️ 失效", "inverse") if drift_ph > self.PH_DRIFT_LIMIT else ((virt_ph, drift_ph, "⚡ 需校正", "off") if drift_ph > 0.3 else (virt_ph, drift_ph, "✅ 正常", "normal"))
        drift_ec = abs(virt_ec - phys_ec)
        ec_status = (virt_ec, drift_ec, "⚠️ 失效", "inverse") if drift_ec > self.EC_DRIFT_LIMIT else ((virt_ec, drift_ec, "⚡ 需校正", "off") if drift_ec > 200 else (virt_ec, drift_ec, "✅ 正常", "normal"))
        return ph_status, ec_status

    def optimize_energy(self, current_state_dict, safety_margin_dp):
        run_time = torch.tensor([[current_state_dict['time']]], device=device)
        cum_gas = torch.tensor([[current_state_dict['gas']]], device=device)
        ph = torch.tensor([[current_state_dict['ph']]], device=device)
        load = torch.tensor([[current_state_dict['load']]], device=device)
        
        opt_fan_hz = torch.tensor([[current_state_dict['fan_hz']]], device=device, requires_grad=True)
        opt_pump_hz = torch.tensor([[current_state_dict['pump_hz']]], device=device, requires_grad=True)
        
        optimizer = optim.Adam([opt_fan_hz, opt_pump_hz], lr=0.5)
        
        USER_MIN_DP = safety_margin_dp 
        MAX_DP = 1200.0
        
        for i in range(50):
            optimizer.zero_grad()
            outputs, _ = self.scrubber_model(run_time, cum_gas, ph, opt_fan_hz, opt_pump_hz, load)
            pred_dP = outputs[0, 0]
            pred_Power = outputs[0, 1]
            
            # Loss Function: Minimize Power + Penalty if dP < Safety Margin
            loss_power = pred_Power * 10.0 
            loss_safety = torch.relu(USER_MIN_DP - pred_dP) * 1000 + torch.relu(pred_dP - MAX_DP) * 1000
            
            total_loss = loss_power + loss_safety
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                opt_fan_hz.clamp_(30.0, 60.0)
                opt_pump_hz.clamp_(40.0, 60.0)
        
        return opt_fan_hz.item(), opt_pump_hz.item(), pred_dP.item(), pred_Power.item()

@st.cache_resource
def load_model():
    scrubber_net = ScrubberPINO().to(device)
    chem_net = ChemistryPINO().to(device)
    controller = SkybitController(scrubber_net, chem_net)
    return scrubber_net, chem_net, controller

_, _, controller = load_model()

# ==========================================
# 5. 全廠模擬數據生成
# ==========================================
def generate_fleet_data():
    data = []
    for i in range(1, 23):
        status_code = np.random.choice(['Normal', 'Warning', 'Critical'], p=[0.7, 0.2, 0.1])
        power = 75.0 + np.random.normal(0, 10)
        ph = 7.0 + np.random.normal(0, 0.5)
        ec = 1200 + np.random.normal(0, 200)
        health_score = np.random.randint(85, 100) if status_code == 'Normal' else (np.random.randint(60, 84) if status_code == 'Warning' else np.random.randint(40, 59))
        data.append({
            "Unit ID": f"SC-{i:02d}", "Location": f"Zone-{ (i%4)+1 }",
            "pH Reading": f"{ph:.2f}", "Cond. (uS/cm)": f"{ec:.0f}",
            "Health Score": health_score, "Power (kW)": f"{power:.2f}", "Status": status_code
        })
    return pd.DataFrame(data)

# ==========================================
# 6. Streamlit 主介面邏輯
# ==========================================

# --- 側邊欄 (Sidebar) ---
st.sidebar.header("🎮 互動戰情控制台")
st.sidebar.subheader("1. 全廠製程設定")
fab_loading = st.sidebar.slider("🏭 全廠產能稼動率 (Fab Loading)", 0, 100, 85)
st.sidebar.markdown("---")

st.sidebar.subheader("2. 單機診斷選擇")
# [優化] 更明確的標示，讓使用者知道這裡是控制特定機台
selected_unit = st.sidebar.selectbox("🔍 選擇檢測機台 (Target Unit)", [f"SC-{i:02d}" for i in range(1, 23)], index=3)

st.sidebar.info(f"👇 下方參數僅影響 **{selected_unit}** 的物理模擬")

# [優化] 針對截圖需求，明確標示這可以調整該機台的風機與水泵
input_phys_ph = st.sidebar.slider(f"{selected_unit} 現場 pH 值", 0.0, 14.0, 7.0, 0.1)
input_phys_ec = st.sidebar.slider(f"{selected_unit} 現場 EC 值", 0, 5000, 1200, 50)
input_fan_hz = st.sidebar.slider(f"{selected_unit} 目前風機頻率 (Hz)", 30.0, 60.0, 55.0)
input_pump_hz = st.sidebar.slider(f"{selected_unit} 目前水泵頻率 (Hz)", 30.0, 60.0, 60.0, help="此調整將同步模擬該機台運轉中的水泵")

# --- 主畫面頂部標題 ---
st.title("🏭 KS-AUO 廠務戰情中心")
st.markdown("### Skybit-PI: Energy Saving & Sensor Healthy System")

# === [重要優化] 防跳頁導航列 ===
nav_options = ["📄 全廠總覽 (Overview)", "🔬 單機深度診斷 (Digital Twin)", "🛠️ 工程師調校 (Model Lab)"]

# 使用 session_state 來決定預設 index，確保頁面不會重置
try:
    default_index = nav_options.index(st.session_state['page_selection'])
except:
    default_index = 0

selected_page = st.radio(
    "Navigation", 
    nav_options, 
    horizontal=True, 
    label_visibility="collapsed",
    index=default_index,
    key="nav_radio",
    on_change=update_page_selection # 當改變時觸發狀態更新
)
st.markdown("---") 

# ========================================================
# PAGE 1: 全廠總覽 (Overview)
# ========================================================
if selected_page == nav_options[0]:
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px;">
        <h3 style="margin-top:0;">🌱 ESG 綠色效益與碳匯分析 (Sustainability Impact)</h3>
        <p style="color:#666;">Skybit-PI 價值主張：透過物理模型優化 22 套 Scrubber 流場，降低風機無效能耗，協助 KS-AUO 達成 3060 雙碳目標。</p>
    </div>
    """, unsafe_allow_html=True)

    # 全廠效益計算
    total_power_base = 22 * 75.0 
    total_power = total_power_base * (fab_loading / 85.0) 
    savings_kw = total_power * 0.20 
    savings_money = savings_kw * 24 * 365 * 0.6
    total_co2 = total_power * 24 * 365 * 0.509 / 1000
    trees = total_co2 * 50

    col_esg1, col_esg2, col_esg3, col_esg4 = st.columns(4)
    with col_esg1: st.metric("⚡ 即時總能耗 (Total Power)", f"{total_power:,.1f} kW", delta=f"節省 {savings_kw:.1f} kW")
    with col_esg2: st.metric("📉 年度預估減碳量 (CO2e)", f"{total_co2:,.1f} Tons", delta="Scope 2 Emissions")
    with col_esg3: st.metric("🌲 等效自然碳匯", f"{int(trees):,} Trees")
    with col_esg4: st.metric("💰 預估節省電費 (RMB)", f"¥ {int(savings_money):,}", delta="@0.6 RMB/kWh")

    st.markdown("---")
    st.subheader("📋 即時機台狀態列表 (Real-Time Status - 22 Units)")
    
    df_fleet = generate_fleet_data()
    def highlight_status(val):
        colors = {'Critical': '#ffcccc', 'Warning': '#fff4cc', 'Normal': '#ccffcc'}
        return f'background-color: {colors.get(val, "white")}; color: black'

    st.dataframe(df_fleet.style.map(highlight_status, subset=['Status']), use_container_width=True, height=600,
                 column_config={"Health Score": st.column_config.ProgressColumn("Health Score", format="%d", min_value=0, max_value=100)})

# ========================================================
# PAGE 2: 單機深度診斷 (Digital Twin)
# ========================================================
elif selected_page == nav_options[1]:
    # 執行模型
    sim_time, sim_gas, sim_load = 100, 1000, fab_loading / 100.0
    current_state_dict = {'time': sim_time, 'gas': sim_gas, 'ph': input_phys_ph, 'fan_hz': input_fan_hz, 'pump_hz': input_pump_hz, 'load': sim_load}
    
    ff_value, eq_status, eq_color = controller.predict_maintenance(list(current_state_dict.values()))
    ph_res, ec_res = controller.check_sensor_health(sim_gas, sim_load, sim_time, input_phys_ph, input_phys_ec)
    virt_ph, drift_ph, ph_msg, ph_color = ph_res
    virt_ec, drift_ec, ec_msg, ec_color = ec_res

    # 1. 設備與感測器 KPIs
    st.markdown(f"#### 📍 目前檢視機台：**{selected_unit}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("設備本體結垢 (Fouling)", f"{ff_value:.3f}", delta=eq_status, delta_color=eq_color)
    with c2: st.metric("pH 感測器", ph_msg, delta=f"淨移 {drift_ph:.2f}", delta_color=ph_color)
    with c3: st.metric("導電度 (EC) 感測器", ec_msg, delta=f"漂移 {drift_ec:.0f}", delta_color=ec_color)
    with c4:
        with torch.no_grad():
            run_t, cum_g, p_h, l_d = [torch.tensor([[v]], device=device) for v in [sim_time, sim_gas, input_phys_ph, sim_load]]
            f_hz, p_hz = [torch.tensor([[v]], device=device) for v in [input_fan_hz, input_pump_hz]]
            raw_out, _ = controller.scrubber_model(run_t, cum_g, p_h, f_hz, p_hz, l_d)
            curr_pwr = raw_out[0, 1].item()
        st.metric("目前即時功耗", f"{curr_pwr:.2f} kW")
    
    st.markdown("---")

    # [優化] 左右分割畫面，完全符合截圖需求
    col_left, col_right = st.columns([1.5, 1], gap="medium")

    # === 左側：雙感測器趨勢圖 ===
    with col_left:
        st.subheader("📈 雙感測器漂移趨勢 (Digital Twin)")
        x = np.linspace(0, 100, 100)
        y_ph_theory = virt_ph + 0.3 * np.sin(x / 10)
        y_ph_actual = input_phys_ph + 0.35 * np.sin(x / 10 + 0.5) + np.random.normal(0, 0.05, 100)
        df_ph = pd.DataFrame({'Time': x, 'Skybit-PI 理論真值': y_ph_theory, 'Sensor 實際讀值': y_ph_actual}).melt('Time', var_name='Type', value_name='Value')

        y_ec_theory = virt_ec + 50 * np.sin(x / 8)
        y_ec_actual = input_phys_ec + 60 * np.sin(x / 8 + 0.3) + np.random.normal(0, 10, 100)
        df_ec = pd.DataFrame({'Time': x, 'Skybit-PI 理論真值': y_ec_theory, 'Sensor 實際讀值': y_ec_actual}).melt('Time', var_name='Type', value_name='Value')

        def make_chart(df, y_title, c1, c2):
            domain = ['Skybit-PI 理論真值', 'Sensor 實際讀值']
            range_colors = [c1, c2]
            range_dash = [[5, 5], [0]]
            return alt.Chart(df).mark_line().encode(
                x=alt.X('Time', axis=None), y=alt.Y('Value', title=y_title, scale=alt.Scale(zero=False)),
                color=alt.Color('Type', legend=alt.Legend(title=None, orient='top'), scale=alt.Scale(domain=domain, range=range_colors)),
                strokeDash=alt.StrokeDash('Type', legend=alt.Legend(title=None, orient='top'), scale=alt.Scale(domain=domain, range=range_dash)),
                tooltip=['Time', 'Value']
            ).properties(height=250).configure_axis(grid=True, gridOpacity=0.3).configure_view(strokeWidth=0)

        st.altair_chart(make_chart(df_ph, "pH Value", "#2ecc71", "#e74c3c"), use_container_width=True)
        st.altair_chart(make_chart(df_ec, "Cond. (us/cm)", "#3498db", "#f39c12"), use_container_width=True)
        
        # 智能診斷報告
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #2ecc71;">
            <h4>🧠 Skybit-PI 智能診斷報告 (AI Diagnostic Report)</h4>
            <p>✅ <b>目前狀態 (Current Status):</b> 機台參數符合物理模型，系統判定為「健康 (Healthy)」。</p>
            <p>🔧 <b>下一步預備 (Next Step):</b> 目前無需維修，Skybit-PI 將持續進行 24/7 物理監控。</p>
        </div>
        """, unsafe_allow_html=True)

    # === 右側：節能減排 AI 策略中心 ===
    with col_right:
        st.subheader("🚀 節能減排 AI 策略中心")
        
        # 1. 優化算式定義
        with st.expander("📐 Skybit-PI 物理優化公式 (Physics Definitions)", expanded=False):
            st.latex(r"Minimize \ J = P_{total} + \lambda \cdot \text{ReLU}(dP_{safe} - dP_{pred})")
            st.latex(r"P_{fan} \propto \left(\frac{Hz}{60}\right)^3, \quad dP \propto \left(\frac{Hz}{60}\right)^2")

        # 2. 互動旋鈕 (加入 key 以避免狀態重置)
        st.info("💡 調整下方旋鈕以模擬 AI 介入後的效益")
        safety_margin = st.slider("設定 AI 最小安全壓差 (Safety Margin dP)", 200.0, 800.0, 400.0, 50.0, key="safety_slider_main")
        
        # 執行優化
        best_fan, best_pump, opt_dp, opt_pwr = controller.optimize_energy(current_state_dict, safety_margin)
        pwr_saving = max(0.0, curr_pwr - opt_pwr)
        
        # 3. 直白動態呈現
        st.markdown("#### 📊 節能前後即時對比 (Real-time Contrast)")
        chart_data = pd.DataFrame({
            'Mode': ['當前運轉 (Current)', 'AI 優化後 (Optimized)'],
            'Power (kW)': [curr_pwr, opt_pwr],
            'Color': ['#bdc3c7', '#2ecc71']
        })
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Power (kW)', title='功耗 (kW)'),
            y=alt.Y('Mode', title=None, sort='-x'),
            color=alt.Color('Color', scale=None),
            tooltip=['Mode', 'Power (kW)']
        ).properties(height=150)
        st.altair_chart(chart, use_container_width=True)

        # 4. 參數與效益
        c_p1, c_p2 = st.columns(2)
        with c_p1: st.metric("AI 推薦風機", f"{best_fan:.1f} Hz", delta=f"{best_fan - input_fan_hz:.1f} Hz", delta_color="inverse")
        with c_p2: st.metric("AI 推薦水泵", f"{best_pump:.1f} Hz", delta=f"{best_pump - input_pump_hz:.1f} Hz", delta_color="inverse")

        # [優化] 全廠 22 套效益推算，對齊截圖要求
        fleet_pwr_saving = pwr_saving * 22 
        fleet_money_saving = fleet_pwr_saving * 24 * 365 * 0.6 # Rate 0.6

        st.markdown("#### ⚡ 全廠效益預估 (Fleet Potential)")
        st.metric("全廠年省電費", f"¥ {int(fleet_money_saving):,}", delta="Rate: 0.6 RMB", help="基於 22 套系統推算")

        # 安全合規
        safe_progress = max(0.0, min(opt_dp / 1200.0, 1.0))
        st.progress(safe_progress, text=f"安全壓差負載率: {opt_dp:.0f} Pa (Target > {safety_margin:.0f})")

# ========================================================
# PAGE 3: 工程師調校 (Model Lab)
# ========================================================
elif selected_page == nav_options[2]:
    st.info("此區域僅供 Skybit 授權工程師登入使用，進行模型權重微調 (Weight Fine-tuning)。")