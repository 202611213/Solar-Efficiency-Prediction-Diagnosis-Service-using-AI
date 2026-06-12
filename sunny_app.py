st.set_page_config(page_title="태양광 AI 대시보드", page_icon="☀️", layout="wide")
st.title("☀️ 기상 통합 데이터를 활용한 태양광 발전 효율 예측 및 유지보수 시점 진단 AI 웹서비스")
st.markdown("본 대시보드는 실제 캐글(Kaggle) 태양광 데이터와 학습된 랜덤포레스트 모델을 활용하여 구동됩니다.")
st.divider()

# -------------------------------------------------------------------------
# [파일 불러오기 및 열 이름 무조건 강제 매핑]
# -------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("presun.pkl")
    data = pd.read_csv("Plant_1_Generation_Data.csv")
    
    # [핵심] 컬럼명이 영어든 한글든 상관없이 위치(순서) 기준이나 이름 기준으로 강제 통일합니다.
    data.columns = data.columns.str.strip()
    
    # 캐글 원본 순서: 보통 첫 번째가 DATE_TIME, 순서대로 DC, AC, YIELD가 옵니다.
    # 안전하게 기존에 존재할 만한 모든 이름을 한글로 바꿉니다.
    rename_dict = {
        'DATE_TIME': '발생일시', 'date_time': '발생일시', 'Date_Time': '발생일시',
        'DC_POWER': '직류전력량', 'dc_power': '직류전력량',
        'AC_POWER': '교류전력량', 'ac_power': '교류전력량',
        'DAILY_YIELD': '당일발전량', 'daily_yield': '당일발전량'
    }
    data = data.rename(columns=rename_dict)
    
    # 혹시라도 rename이 실패했을 경우를 대비한 최후의 보루 (첫 4개 컬럼 강제 이름 지정)
    if '발생일시' not in data.columns and len(data.columns) >= 5:
        data.columns.values[0] = '발생일시'
        data.columns.values[1] = '식별코드'
        data.columns.values[2] = '직류전력량'
        data.columns.values[3] = '교류전력량'
        data.columns.values[4] = '당일발전량'

    return model, data

try:
    rf_model, df = load_assets()
    st.sidebar.success("✅ 파일 연결 성공!")
except Exception as e:
    st.sidebar.error("❌ 파일을 찾을 수 없습니다.")
    st.stop()

# -------------------------------------------------------------------------
# 2. 데이터 전처리 (KeyError 절대 방지)
# -------------------------------------------------------------------------
# '발생일시' 컬럼을 안전하게 변환
df['변환일시'] = pd.to_datetime(df['발생일시'], errors='coerce')
df['시간'] = df['변환일시'].dt.hour + df['변환일시'].dt.minute / 60

# -------------------------------------------------------------------------
# 3. 화면 구성 (100% 한국어 탭 구조)
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 데이터 분석 및 진단", "🔮 AI 발전량 예측"])

with tab1:
    st.subheader("📋 데이터 미리보기 (Plant_1_Generation_Data.csv)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("📈 데이터 시각화 분석")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. 변수 간 상관관계 분석 (Heatmap)**")
        # 안전한 컬럼들만 추려서 상관관계 계산
        valid_cols = [c for c in ['시간', '직류전력량', '교류전력량', '당일발전량'] if c in df.columns]
        plot_df = df[valid_cols].copy()
        correlation_matrix = plot_df.corr()
        
        # 🔥 대안 적용: Matplotlib 없이 한글이 100% 출력되는 데이터프레임 스타일러 히트맵 구현
        styled_corr = correlation_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}")
        st.dataframe(styled_corr, use_container_width=True)
        st.caption("※ 각 지표가 1에 가까울수록 서로 강한 연관성이 있음을 의미합니다.")
        
    with col2:
        st.markdown("**2. 시간대별 평균 태양광 발전량 변화**")
        target_dc = '직류전력량' if '직류전력량' in df.columns else df.columns[2]
        df_hourgroup = df.groupby(df['변환일시'].dt.hour)[target_dc].mean().reset_index()
        # 컬럼 이름 가독성 좋게 변경 (차트 표시용)
        df_hourgroup.columns = ['시간(시)', '평균 직류전력량(kW)']
        
        # 🔥 대안 적용: 한글 폰트 설정이 전혀 필요 없는 Streamlit 내장 라인 차트 사용
        st.line_chart(df_hourgroup, x='시간(시)', y='평균 직류전력량(kW)', use_container_width=True)
        st.caption("※ X축: 시간(0시~24시), Y축: 평균 발전량(kW)")

    st.divider()
    
    st.subheader("⚙️ 발전소 가동 시간 진단")
    target_dc = '직류전력량' if '직류전력량' in df.columns else df.columns[2]
    optime = df[df[target_dc] > 0]['시간']
    if not optime.empty:
        c1, c2 = st.columns(2)
        c1.metric(label="발전소 가동 시작 시간", value=f"약 {optime.min():.1f}시")
        c2.metric(label="발전소 가동 종료 시간", value=f"약 {optime.max():.1f}시")
        st.info(f"💡 **종합 진단 의견:** 현재 태양광 발전소는 약 **{optime.min():.1f}시**에 가동을 시작하여 **{optime.max():.1f}시**에 종료되는 패턴을 보입니다. 이 시간대 외에 발전량이 급감하거나 비정상적인 수치가 감지되면 즉시 패널 세척 및 인버터 점검(유지보수)이 필요합니다.")

with tab2:
    st.subheader("🔮 랜덤포레스트 모델 기반 실시간 발전량 예측")
    st.write("아래 슬라이더를 조절해 예측하고 싶은 시간대를 설정하면, 학습된 AI 모델이 발전량을 실시간으로 예측합니다.")
    
    input_hour = st.slider(
        "예측하고 싶은 시간대를 선택하세요 (0시 ~ 23시):",
        min_value=0.0, max_value=23.0, value=12.0, step=0.5
    )
    
    input_data = pd.DataFrame([[input_hour]], columns=['시간'])
    predicted_power = rf_model.predict(input_data)
    
    st.markdown("### 📢 AI 분석 결과")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric(label="선택한 시간대", value=f"{input_hour} 시")
    res_col2.metric(label="AI 예상 태양광 발전량", value=f"{predicted_power[0]:.2f} kW")
    
    if 6 <= input_hour <= 18:
        st.success(f"✅ {input_hour}시는 안정적인 태양광 발전 효율이 기대되는 구간입니다.")
    else:
        st.warning(f"💤 {input_hour}시는 일조량이 없거나 부족하여 발전량이 매우 낮거나 없을 것으로 예상됩니다.")
