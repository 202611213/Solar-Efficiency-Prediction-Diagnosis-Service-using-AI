import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 

import matplotlib.font_manager as fm  # 이 부분들로 깔끔하게 대체하셨습니다!
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 


st.set_page_config(
    page_title="태양광 AI 대시보드", 
    page_icon="☀️", 
    layout="wide"
)

st.title("☀️ 기상 통합 데이터를 활용한 태양광 발전 효율 예측 및 유지보수 시점 진단 AI 웹서비스")
st.markdown("본 대시보드는 실제 캐글(Kaggle) 태양광 데이터와 학습된 랜덤포레스트 모델을 활용하여 구동됩니다.")
st.divider()

# -------------------------------------------------------------------------
# [파일 불러오기]
# -------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("presun.pkl")
    data = pd.read_csv("Plant_1_Generation_Data.csv")
    return model, data

try:
    rf_model, df = load_assets()
    st.sidebar.success("✅ 파일 연결 성공! (presun.pkl, Plant_1_Generation_Data.csv)")
except Exception as e:
    st.sidebar.error("❌ 파일을 찾을 수 없습니다.")
    st.sidebar.info("💡 'sunny_app.py'와 같은 폴더 안에 [presun.pkl] 파일과 [Plant_1_Generation_Data.csv] 파일이 들어있는지 확인해주세요!")
    st.stop()


# -------------------------------------------------------------------------
# 2. 데이터 전처리 (시간 추출)
# -------------------------------------------------------------------------
df['발생일시'] = pd.to_datetime(df['발생일시'], format='%d-%m-%Y %H:%M', errors='coerce')
df['시간'] = df['발생일시'].dt.hour + df['발생일시'].dt.minute / 60


# -------------------------------------------------------------------------
# 3. 화면 구성 (100% 한국어 탭 구조)
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 데이터 분석 및 진단", "🔮 AI 발전량 예측"])

# --- [첫 번째 탭: 데이터 시각화] ---
with tab1:
    st.subheader("📋 데이터 미리보기 (Plant_1_Generation_Data.csv)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("📈 데이터 시각화 분석")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. 변수 간 상관관계 분석 (Heatmap)**")
        # 한글 깨짐을 완전히 방지하기 위해 히트맵 내부의 영문 컬럼명을 일시적으로 한글로 변환
        plot_df = df[['시간', '직류전력량', '교류전력량', '당일발전량']].copy()
        correlation_matrix = plot_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # 폰트 에러 방지를 위해 차트 안에는 숫자 중심 표현, 제목은 상단 텍스트로 대체
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        st.caption("※ 각 지표가 1에 가까울수록 서로 강한 연관성이 있음을 의미합니다.")
        
    with col2:
        st.markdown("**2. 시간대별 평균 태양광 발전량 변화**")
        df_hourgroup = df.groupby(df['발생일시'].dt.hour)['직류전력량'].mean()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df_hourgroup.plot(kind='line', marker='o', color='red', linewidth=2, ax=ax2)
        
        # 축 레이블을 영어로 두되 의미를 쉽게 파악하도록 세팅 (에러 완전 방지)
        ax2.set_xlabel('Time (Hour)')
        ax2.set_ylabel('Mean DC Power (kW)')
        ax2.set_xticks(range(0, 25))
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)
        st.caption("※ X축: 시간(0시~24시), Y축: 평균 발전량(kW)")

    st.divider()
    
    st.subheader("⚙️ 발전소 가동 시간 진단")
    optime = df[df['직류전력량'] > 0]['시간']
    if not optime.empty:
        c1, c2 = st.columns(2)
        c1.metric(label="발전소 가동 시작 시간", value=f"약 {optime.min():.1f}시")
        c2.metric(label="발전소 가동 종료 시간", value=f"약 {optime.max():.1f}시")
        st.info(f"💡 **종합 진단 의견:** 현재 태양광 발전소는 약 **{optime.min():.1f}시**에 가동을 시작하여 **{optime.max():.1f}시**에 종료되는 패턴을 보입니다. 이 시간대 외에 발전량이 급감하거나 비정상적인 수치가 감지되면 즉시 패널 세척 및 인버터 점검(유지보수)이 필요합니다.")

# --- [두 번째 탭: AI 예측] ---
with tab2:
    st.subheader("🔮 랜덤포레스트 모델 기반 실시간 발전량 예측")
    st.write("아래 슬라이더를 조절해 예측하고 싶은 시간대를 설정하면, 학습된 AI 모델이 발전량을 실시간으로 예측합니다.")
    
    # 시간 입력 슬라이더 (한국어 명시)
    input_hour = st.slider(
        "예측하고 싶은 시간대를 선택하세요 (0시 ~ 23시):",
        min_value=0.0,
        max_value=23.0,
        value=12.0,
        step=0.5,
        help="태양광 발전이 활발한 6시에서 18시 사이 설정을 권장합니다."
    )
    
    input_data = pd.DataFrame([[input_hour]], columns=['시간'])
    predicted_power = rf_model.predict(input_data)
    
    st.markdown("### 📢 AI 분석 결과")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric(label="선택한 시간대", value=f"{input_hour} 시")
    res_col2.metric(label="AI 예상 태양광 발전량", value=f"{predicted_power[0]:.2f} kW")
    
    # 한국어 유지보수 팁 가이드라인 제공
    if 6 <= input_hour <= 18:
        st.success(f"✅ {input_hour}시는 안정적인 태양광 발전 효율이 기대되는 구간입니다.")
    else:
        st.warning(f"💤 {input_hour}시는 일조량이 없거나 부족하여 발전량이 매우 낮거나 없을 것으로 예상됩니다.")
