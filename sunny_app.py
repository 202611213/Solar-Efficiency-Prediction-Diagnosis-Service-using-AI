import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # pkl 파일을 불러오기 위한 라이브러리

# 1. 페이지 기본 설정 (웹 브라우저 탭에 표시될 이름과 아이콘)
st.set_page_config(
    page_title="태양광 발전 효율 예측 및 진단", 
    page_icon="☀️", 
    layout="wide"
)

st.title("☀️ 기상 통합 데이터를 활용한 태양광 발전 효율 예측 및 유지보수 시점 진단")
st.markdown("실제 캐글(Kaggle) 태양광 데이터와 학습된 랜덤포레스트 모델을 활용한 대시보드입니다.")
st.divider()

# -------------------------------------------------------------------------
# [중요] 내 컴퓨터에 있는 CSV(원본 이름)와 PKL 파일을 불러오는 부분
# -------------------------------------------------------------------------
@st.cache_resource  # 웹페이지가 새로고침되어도 파일 로딩을 빠르게 유지해줍니다.
def load_assets():
    # 1. 모델 불러오기 (구글 드라이브에서 다운받아 폴더에 넣은 파일)
    model = joblib.load("presun.pkl")
    
    # 2. 데이터 불러오기 (코랩에서 다운받아 폴더에 넣은 파일 - 원본 이름 그대로)
    data = pd.read_csv("Plant_1_Generation_Data.csv")
    
    return model, data

# 파일이 폴더 안에 제대로 있는지 검사하고 불러옵니다.
try:
    rf_model, df = load_assets()
    st.sidebar.success("✅ 파일 연동 완료! (presun.pkl, Plant_1_Generation_Data.csv)")
except Exception as e:
    st.sidebar.error("❌ 파일을 찾을 수 없습니다.")
    st.sidebar.info("💡 'app.py'와 같은 폴더 안에 [presun.pkl] 파일과 [Plant_1_Generation_Data.csv] 파일이 들어있는지 확인해주세요!")
    st.stop() # 파일이 없으면 아래 코드가 실행되지 않고 프로그램이 멈춥니다.


# -------------------------------------------------------------------------
# 2. 코랩 노트북과 똑같은 데이터 전처리 진행
# -------------------------------------------------------------------------
# '발생일시'를 날짜형으로 바꾸고 '시간' 정보 숫자로 추출 (예: 13시 30분 -> 13.5)
df['발생일시'] = pd.to_datetime(df['발생일시'], format='%d-%m-%Y %H:%M', errors='coerce')
df['시간'] = df['발생일시'].dt.hour + df['발생일시'].dt.minute / 60


# -------------------------------------------------------------------------
# 3. 화면 구성 (탭 구조)
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
        correlation_matrix = df[['시간', '직류전력량', '교류전력량', '당일발전량']].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title('태양광 데이터 변수 간 상관관계 분석 (Heatmap)')
        st.pyplot(fig)
        
    with col2:
        st.markdown("**2. 시간대별 평균 태양광 발전량 변화**")
        df_hourgroup = df.groupby(df['발생일시'].dt.hour)['직류전력량'].mean()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df_hourgroup.plot(kind='line', marker='o', color='red', linewidth=2, ax=ax2)
        ax2.set_title('시간대별 평균 태양광 발전량 변화')
        ax2.set_xlabel('시간 (Hour)')
        ax2.set_ylabel('평균 직류전력량 (kW)')
        ax2.set_xticks(range(0, 25))
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)

    st.divider()
    
    st.subheader("⚙️ 발전소 가동 시간 분석")
    optime = df[df['직류전력량'] > 0]['시간']
    if not optime.empty:
        c1, c2 = st.columns(2)
        c1.metric(label="발전소 가동 시작 시간", value=f"약 {optime.min():.1f}시")
        c2.metric(label="발전소 가동 종료 시간", value=f"약 {optime.max():.1f}시")
        st.info(f"💡 **진단:** 현재 발전소는 약 {optime.min():.1f}시에 작동을 시작해 {optime.max():.1f}시에 종료됩니다.")

# --- [두 번째 탭: AI 예측] ---
with tab2:
    st.subheader("🔮 랜덤포레스트 모델 기반 발전량 예측")
    st.write("시간대를 조절하면 AI 모델(`presun.pkl`)이 실시간으로 발전량을 예측합니다.")
    
    # 시간 입력 슬라이더
    input_hour = st.slider(
        "예측하고 싶은 시간대를 선택하세요 (0~23 사이):",
        min_value=0.0,
        max_value=23.0,
        value=12.0,
        step=0.5,
        help="6시 ~ 18시 사이를 권장합니다."
    )
    
    # 모델에 넣을 데이터프레임 형태 생성
    input_data = pd.DataFrame([[input_hour]], columns=['시간'])
    
    # 실제 불러온 pkl 모델로 예측 수행!
    predicted_power = rf_model.predict(input_data)
    
    # 결과 출력
    st.markdown("### 📢 AI 예측 결과")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric(label="입력 시간", value=f"{input_hour} 시")
    res_col2.metric(label="AI 예상 태양광 발전량", value=f"{predicted_power[0]:.2f} kW")
