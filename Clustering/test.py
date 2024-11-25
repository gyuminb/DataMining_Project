import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: 데이터 로드
file_path = './preprocessed_data.csv'  # 파일 경로를 입력하세요
data = pd.read_csv(file_path)

# Step 2: 데이터 정리
# 문자열로 표시된 값에서 숫자만 추출
def extract_number(value):
    try:
        # "$" 제거, "," 제거, 나머지는 float로 변환
        return float(value.replace('$', '').replace(',', '').strip())
    except:
        return 0.0  # 변환 실패 시 0으로 처리

# 'Value', 'Portfolio %', 'Txn Count' 등의 컬럼 정리
data['Value'] = data['Value'].apply(extract_number)
data['Portfolio %'] = data['Portfolio %'].str.replace('<', '').str.replace('%', '').astype(float)
data['Txn Count'] = pd.to_numeric(data['Txn Count'], errors='coerce').fillna(0)

# Step 3: 주요 특징 추출
features = data.groupby('Address').agg({
    'Value': 'sum',  # 총 보유 자산 값
    'Token': 'nunique',  # 토큰의 다양성
    'Txn Count': 'sum',  # 총 거래 횟수
    'Portfolio %': 'mean',  # 평균 포트폴리오 비율
})

# 스테이블코인 비율 계산
stablecoins = ['USDT', 'USDC', 'DAI']
data['Stablecoin'] = data['Token'].apply(lambda x: x in stablecoins)
stablecoin_ratio = data.groupby('Address')['Stablecoin'].mean()
features['Stablecoin_Ratio'] = stablecoin_ratio

# Step 4: 데이터 스케일링
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 5: K-Means 군집화
n_clusters = 5  # 군집의 개수 설정
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# 클러스터 결과 저장
features['Cluster'] = clusters

# Step 6: PCA 차원 축소 및 시각화
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features_scaled)

plt.figure(figsize=(10, 8))
for cluster in features['Cluster'].unique():
    cluster_data = reduced_features[features['Cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster}")
plt.legend()
plt.title("Cluster Visualization")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# Step 7: 클러스터 분석
cluster_summary = features.groupby('Cluster').mean()
print(cluster_summary)


