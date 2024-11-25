from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Step 1: 데이터 로드
file_path = './groupby_address.csv'
data = pd.read_csv(file_path)

# 'Chain-Token-Contract' 컬럼을 토큰 목록으로 변환
data['Tokens'] = data['Chain-Token-Contract'].apply(
    lambda x: ' '.join(eval(x)) if isinstance(x, str) else ''
)

# Step 2: 벡터화 (CountVectorizer)
vectorizer = CountVectorizer()
token_matrix = vectorizer.fit_transform(data['Tokens'])

# 밀집 행렬(Dense Matrix)로 변환
token_matrix_dense = token_matrix.toarray()

# Step 3: 계층적 군집화 모델 생성
n_clusters = 10  # 원하는 클러스터 개수 설정
hierarchical_clustering = AgglomerativeClustering(
    n_clusters=n_clusters, metric='euclidean', linkage='ward'
)
clusters = hierarchical_clustering.fit_predict(token_matrix_dense)

# Step 4: 결과 저장
data['Cluster'] = clusters
output_file_path_clusters = './address_clusters_hierarchical.csv'
data[['Address', 'Cluster']].to_csv(output_file_path_clusters, index=False)

# Step 5: 덴드로그램 시각화
linkage_matrix = linkage(token_matrix_dense[:100], method='ward')  # 샘플링(최대 100개로 제한)
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=data['Address'][:100].values, leaf_rotation=90, leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Address")
plt.ylabel("Distance")
plt.show()

# 결과 파일 경로 제공
{
    "Clustered Data File": output_file_path_clusters
}
