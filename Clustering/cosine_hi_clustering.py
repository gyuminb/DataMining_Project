from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
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

# Step 3: 코사인 유사도 계산
cosine_sim_matrix = cosine_similarity(token_matrix)

# Step 4: 코사인 유사도를 거리 행렬로 변환
cosine_distance_matrix = 1 - cosine_sim_matrix

# Step 5: 계층적 군집화 수행
n_clusters = 10  # 원하는 클러스터 개수 설정
hierarchical_clustering = AgglomerativeClustering(
    n_clusters=n_clusters, metric='precomputed', linkage='average'
)
clusters = hierarchical_clustering.fit_predict(cosine_distance_matrix)

# Step 6: 결과 저장
data['Cluster'] = clusters
output_file_path_clusters = './address_clusters_cosine_hierarchical.csv'
data[['Address', 'Cluster']].to_csv(output_file_path_clusters, index=False)

# Step 7: 덴드로그램 시각화 (데이터 크기 제한)
sample_size = 50  # 덴드로그램에 표시할 샘플 크기 제한
sample_indices = np.random.choice(len(data), size=sample_size, replace=False)

# 샘플링된 데이터로 거리 행렬 생성
sample_distance_matrix = cosine_distance_matrix[np.ix_(sample_indices, sample_indices)]
sample_labels = data['Address'].values[sample_indices]

# 덴드로그램 생성
linkage_matrix = linkage(sample_distance_matrix, method='average')
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=sample_labels, leaf_rotation=90, leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram (Sampled Data)")
plt.xlabel("Address")
plt.ylabel("Distance")
plt.show()

# 결과 파일 경로 제공
{
    "Clustered Data File": output_file_path_clusters
}
