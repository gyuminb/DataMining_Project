import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from datasketch import MinHash, MinHashLSH
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: 데이터 로드
file_path = './groupby_address.csv'  # 지갑별 토큰 데이터를 준비
data = pd.read_csv(file_path)

# Step 2: Token 컬럼 벡터화
vectorizer = CountVectorizer()
token_matrix = vectorizer.fit_transform(data['Chain-Token-Contract'].apply(lambda x: ' '.join(eval(x))))

# Step 3: MinHash LSH 준비
lsh = MinHashLSH(threshold=0.8, num_perm=128)  # 80% 이상 유사한 항목 탐지
minhashes = {}

for idx, row in enumerate(token_matrix):
    m = MinHash(num_perm=128)
    for token in row.indices:
        m.update(str(token).encode('utf8'))
    lsh.insert(f"Wallet_{idx}", m)
    minhashes[f"Wallet_{idx}"] = m

# Step 4: LSH를 통해 유사한 지갑 관계를 기반으로 벡터 생성
lsh_vectors = []

for i in range(token_matrix.shape[0]):
    wallet = f"Wallet_{i}"
    similar_wallets = lsh.query(minhashes[wallet])  # 유사한 지갑 찾기
    vector = [1 if f"Wallet_{j}" in similar_wallets else 0 for j in range(token_matrix.shape[0])]
    lsh_vectors.append(vector)

lsh_vectors = pd.DataFrame(lsh_vectors)

# Step 5: K-Means 클러스터링
n_clusters = 5  # 원하는 클러스터 개수 설정
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(lsh_vectors)

# Step 6: 결과 저장 및 시각화
data['Cluster'] = clusters
data.to_csv('./kmeans_lsh_clusters.csv', index=False)

print("K-Means 클러스터링 결과 저장 완료: kmeans_lsh_clusters.csv")

# Step 7: PCA를 사용한 시각화
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(lsh_vectors)

plt.figure(figsize=(10, 6))
for cluster_id in range(n_clusters):
    plt.scatter(
        reduced_data[data['Cluster'] == cluster_id, 0],
        reduced_data[data['Cluster'] == cluster_id, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.6,
    )
plt.title("K-Means Clustering with LSH Features")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()

