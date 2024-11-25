import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from datasketch import MinHash, MinHashLSH

# Step 1: 데이터 로드
file_path = './groupby_address.csv'  # 지갑별 토큰 데이터를 준비
data = pd.read_csv(file_path)

# Step 2: Token 컬럼 벡터화
vectorizer = CountVectorizer()
token_matrix = vectorizer.fit_transform(data['Chain-Token-Contract'].apply(lambda x: ' '.join(eval(x))))

# Step 3: MinHash LSH 준비
lsh = MinHashLSH(threshold=0.96, num_perm=128)  # 80% 이상 유사한 항목 탐지
minhashes = {}

for idx, row in enumerate(token_matrix):
    m = MinHash(num_perm=128)
    for token in row.indices:
        m.update(str(token).encode('utf8'))
    lsh.insert(f"Wallet_{idx}", m)
    minhashes[f"Wallet_{idx}"] = m

# Step 4: 유사 지갑 검색 (예: Wallet_0과 유사한 지갑 찾기)
query = minhashes["Wallet_10"]
similar_wallets = lsh.query(query)

print(f"Wallet_0과 유사한 지갑들: {similar_wallets}")
