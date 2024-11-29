import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import Binarizer

# CSV 데이터 불러오기
file_path = './preprocessed_data.csv'
data = pd.read_csv(file_path)

# 'Amount' 열에서 쉼표 제거 및 숫자로 변환
data['Amount'] = data['Amount'].replace({',': ''}, regex=True).astype(float)

# 피벗 테이블 생성: 지갑 주소를 행(Index), 코인을 열(Columns), 보유량(Amount)을 값으로
pivot_data = data.pivot_table(
    index="Address",
    columns="Token",
    values="Amount",
    aggfunc="sum",
    fill_value=0
)

# 이진화 (0/1로 변환, 보유 여부만 표시)
binarizer = Binarizer(threshold=0)
binary_pivot_data = pd.DataFrame(
    binarizer.fit_transform(pivot_data),
    index=pivot_data.index,
    columns=pivot_data.columns
)

# 자카드 유사도 매트릭스 생성 (벡터화)
from scipy.spatial.distance import pdist, squareform

def compute_jaccard_similarity_optimized(binary_data):
    """
    Calculate pairwise Jaccard similarity using vectorized operations.
    """
    jaccard_distances = pdist(binary_data, metric="jaccard")
    jaccard_similarity_matrix = 1 - squareform(jaccard_distances)

    # DataFrame으로 변환
    addresses = binary_data.index
    similarity_df = pd.DataFrame(jaccard_similarity_matrix, index=addresses, columns=addresses)
    return similarity_df


similarity_df = compute_jaccard_similarity_optimized(binary_pivot_data)

# 특정 지갑 주소에 기반한 추천 함수 정의
def recommend_wallet_user_based(address, data, similarity_df, top_n=5, k=3):
    """
    Recommend tokens based on User-User Collaborative Filtering.
    """
    if address not in similarity_df.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")

    # 1. k명의 가장 유사한 사용자 선택
    similar_wallets = similarity_df[address].sort_values(ascending=False).drop(address).head(k)

    # 2. 각 아이템에 대한 점수 예측
    weighted_scores = {}
    for token in data.columns:
        # 유사 사용자들의 평가 점수와 유사도를 가중 평균으로 계산
        scores = [
            (data.loc[user, token] * similarity)
            for user, similarity in similar_wallets.items()
        ]
        weighted_scores[token] = sum(scores) / similar_wallets.sum()

    # 3. 해당 사용자가 이미 보유한 아이템 제외
    existing_tokens = data.loc[address][data.loc[address] > 0].index
    recommendations = pd.Series(weighted_scores).drop(existing_tokens).sort_values(ascending=False).head(top_n)

    return recommendations


# 테스트: 특정 지갑 주소에 대해 추천 코인 실행
address_to_recommend = "0x062a31bd836cecb1b6bc82bb107c8940a0e6a01d"  # 테스트용 지갑 주소
recommendations = recommend_wallet_user_based(address_to_recommend, pivot_data, similarity_df)

# 결과 출력
print(f"추천 코인 리스트 (지갑 주소: {address_to_recommend}):")
print(recommendations)
