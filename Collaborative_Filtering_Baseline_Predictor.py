import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(file_path, user_data):
    # 기존 데이터 로드
    data = pd.read_csv(file_path)

    # 'Amount' 열에서 쉼표 제거 및 숫자로 변환
    data['Amount'] = data['Amount'].replace({',': ''}, regex=True).astype(float)

    # 기존 데이터에서 사용자의 Address와 동일한 행 제거
    user_address = user_data['Address'].iloc[0]  # 사용자는 하나의 주소만 가진다고 가정
    data = data[data['Address'] != user_address]

    # 새로운 사용자의 데이터를 기존 데이터에 추가
    user_data['TotalAmount'] = user_data['Amount'].sum()
    user_data['AmountRatio'] = user_data['Amount'] / user_data['TotalAmount']
    data = pd.concat([data, user_data], ignore_index=True)

    # 각 지갑 주소별 총 보유량 계산 (전체 데이터 기준)
    data['TotalAmount'] = data.groupby('Address')['Amount'].transform('sum')

    # 각 지갑 주소와 코인의 보유 비율 계산 (전체 데이터 기준)
    data['AmountRatio'] = data['Amount'] / data['TotalAmount']

    # 피벗 테이블 생성
    pivot_data = data.pivot_table(
        index="Address",
        columns="Item",
        values="AmountRatio",
        aggfunc="sum",
        fill_value=0
    )

    return pivot_data


# row 기반 코사인 유사도 계산
def calculate_row_similarity(pivot_data):
    row_cosine_sim = cosine_similarity(pivot_data)
    row_similarity_df = pd.DataFrame(row_cosine_sim, index=pivot_data.index, columns=pivot_data.index)
    return row_similarity_df

# Baseline Predictor 계산 함수
def calculate_baseline_predictor(pivot_data):
    """
    Calculate global average, user bias, and item bias for Baseline Predictor.
    """
    # Global average (μ)
    global_mean = pivot_data.values.mean()

    # row bias (b_u)
    row_bias = pivot_data.mean(axis=1) - global_mean

    # col bias (b_i)
    col_bias = pivot_data.mean(axis=0) - global_mean

    return global_mean, row_bias, col_bias

def CF_baseline_predictor_userbased(address, pivot_data, row_similarity_df, global_mean, row_bias, col_bias, top_k=5):

    # 사용자가 보유한 토큰 및 보유 비율 가져오기
    user_items = pivot_data.loc[address]

    K_similar_users = row_similarity_df[address].sort_values(ascending=False).drop(address).head(top_k)
    while K_similar_users.sum() == 0:
        top_k += 1
        K_similar_users = row_similarity_df[address].sort_values(ascending=False).drop(address).head(top_k)

    zero_items = user_items[user_items == 0].index
    predicted_values = {}
    
    for item in zero_items:
        result = 0
        for similar_user, similarity_score in K_similar_users.items():
            rating = pivot_data.loc[similar_user, item]
            bias = global_mean + row_bias[similar_user] + col_bias[item]
            result += (similarity_score / K_similar_users.sum()) * (rating - bias)
        result += global_mean + row_bias[address] + col_bias[item]
        predicted_values[item] = result
    
    return predicted_values

def CF_baseline_predictor_itembased(address, pivot_data, row_similarity_df, global_mean, row_bias, col_bias, top_k=5):
    
    # 사용자가 보유한 토큰 및 보유 비율 가져오기
    user_items = pivot_data.loc[address]
    predicted_values = {}
    
    zero_items = user_items[user_items == 0].index
    for item in zero_items:
        K_similar_items = row_similarity_df[item].sort_values(ascending=False).drop(item).head(top_k)
        result = 0
        if K_similar_items.sum() != 0:
            for similar_item, similarity_score in K_similar_items.items():
                rating = pivot_data.loc[address, similar_item]
                bias = global_mean + row_bias[similar_item] + col_bias[address]
                result += (similarity_score / K_similar_items.sum()) * (rating - bias)
        result += global_mean + row_bias[item] + col_bias[address]
        predicted_values[item] = result
    
    return predicted_values