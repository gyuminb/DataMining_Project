import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(file_path, user_data):
    # 기존 데이터 로드
    data = pd.read_csv(file_path)

    # 'Amount' 열에서 쉼표 제거 및 숫자로 변환
    data['Amount'] = data['Amount'].replace({',': ''}, regex=True).astype(float)

    # 'Value' 열에서 특수 문자열 처리 및 숫자로 변환
    data['Value'] = data['Value'].replace({r'[\$,]': '', r'<0.000001': '0.000001'}, regex=True)
    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')  # 변환 실패 시 NaN으로 처리

    # NaN 값 제거 (Value 또는 Amount가 숫자로 변환되지 않은 행 제거)
    data = data.dropna(subset=['Value', 'Amount'])

    # 기존 데이터에서 사용자의 여러 Address와 동일한 행 제거
    user_addresses = user_data['Address'].unique()  # 사용자 지갑 주소 목록
    data = data[~data['Address'].isin(user_addresses)]  # 여러 주소와 일치하는 행 제거

    # 새로운 사용자의 데이터를 기존 데이터에 추가
    user_data = user_data.copy()  # 복사본 생성
    user_data['TotalAmount'] = user_data['Amount'].sum()
    user_data['TotalAssetValue'] = user_data['Amount'] * user_data['Value']  # 투자 가치 계산
    data = pd.concat([data, user_data], ignore_index=True)

    # 각 지갑 주소별 총 투자 가치 계산 (전체 데이터 기준)
    data['TotalAssetValue'] = data['Amount'] * data['Value']
    data['TotalPortfolioValue'] = data.groupby('Address')['TotalAssetValue'].transform('sum')

    # 각 지갑 주소와 코인의 투자 가치 비율 계산 (전체 데이터 기준)
    data['InvestmentRatio'] = data['TotalAssetValue'] / data['TotalPortfolioValue']

    


    # 각 지갑별 InvestmentRatio_shift 평균 계산
    address_mean_investment_ratio_shift = data.groupby('Address')['InvestmentRatio'].mean()

    # 피벗 테이블 생성
    pivot_data = data.pivot_table(
        index="Address",
        columns="Item",
        values="InvestmentRatio",
        aggfunc="sum",

    )
    pivot_data = pivot_data.fillna(0)
    # 각 지갑의 평균값을 피봇 테이블에서 빼기
    for address in pivot_data.index:
        pivot_data.loc[address] = pivot_data.loc[address].apply(
            lambda value: value - address_mean_investment_ratio_shift[address] if value != 0 else 0
        )

    #print(pivot_data)

    return pivot_data, address_mean_investment_ratio_shift


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

def CF_baseline_predictor_userbased(address, pivot_data,address_mean_investment_ratio_shift,row_similarity_df, global_mean, row_bias, col_bias, top_k=5):

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
        result += global_mean + row_bias[address] + col_bias[item] + address_mean_investment_ratio_shift[address] #shift 평
        predicted_values[item] = result
    
    return predicted_values

def CF_baseline_predictor_itembased(address, pivot_data,address_mean_investment_ratio_shift ,row_similarity_df, global_mean, row_bias, col_bias, top_k=5):
    
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
        result += global_mean + row_bias[item] + col_bias[address] + address_mean_investment_ratio_shift[address]
        predicted_values[item] = result
    
    return predicted_values