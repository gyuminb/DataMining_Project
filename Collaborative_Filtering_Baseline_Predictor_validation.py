import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from AssociationRule_Recommendation import get_wallet_portfolio, preprocess_user_portfolio_data
import time

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

def create_pivot_data_from_portfolio(combined_real_portfolio_data):
    
    # Total Asset Value 계산 (Amount * Value)
    combined_real_portfolio_data['TotalAssetValue'] = (
        combined_real_portfolio_data['Amount'] * combined_real_portfolio_data['Value']
    )

    # 각 Address의 총 포트폴리오 가치 계산
    combined_real_portfolio_data['TotalPortfolioValue'] = combined_real_portfolio_data.groupby('Address')['TotalAssetValue'].transform('sum')

    # Investment Ratio 계산
    combined_real_portfolio_data['InvestmentRatio'] = (
        combined_real_portfolio_data['TotalAssetValue'] / combined_real_portfolio_data['TotalPortfolioValue']
    )

    

    # 피벗 테이블 생성
    pivot_data = combined_real_portfolio_data.pivot_table(
        index="Address",
        columns="Item",
        values="InvestmentRatio",
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



import pandas as pd

def remove_middle_and_min_items(portfolio_data, num_items_to_remove=2):
    
    # Ensure there are enough items to remove
    if len(portfolio_data) <= num_items_to_remove:
        return portfolio_data, []  # Not enough items to remove

    # Sort the data by 'Value' to identify min and median
    sorted_data = portfolio_data.sort_values(by="Value").reset_index(drop=True)

    # Find the minimum and median indices
    min_value_index = sorted_data.index[0]
    median_value_index = sorted_data.index[len(sorted_data) // 2]

    # Select rows between the minimum and median
    items_between_min_and_median = sorted_data.loc[min_value_index + 1: median_value_index - 1]

    # Randomly select items to remove from the range
    if len(items_between_min_and_median) < num_items_to_remove:
        num_items_to_remove = len(items_between_min_and_median)  # Adjust if range is too small

    removed_items = items_between_min_and_median.sample(n=num_items_to_remove, random_state=42)["Item"].tolist()

    # Remove the selected items
    updated_portfolio_data = portfolio_data[~portfolio_data["Item"].isin(removed_items)]

    return updated_portfolio_data, removed_items


def store_removed_user_differences(removed_items_dict,real_data, pivot_data, row_similarity_df, global_mean, row_bias, col_bias,address_mean_investment_ratio_shift):
    user_based_differences = {}
    item_based_differences = {}

    #User-based
    for address, removed_items in removed_items_dict.items():
        if address not in pivot_data.index:
            continue

        # 예측값 계산
        predicted_values = CF_baseline_predictor_userbased(
            address, pivot_data, address_mean_investment_ratio_shift,row_similarity_df, global_mean, row_bias, col_bias,top_k=5
        )

        # 차이 계산
        differences = calculate_difference_for_removed_items(address, removed_items, real_data, predicted_values)
        user_based_differences[address] = differences

    return user_based_differences
def store_removed_item_differences(removed_items_dict,real_data, pivot_data, row_similarity_df, global_mean, row_bias, col_bias,address_mean_investment_ratio_shift):
    item_based_differences = {} ##테이블 뒤집혓을 때 지갑주소 찾아야
    #Item-based
    for address, removed_items in removed_items_dict.items():

        ## 예측값 계산
        predicted_values = CF_baseline_predictor_itembased(
            address, pivot_data.T, address_mean_investment_ratio_shift,row_similarity_df, global_mean, row_bias, col_bias,top_k=5
        )
        # 차이 계산
        differences = calculate_difference_for_removed_items(address, removed_items, real_data, predicted_values)
        item_based_differences[address] = differences

    return item_based_differences

def calculate_difference_for_removed_items(address, removed_items, real_data, predicted_values):
    differences = {}
    for item in removed_items:
        original_ratio = real_data.loc[address, item]  # 원래 보유 비율
        predicted_value = predicted_values.get(item, 0)  # 예측값
        #print(address,item,"original_ratio : \n",original_ratio)
        #print(address,item,"predicted_ratio : \n",predicted_value)
        differences[item] = predicted_value - original_ratio
        
    return differences

# Main script
if __name__ == "__main__":
    file_path = './data/preprocessed_data.csv'
    
    # Portfolio 데이터 처리 및 유틸리티 매트릭스 생성
    address_data = pd.read_csv('./collaborative_validation_address.csv')
    
    portfolio_data_list = []
    removed_items_dict = {} 
    real_portfolio_data_list =[]
    
    for idx, addr in enumerate(address_data["Address"], start=1):
        print(f"\nFetching portfolio for address: {addr} (Progress: {idx}/{len(address_data)})")
        user_portfolio_data = get_wallet_portfolio(addr)
        if not user_portfolio_data:
            print(f"Failed to retrieve portfolio data for address {addr}. Skipping.")
            continue
        real_data = preprocess_user_portfolio_data(user_portfolio_data)
        updated_portfolio, removed_items = remove_middle_and_min_items(real_data, num_items_to_remove=2)
        
        # Save the updated portfolio and removed items
        portfolio_data_list.append(updated_portfolio)
        real_portfolio_data_list.append(real_data)
        removed_items_dict[addr] = removed_items
        time.sleep(1)  # 서버로부터 블락을 피하기 위해 딜레이 추가

    #print("\n\nremoved item list : ",removed_items_dict,"\n\n")

    # 3. 포트폴리오 데이터 병합
    combined_portfolio_data = pd.concat(portfolio_data_list, ignore_index=True)
    combined_real_portfolio_data = pd.concat(real_portfolio_data_list, ignore_index=True)
    
    
    real_data = create_pivot_data_from_portfolio(combined_real_portfolio_data)
    pivot_data, address_mean_investment_ratio_shift = preprocess_data(file_path, combined_portfolio_data)

    
    row_similarity_df = calculate_row_similarity(pivot_data)
    global_mean, row_bias, col_bias = calculate_baseline_predictor(pivot_data)
    removed_user_based_differences = store_removed_user_differences(removed_items_dict,real_data, pivot_data, row_similarity_df, global_mean, row_bias, col_bias,address_mean_investment_ratio_shift)
    
    row_similarity_df = calculate_row_similarity(pivot_data.T)
    global_mean, row_bias, col_bias = calculate_baseline_predictor(pivot_data.T)
    removed_item_based_differences = store_removed_item_differences(removed_items_dict,real_data, pivot_data.T, row_similarity_df,global_mean, row_bias, col_bias,address_mean_investment_ratio_shift)
     
    print(removed_item_based_differences) #empty,,?
    for wallet, differences in removed_item_based_differences.items():
        print(f"\nDifferences for wallet: {wallet}")
        for item, diff in differences.items():
            print(f" - {item}: {diff:.20f}")


    #RMSE
    diff_sum = 0
    for wallet, differences in removed_user_based_differences.items():
        for item, diff in differences.items():    
            diff_sum += diff**2
    result = (diff_sum / len(address_data))**(1/2)
    print("\nUser based Diff",result)

    diff_sum = 0
    for wallet, differences in removed_item_based_differences.items():
        for item, diff in differences.items():    
            diff_sum += diff**2
    result = (diff_sum / len(address_data))**(1/2)
    print("\nItem based Diff",result)





   
