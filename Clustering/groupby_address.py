import pandas as pd

# Step 1: 데이터 로드
file_path = 'preprocessed_data.csv'
data = pd.read_csv(file_path)

# Step 2: Chain, Token, Contract Address를 결합한 문자열 생성
data['Chain-Token-Contract'] = data['Chain'] + '_' + data['Token'] + '_' + data['Contract Address']

# Step 3: Group by Address와 Token-Value 결합
# Chain-Token-Contract와 Value를 함께 묶어 리스트로 저장
data['Token-Value'] = data['Chain-Token-Contract'] + ':' + data['Value'].astype(str)
grouped_data = data.groupby('Address')['Token-Value'].apply(list).reset_index()

# Step 4: 결과 저장
output_path = './groupby_address_values.csv'
grouped_data.to_csv(output_path, index=False)
