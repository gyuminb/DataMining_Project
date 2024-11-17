import pandas as pd


# Load the data file
file_path = 'preprocessed_data.csv'
data = pd.read_csv(file_path)

# Combine Chain, Token, and Contract Address into a single string
data['Chain-Token-Contract'] = data['Chain'] + '_' + data['Token'] + '_' + data['Contract Address']

# Group by Address and aggregate the combined information
grouped_data = data.groupby('Address')['Chain-Token-Contract'].apply(list).reset_index()

# 결과를 CSV 파일로 저장
output_path = './groupby_address.csv'
grouped_data.to_csv(output_path, index=False)

