import pandas as pd

# 파일 경로 설정
file_path = './data/preprocessed_data.csv'

# 데이터 로드
data = pd.read_csv(file_path)

# 7개 이상의 코인을 가진 지갑 주소 필터링
addresses_with_7_or_more_tokens = (
    data.groupby("Address")["Token"]
    .nunique()  # 주소별 고유 코인 종류 개수 계산
    .loc[lambda x: x >= 6]  # 7개 이상인 주소 필터링
    .index.tolist()  # 주소 리스트로 변환
)

# 중복 제거된 주소를 DataFrame으로 변환
unique_addresses = pd.DataFrame(addresses_with_7_or_more_tokens, columns=["Address"])

# 결과를 CSV 파일로 저장
output_file_path = 'collaborative_validation_address.csv'
unique_addresses.to_csv(output_file_path, index=False)

print(f"Unique addresses saved to {output_file_path}")