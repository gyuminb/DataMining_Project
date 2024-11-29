import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def get_wallet_portfolio(address):
    portfolio_data = []
    portfolio_url = f"https://etherscan.io/address/{address}#multichain-portfolio"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36'
    }
    response = requests.get(portfolio_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 테이블 찾기
    portfolio_table = soup.find('table', {'id': 'js-chain-table'})
    if not portfolio_table:
        print(f"No portfolio table found for address {address}")
        return portfolio_data  # 테이블이 없다면 빈 리스트 반환

    rows = portfolio_table.find('tbody').find_all('tr')
    if not rows:
        print(f"No data found in portfolio table for address {address}")
        return portfolio_data  # 데이터가 없다면 빈 리스트 반환

    # 각 행에서 데이터 추출
    for row in rows:
        cols = row.find_all('td')
        chain = cols[0].text.strip()
        token = cols[1].text.strip()
        portfolio_percent = cols[2].text.strip()
        price = cols[3].text.strip()
        amount = cols[4].text.strip()
        value = cols[5].text.strip()
        # 컨트랙트 주소가 있는 경우 추출
        contract_link = cols[1].find('a')
        contract_address = contract_link.get('href').split('/')[-1].split('?')[0] if contract_link else None
        portfolio_data.append({
            "Address": address,
            "Chain": chain,
            "Token": token,
            "Contract Address": contract_address,
            "Portfolio %": portfolio_percent,
            "Price": price,
            "Amount": amount,
            "Value": value
        })

    return portfolio_data

def preprocess_user_portfolio_data(portfolio_data):
    
    # 데이터프레임 변환
    df = pd.DataFrame(portfolio_data)
    
    # Contract Address와 Address가 동일한 경우 Contract Address를 null로 변경
    df['Contract Address'] = df.apply(
        lambda row: None if row['Address'] == row['Contract Address'] else row['Contract Address'], axis=1
    )
    
    # (Chain, Token, Contract Address) 아이템 생성
    df['Item'] = df.apply(
        lambda row: f"{row['Chain']}_{row['Token']}" if pd.isna(row['Contract Address']) else f"{row['Chain']}_{row['Token']}_{row['Contract Address']}",
        axis=1
    )
    
    # 'ETH_Ether (ETH)' 제거 (Chain이 ETH, Token이 Ether (ETH), Contract Address가 NULL인 경우)
    df = df[~((df['Chain'] == 'ETH') & (df['Token'] == 'Ether (ETH)') & (df['Contract Address'].isna()))]

    # 쉼표 제거 후 Amount 열을 숫자로 변환
    df['Amount'] = df['Amount'].str.replace(',', '')  # 쉼표 제거
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')  # 숫자로 변환
    
    # NaN 또는 0인 Amount 제거
    df = df[df['Amount'] > 0]
    
    # Portfolio % 다시 계산
    total_amount = df['Amount'].sum()
    print(f"총 Amount: {total_amount}")
    
    if total_amount > 0:
        df['Portfolio %'] = (df['Amount'] / total_amount) * 100
    else:
        df['Portfolio %'] = 0  # 전체 Amount가 0인 경우
    
    return df
    
# Association Rule을 기반으로 추천 아이템 생성
def recommend_items_based_on_association_rule(user_portfolio, association_rules_csv):

    # Association Rule 로드
    association_rules = pd.read_csv(association_rules_csv)
    
    # 사용자의 포트폴리오를 set으로 변환
    user_items = set(user_portfolio)

    # 1차: antecedent가 사용자 아이템에 완전히 속하는 Rule 필터링
    applicable_rules = association_rules[
        association_rules['antecedent'].apply(lambda x: set(eval(x)).issubset(user_items))
    ]

    # 2차: consequent가 사용자 아이템에 없는 경우만 유지
    filtered_rules = []
    recommended_items = set()

    for _, rule in applicable_rules.iterrows():
        consequent = set(eval(rule['consequent']))
        new_consequents = consequent - user_items  # 사용자에게 없는 consequent만 선택

        if new_consequents:  # 새로운 추천 아이템이 있는 경우만 필터링
            filtered_rules.append(rule)
            recommended_items.update(new_consequents)

    # applicable_rules를 필터링된 결과로 업데이트
    applicable_rules = pd.DataFrame(filtered_rules)

    return recommended_items, applicable_rules



if __name__ == "__main__":
    # 사용자로부터 지갑 주소 입력받기
    user_address = input("Enter the user's wallet address: ").strip()
    association_rules_csv = input("Enter the path to the association rules CSV file: ").strip()

    # 스크래핑으로 포트폴리오 데이터 가져오기
    print(f"\nFetching portfolio for address: {user_address}")
    user_portfolio_data = get_wallet_portfolio(user_address)
    if not user_portfolio_data:
        print(f"Failed to retrieve portfolio data for address {user_address}. Exiting.")
        exit()
    
    # 포트폴리오 데이터 전처리
    portfolio_df = preprocess_user_portfolio_data(user_portfolio_data)

    # 사용자 포트폴리오 출력
    print("\nUser Portfolio Information:")
    print(portfolio_df[["Item", "Portfolio %", "Amount", "Value"]])
    
    # 사용자 포트폴리오 아이템 리스트 생성
    user_portfolio = portfolio_df['Item'].tolist()
    
    # Association Rule 기반 추천
    print("\nChecking for recommendations...")
    if not os.path.exists(association_rules_csv):
        print(f"Association Rules file not found: {association_rules_csv}")
        exit()

    recommended_items, applicable_rules = recommend_items_based_on_association_rule(user_portfolio, association_rules_csv)

    # 추천 결과 출력
    if recommended_items:
        print("\nRecommended Items:")
        for item in recommended_items:
            print(f" - {item}")
        
        # print("\nApplicable Association Rules:")
        # print(applicable_rules[["antecedent", "consequent", "confidence", "support"]])
        
    else:
        print("No recommendations available based on the current Association Rules.")

