import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

# Step 1: 각 페이지에서 지갑 주소와 전체 표 정보 수집
def get_top_accounts(page_num):
    url = f"https://etherscan.io/accounts/{page_num}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36'
    }
    accounts = []
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 각 지갑 정보가 포함된 테이블 찾기
    table = soup.find('table')
    if table is None:
        print(f"Failed to find table on page {page_num}")
        return accounts  # 빈 리스트 반환
    
    rows = table.find_all('tr')[1:]  # 첫 번째 행은 헤더이므로 제외

    for row in rows:
        cols = row.find_all('td')
        rank = cols[0].text.strip()  # Rank 정보를 첫 번째 열에서 가져옴
        address = cols[1].find('a').get('href').split('/')[-1].strip()  # 전체 주소 추출
        name_tag = cols[2].text.strip()
        balance = cols[3].text.strip()
        percentage = cols[4].text.strip()
        txn_count = cols[5].text.strip()
        accounts.append({
            "Rank": rank,
            "Address": address,
            "Name Tag": name_tag,
            "Balance": balance,
            "Percentage": percentage,
            "Txn Count": txn_count
        })
    
    return accounts

# Step 2: 개별 지갑 주소 페이지에서 포트폴리오 정보 수집
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

# Step 3: 메인 함수 실행
def main():
    all_accounts_data = []
    all_portfolio_data = []

    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created folder: {data_folder}")
        
    # 여러 페이지에서 top accounts 가져오기
    for page_num in range(1, 401):  # 페이지 1~400까지 순회
        print(f"Fetching page {page_num}")
        top_accounts = get_top_accounts(page_num)
        all_accounts_data.extend(top_accounts)
        time.sleep(1)  # 서버로부터 블락을 피하기 위해 딜레이 추가

    # top accounts DataFrame으로 변환
    df_accounts = pd.DataFrame(all_accounts_data)
    accounts_file = os.path.join(data_folder, "etherscan_top_accounts.csv")
    df_accounts.to_csv(accounts_file, index=False)
    print(f"Top accounts data saved to {accounts_file}")

    # 각 주소의 portfolio 데이터 가져오기
    for idx, account_info in enumerate(all_accounts_data, start=1):
        address = account_info["Address"]
        rank = account_info["Rank"]
        print(f"Fetching portfolio for address: {address} (Rank: {rank}, Progress: {idx}/{len(all_accounts_data)})")
        portfolio = get_wallet_portfolio(address)
        all_portfolio_data.extend(portfolio)
        time.sleep(1)  # 서버로부터 블락을 피하기 위해 딜레이 추가

    # portfolio DataFrame으로 변환
    df_portfolio = pd.DataFrame(all_portfolio_data)

    # Address 컬럼을 기준으로 병합
    df_merged = pd.merge(df_portfolio, df_accounts, on="Address", how="left")

    # CSV 파일로 저장
    merged_file = os.path.join(data_folder, "etherscan_merged_data.csv")
    df_merged.to_csv(merged_file, index=False)
    print(f"Data saved to {merged_file}")
    
    
if __name__ == "__main__":
    main()
