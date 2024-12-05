# Whale Investors-Favored Crypto Recommender System

This project develops a cryptocurrency recommendation system leveraging whale portfolio insights to provide actionable investment suggestions. By combining **Association Rule Mining (ARM)** and **Collaborative Filtering (CF)** techniques, the system generates personalized and trend-based recommendations. 

---

## **Project Overview**

### **Key Features**
1. **Association Rule Mining**:
   - Custom **Apriori-based SON Algorithm** implemented using **MapReduce on Spark**.
   - Identifies frequent itemsets and generates association rules reflecting whale trading patterns.

2. **Collaborative Filtering**:
   - Implemented **User-Based CF (Centered cosine similarity)** and **Item-Based CF (Non-Centered cosine similarity)** using a **Baseline Predictor** for personalized recommendations.

3. **Scoring Integration**:
   - Final recommendations combine ARM and CF results using the weighted scoring formula:
     ```
     Final Score = W1 · Association Score + W2 · User-Based Score + W3 · Item-Based Score + Bonus Score
     ```
   - **Weights and Adjustments**:
     - **Association Rule**: W1 = 2 (prioritizes macro-level market trends).
     - **User-Based CF**: W2 = 1.5.
     - **Item-Based CF**: W3 = 1.75 (higher accuracy based on RMSE).
     - **Bonus Points**: +10 for overlap in 2 methods; +20 for overlap in all 3 methods.
   - **Diversity**:
     - Top 8 recommendations are based on highest scores.
     - Final 2 recommendations are randomly selected from lower-ranked results.

---

## **Project Environment**

- **Operating Systems**: 
  - Windows 11 Pro 23H2 (22631.4460)
- **Programming Language**: Python 3.10.0
- **Big Data Framework**: Apache Spark 3.5.3 (Hadoop 3)
- **Hardware**:
  - **CPU**: Intel Core i9-13900K
  - **RAM**: 64GB
- **Key Libraries**:
  - **Data Processing**: pandas, pyspark
  - **Scraping**: BeautifulSoup
  - **Similarity Calculation**: sklearn
  - **Visualization**: matplotlib, seaborn

---

## **Directory Structure**
```
DataMining_Project/
├── data/ 
│   ├── association_rules_full_base_0.05_0
│   ├── association_rules_full_base_0.05_0.7.csv      # Precomputed association rules
│   ├── etherscan_top_accounts.csv                    # Top 10,000 wallet details
│   ├── etherscan_merged_data.csv                     # Portfolio scraped from wallets
│   ├── preprocessed_data.csv                         # Cleaned whale portfolio dataset
│   ├── unique_bucket_itemsets.csv                    # Data for frequent itemsets & rules
│   └── duplicate_items.csv                           # Duplicate identification
├── SON_Apriori_FrequentItemsets_AssociationRule.py   # Custom SON algorithm for ARM
├── AssociationRule_Recommendation.py                # Association Rule recommendation
├── Collaborative_Filtering_Baseline_Predictor.py    # User-Based & Item-Based CF
├── Cryptocurrency_Recommendation.py                 # Main recommendation system
├── etherscan_scraper.py                              # Wallet scraping script
├── etherscan_data_preprocessing.py                  # Data preprocessing script
├── README.md                                        # Project documentation
```
## **How to Run**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gyuminb/DataMining_Project.git
   cd DataMining_Project
   ```
    
2. **Set Up Environment**:
    - Ensure Python 3.10+ and Apache Spark 3.5.3 are installed.
    - Install required Python libraries:
    ```bash
    pip install pandas pyspark beautifulsoup4 matplotlib seaborn scikit-learn
    ```

3. **Run the Main System**:
    - Execute the Cryptocurrency_Recommendation.py script:
    ```bash
    python Cryptocurrency_Recommendation.py
    ```
    - Input your wallet address when prompted to receive personalized cryptocurrency recommendations.

4. **Outputs**:
    - ARM Recommendations: Derived from frequent itemsets and association rules.
    - CF Recommendations: Predictions from User-Based and Item-Based CF.
    - Final Recommendations: Top-ranked coins based on scoring integration.
