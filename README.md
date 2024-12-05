# Evaluation Branch

This branch focuses on the evaluation of the **Whale Investors-Favored Crypto Recommender System**, assessing the performance of both **Association Rule Mining (ARM)** and **Collaborative Filtering (CF)** methods. The evaluation process includes precision, recall, and F1-score calculations for ARM and RMSE calculations for CF to validate the effectiveness of the implemented recommendation algorithms.

---

## Directory Structure

```plaintext
.
├── AssociationRule_Recommendation.py           # ARM-based recommendation script
├── AssociationRule_validation.py               # Evaluates ARM rules against test data
├── AssociationRule_validation_visualization.py # Visualizes ARM evaluation results
├── Collaborative_Filtering_Baseline_Predictor_validation.py # Evaluates CF using RMSE
├── SON_Apriori_FrequentItemsets_AssociationRule.py # SON algorithm for ARM
├── data                                        # Contains intermediate and evaluation datasets
│   ├── association_rules_train_test_*.csv      # ARM rules for specific train/test splits
│   ├── association_rules_full_*.csv            # ARM rules generated from the full dataset
│   └── evaluation_summary.csv                  # Summary of evaluation metrics
```
## Evaluation Process
1. **Association Rule Mining (ARM)**:
    - **Rule Generation**:
      - Run the `SON_Apriori_FrequentItemsets_AssociationRule.py` script to generate association rules.
      - Supports two modes:
          1. **Full Dataset**: Generates rules for all data using combinations of min-support (0.05–0.2) and min-confidence (0.6–0.8).
          2. **Train/Test Split**: Splits data into 80% train and 20% test, then generates rules from the train set.
      - Results are saved in the data/ folder:
          - association_rules_full/association_rules_full_{min_support}_{min_confidence}.csv for the full dataset.
          - association_rules_train_test_{timestamp}/association_rules_train_{min_support}_{min_confidence}.csv for train/test splits.
    - **Rule Evaluation**:
      - Use `AssociationRule_validation.py` to evaluate precision, recall, and F1-score for each rule combination on the test data.
    - **Visualization**:
      - Run `AssociationRule_validation_visualization.py` to visualize the evaluation metrics (e.g., precision, recall, F1-score) across different combinations of min-support and min-confidence.
    - **Final Selection**:
      - Based on evaluation results, `min-support = 0.05` and `min-confidence = 0.7` were selected as the optimal parameters for ARM. These rules are used in the recommendation system.
    
2. **Collaborative Filtering (CF)**:
    - **Evaluation**:
      - Run `Collaborative_Filtering_Baseline_Predictor_validation.py` to evaluate CF performance.
      - Randomly removes a portion of the values (1%) in the User-Item Matrix to simulate missing data.
      - Predicts the removed values using both `Centered Cosine Similarity` and `Non-Centered`.
    - **Results**:
      - Centered User-Based CF: RMSE = 0.2631
      - Non-Centered User-Based CF: RMSE = 0.3303
      - Centered Item-Based CF: RMSE = 0.2268
      - Non-Centered Item-Based CF: RMSE = 0.1748
    - **Conclusion**:
      - User-Based CF performed better with centered matrices.
      - Item-Based CF performed better with non-centered matrices.
      - Item-Based CF consistently achieved lower RMSE, indicating higher predictive accuracy.
    - **Summary**:
      - The evaluation validated the use of centered User-Based CF and non-centered Item-Based CF in the final scoring mechanism.

3. **How to Run Evaluations**:
    1. Association Rule Mining:
       - Generate Rules:
        ```bash
        python SON_Apriori_FrequentItemsets_AssociationRule.py
        ```
       - Evaluate Rules:
        ```bash
        python AssociationRule_validation.py
        ```
       - Visualize Results:
        ```bash
        python AssociationRule_validation_visualization.py
        ```
    2. Collaborative Filtering:
        - Evaluate CF:
        ```bash
        python Collaborative_Filtering_Baseline_Predictor_validation.py
        ```