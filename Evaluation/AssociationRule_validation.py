import pandas as pd
import os, time

# Test Set의 개별 address의 itemset을 기반으로 Association Rule을 검증 (직접 탐색 방식)
def validate_association_rules_direct(test_df, association_rules_csv):
    # Association Rules 로드
    association_rules = pd.read_csv(association_rules_csv)

    evaluation_results = []

    for _, test_row in test_df.iterrows():
        test_address = test_row['Address']
        test_items = set(test_row['Items'])  # Test 사용자의 포트폴리오

        # Association Rule 전체에서 Test 사용자와 관련된 Rule 필터링
        relevant_rules = association_rules[
            association_rules['antecedent'].apply(lambda x: set(eval(x)).issubset(test_items))
        ]

        # 추천 아이템 생성
        recommended_items = set()
        for _, rule in relevant_rules.iterrows():
            consequent = set(eval(rule['consequent']))
            recommended_items.update(consequent)

        # Test 사용자의 실제 아이템과 비교
        correct_recommendations = recommended_items.intersection(test_items)

        # 결과 기록
        evaluation_results.append({
            "Address": test_address,
            "Recommended": len(recommended_items),
            "Correct": len(correct_recommendations),
            "TestItems": len(test_items),
            "Precision": len(correct_recommendations) / len(recommended_items) if recommended_items else 0,
            "Recall": len(correct_recommendations) / len(test_items) if test_items else 0
        })

    # DataFrame으로 변환
    results_df = pd.DataFrame(evaluation_results)

    # 전체 Precision, Recall, F1-Score 계산
    total_recommended = results_df['Recommended'].sum()
    total_correct = results_df['Correct'].sum()
    total_test_items = results_df['TestItems'].sum()

    precision = total_correct / total_recommended if total_recommended > 0 else 0
    recall = total_correct / total_test_items if total_test_items > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Validation Metrics with Direct Search:\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    return results_df, precision, recall, f1

def run_validation():
    # 고유 식별자를 위한 타임스탬프 (수동 입력 허용)
    timestamp = input("Enter a timestamp (ex. 20231125-123456): ")
    
    # min_support와 min_confidence 범위 설정
    min_support_values = [y / 100 for y in range(5, 21)]  # 0.05 ~ 0.2 (0.01 간격)
    min_confidence_values = [y / 100 for y in range(50, 81, 10)]  # 0.5 ~ 0.8 (0.1 간격)

    # 파일 경로 설정
    data_folder = "data"
    train_test_folder = os.path.join(data_folder, f"association_rules_train_test_{timestamp}")
    if not os.path.exists(train_test_folder):
        print(f"File not found: {train_test_folder}")
        exit()

    # 입력 데이터 경로
    test_csv = os.path.join(train_test_folder, "test_set.csv")
    train_csv = os.path.join(train_test_folder, "training_set.csv")
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(test_csv)
    test_df['Items'] = test_df['Items'].apply(eval)  # 문자열을 리스트로 변환

    # 학습 데이터 로드
    train_df = pd.read_csv(train_csv)
    train_df['Items'] = train_df['Items'].apply(eval)

    # 다양한 min_support와 min_confidence 조합에 대한 결과 저장
    results_summary = []
    
    for min_support in min_support_values:
        for min_confidence in min_confidence_values:
            print(f"\nValidating for min_support={min_support}, min_confidence={min_confidence}")

            # Association Rules 파일 경로 설정
            association_rules_csv = os.path.join(
                train_test_folder, f"association_rules_train_{min_support}_{min_confidence}.csv"
            )

            # 직접 탐색 방식 검증
            print("Running Direct Search Validation...")
            results_direct, precision_d, recall_d, f1_d = validate_association_rules_direct(test_df, association_rules_csv)

            # 결과 저장

            results_summary.append({
                "min_support": min_support,
                "min_confidence": min_confidence,
                "Validation_Type": "Direct",
                "Precision": precision_d,
                "Recall": recall_d,
                "F1-Score": f1_d
            })

            #결과 저장
            results_direct_csv = os.path.join(
                train_test_folder, f"results_direct_{min_support}_{min_confidence}.csv"
            )
            results_direct.to_csv(results_direct_csv, index=False)
            print(f"Direct Search Validation results saved to {results_direct_csv}")

    # Summary DataFrame 생성 및 저장
    summary_df = pd.DataFrame(results_summary)
    summary_csv = os.path.join(train_test_folder, f"validation_summary_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nValidation summary saved to {summary_csv}")

if __name__ == "__main__":
    run_validation()