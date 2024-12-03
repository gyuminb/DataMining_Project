import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def visualize_2d_heatmap(summary_csv):
    # 결과 요약 파일 로드
    summary_df = pd.read_csv(summary_csv)

    # Validation 방식별로 분리
    validation_types = summary_df["Validation_Type"].unique()

    for validation_type in validation_types:
        subset = summary_df[summary_df["Validation_Type"] == validation_type]

        # 서브플롯 생성 (3x1 레이아웃)
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        metrics = ["Precision", "Recall", "F1-Score"]
        for i, metric in enumerate(metrics):
            # DataFrame.pivot()을 올바르게 사용
            heatmap_data = subset.pivot(index="min_support", columns="min_confidence", values=metric)
            sns.heatmap(
                heatmap_data,
                annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i],
                cbar_kws={'label': f"{metric} Score"}  # 색상 막대 레이블 추가
            )
            axes[i].set_title(f"{validation_type} - {metric}", fontsize=14)
            axes[i].set_xlabel("Min Confidence", fontsize=12)
            axes[i].set_ylabel("Min Support", fontsize=12)

            # x축 텍스트 위치 조정
            axes[i].tick_params(axis='x', labelsize=10, rotation=45, pad=10)  # pad로 아래로 이동
            axes[i].tick_params(axis='y', labelsize=10)
            
            # y-axis label 간격 조정
            axes[i].yaxis.label.set_size(12)

        plt.tight_layout(pad=2.5)  # 여백 추가
        plt.subplots_adjust(hspace=0.4)  # 서브플롯 간 간격 조정
        plt.show()





def visualize_3d_surface(summary_csv):
    # 결과 요약 파일 로드
    summary_df = pd.read_csv(summary_csv)

    # Validation 방식별로 분리
    validation_types = summary_df["Validation_Type"].unique()

    for validation_type in validation_types:
        subset = summary_df[summary_df["Validation_Type"] == validation_type]

        # 3D 플롯 생성
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # X, Y, Z 값
        X = subset["min_support"]
        Y = subset["min_confidence"]
        Z = subset["F1-Score"]  # Change to "Precision" or "Recall" for other metrics

        # 3D Surface Plot
        surf = ax.plot_trisurf(X, Y, Z, cmap=cm.viridis, linewidth=0.2)

        ax.set_title(f"{validation_type} - F1-Score Surface")
        ax.set_xlabel("Min Support")
        ax.set_ylabel("Min Confidence")
        ax.set_zlabel("F1-Score")

        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.show()


if __name__ == "__main__":
    # Summary 파일 경로 입력
    summary_csv = input("Enter the path to the summary CSV file: ")

    if not os.path.exists(summary_csv):
        print(f"File not found: {summary_csv}")
    else:
        print("\nGenerating 2D Heatmaps...")
        visualize_2d_heatmap(summary_csv)

        print("\nGenerating 3D Surface Plots...")
        visualize_3d_surface(summary_csv)
