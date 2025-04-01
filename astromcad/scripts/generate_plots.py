from utils.figures import *
import pandas as pd
import random

if __name__ == '__main__':
    norm_results = pd.read_csv("scores/norm_scores.csv")
    anom_results = pd.read_csv("scores/anom_scores.csv")

    plot_recall(list(norm_results['score']), random.sample(list(anom_results['score']), 100))
    plt.savefig("../figures/recall.pdf", bbox_inches='tight')
    plt.show()
    print("Generated Recall Plot")

    median_score(norm_results, anom_results)
    plt.savefig("../figures/median_score.pdf", bbox_inches='tight')
    plt.show()
    print("Generated Median Score Plot")

    distribution(norm_results, anom_results)
    plt.savefig("../figures/anomaly_distribution.pdf", bbox_inches='tight')
    plt.show()
    print("Generated Anomaly Distribution Plot")



