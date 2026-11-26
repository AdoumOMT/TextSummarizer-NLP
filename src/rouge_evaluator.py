from pathlib import Path
import csv
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

METRICS = ["rouge1", "rouge2", "rougeL"]
NAMES = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L"}


def evaluate_rouge(
    reference_text,
    summaries,
    results_dir="outputs/results",
    figures_dir="outputs/figures",
):
    """
    Évalue les résumés générés par rapport au résumé de référence (ROUGE-1, ROUGE-2, ROUGE-L).
    Sauvegarde le CSV et l'interprétation dans `results_dir`, et le graphique dans `figures_dir`.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)
    rows = []

    for method, summary in summaries.items():
        scores = scorer.score(reference_text, summary)
        for metric in METRICS:
            value = scores[metric]
            rows.append({
                "methode": method,
                "metrique": NAMES[metric],
                "precision": round(value.precision, 4),
                "recall": round(value.recall, 4),
                "f1": round(value.fmeasure, 4),
            })

    save_csv(rows, f"{results_dir}/rouge_results.csv")
    save_plot(rows, f"{figures_dir}/rouge_comparison.png")
    save_interpretation(rows, f"{results_dir}/rouge_interpretation.txt")
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows, path):
    methods = sorted(set(row["methode"] for row in rows))
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    plt.figure(figsize=(8, 5))
    for method in methods:
        values = [row["f1"] for row in rows if row["methode"] == method]
        plt.plot(metrics, values, marker="o", linewidth=2, label=method)

    plt.title("Comparaison ROUGE des méthodes", fontweight="bold")
    plt.xlabel("Métrique")
    plt.ylabel("Score F1")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_interpretation(rows, path):
    averages = {}
    for row in rows:
        averages.setdefault(row["methode"], []).append(row["f1"])

    averages = {method: sum(values) / len(values) for method, values in averages.items()}
    best = max(averages, key=averages.get)

    text = (
        f"La meilleure méthode est {best} (score F1 moyen = {averages[best]:.4f}).\n"
        "- ROUGE-1 mesure le chevauchement des mots individuels (unigrammes).\n"
        "- ROUGE-2 mesure le chevauchement des paires de mots consécutifs (bigrammes).\n"
        "- ROUGE-L mesure la plus longue séquence commune de mots (structure de phrase).\n"
    )

    Path(path).write_text(text, encoding="utf-8")
    print("\nInterprétation ROUGE :")
    print(text)