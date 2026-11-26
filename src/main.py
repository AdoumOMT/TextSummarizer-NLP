from pathlib import Path

from src.tfidf_summarizer import generate_tfidf_summary
from src.textrank_summarizer import summarizer_text_rank
from src.rouge_evaluator import evaluate_rouge


def read_file(path):
    return Path(path).read_text(encoding="utf-8").strip()


def save_file(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def main():
    # Chargement de l'article 1 et de sa référence correspondante
    article = read_file("data/raw/article_1.txt")
    reference = read_file("data/references/reference_1.txt")

    # Génération des résumés (TF-IDF et TextRank avec figures)
    tfidf_summary, _, _, _ = generate_tfidf_summary(article, n_sentences=5, language="french")
    textrank_summary, figure_paths = summarizer_text_rank(
        article, sentence_count=5, language="french", output_dir="outputs/figures"
    )

    # Sauvegarde des résumés
    save_file("outputs/summaries/summary_tfidf.txt", tfidf_summary)
    save_file("outputs/summaries/summary_textrank.txt", textrank_summary)

    # Évaluation ROUGE
    evaluate_rouge(
        reference,
        {
            "TF-IDF": tfidf_summary,
            "TextRank": textrank_summary,
        },
        results_dir="outputs/results",
        figures_dir="outputs/figures",
    )

    print("\nPipeline terminé avec succès. Fichiers générés :")
    print("- outputs/summaries/summary_tfidf.txt")
    print("- outputs/summaries/summary_textrank.txt")
    print("- outputs/results/rouge_results.csv")
    print("- outputs/results/rouge_interpretation.txt")
    print("- outputs/figures/rouge_comparison.png")
    for fig in figure_paths:
        print(f"- {fig}")


if __name__ == "__main__":
    main()