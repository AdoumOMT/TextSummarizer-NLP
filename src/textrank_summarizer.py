import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
from sumy.summarizers.text_rank import TextRankSummarizer

MAX_MATRIX_SENTENCES = 500


def summarizer_text_rank(text, sentence_count=5, language="french", output_dir="outputs/figures"):
    """
    Génère un résumé extractif avec Sumy TextRank et sauvegarde les 4 graphiques
    explicatifs dans `output_dir` (par défaut : outputs/figures).
    """
    os.makedirs(output_dir, exist_ok=True)

    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    document = parser.document

    all_sentence_objects = document.sentences
    sentences = [str(s) for s in all_sentence_objects]
    n = len(sentences)

    if n == 0:
        return "", []

    sentence_count = min(sentence_count, n)
    labels = [f"S{i+1}" for i in range(n)]

    sumy_ratings = summarizer.rate_sentences(document)
    pagerank_scores = {
        i: sumy_ratings.get(sent_obj, 0.0)
        for i, sent_obj in enumerate(all_sentence_objects)
    }

    selected_sentences = summarizer(document, sentence_count)
    selected_texts = [str(s) for s in selected_sentences]

    ranked = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    selected_idx = sorted([idx for idx, _ in ranked[:sentence_count]])

    scores_list = [pagerank_scores[i] for i in range(n)]
    min_selected = min(pagerank_scores[i] for i in selected_idx)

    COL_SELECTED = "#e74c3c"
    COL_OTHER = "#1D9E75"
    paths = []

    def get_colors(total, selected):
        return [COL_SELECTED if i in selected else COL_OTHER for i in range(total)]

    if n <= MAX_MATRIX_SENTENCES:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences).toarray()
        sim_matrix = cosine_similarity(tfidf_matrix)

        cell_size = max(0.3, min(0.8, 30 / n))
        fig_w = max(8, n * cell_size + 2)
        fig_h = max(6, n * cell_size + 1)

        fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
        im = ax1.imshow(sim_matrix, cmap="Greens", interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax1, label="Cosine similarity", shrink=0.8)

        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        xlabels = ax1.set_xticklabels(labels, fontsize=max(5, 9 - n // 10), rotation=45, ha="right")
        ylabels = ax1.set_yticklabels(labels, fontsize=max(5, 9 - n // 10))
        for i, (xl, yl) in enumerate(zip(xlabels, ylabels)):
            c = COL_SELECTED if i in selected_idx else "black"
            xl.set_color(c)
            yl.set_color(c)

        if n <= 30:
            for i in range(n):
                for j in range(n):
                    color = "white" if sim_matrix[i][j] > 0.5 else "black"
                    ax1.text(j, i, f"{sim_matrix[i][j]:.2f}", ha="center", va="center", fontsize=6, color=color)

        for idx in selected_idx:
            ax1.add_patch(plt.Rectangle((idx - 0.5, -0.5), 1, n, fill=False, edgecolor=COL_SELECTED, lw=2.5))
            ax1.add_patch(plt.Rectangle((-0.5, idx - 0.5), n, 1, fill=False, edgecolor=COL_SELECTED, lw=2.5))

        ax1.set_title("Matrice de similarité entre les phrases", fontweight="bold", fontsize=12, pad=12)
        plt.tight_layout()
        fig1_path = os.path.join(output_dir, "fig1_similarity_matrix.png")
        plt.savefig(fig1_path, dpi=100, bbox_inches="tight")
        paths.append(fig1_path)
        plt.close()

    if n <= MAX_MATRIX_SENTENCES:
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > 0.05:
                    G.add_edge(i, j, weight=sim_matrix[i][j])

        fig2, ax2 = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=2.5)
        node_sizes = [pagerank_scores[node] * 15000 for node in G.nodes()]
        node_colors = [COL_SELECTED if i in selected_idx else COL_OTHER for i in G.nodes()]
        edge_widths = [G[u][v]["weight"] * 4 for u, v in G.edges()]

        nx.draw_networkx_edges(G, pos, ax=ax2, width=edge_widths, alpha=0.3, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=node_sizes, node_color=node_colors, alpha=0.92)
        nx.draw_networkx_labels(
            G, pos, ax=ax2,
            labels={i: f"S{i+1}" for i in G.nodes()},
            font_size=8, font_weight="bold", font_color="white"
        )

        pos_lower = {k: (v[0], v[1] - 0.15) for k, v in pos.items()}
        nx.draw_networkx_labels(
            G, pos_lower, ax=ax2,
            labels={i: f"{pagerank_scores[i]:.3f}" for i in G.nodes()},
            font_size=7,
            font_color={i: COL_SELECTED if i in selected_idx else "#444444" for i in G.nodes()}
        )

        legend = [
            Patch(color=COL_SELECTED, label="Selected by Sumy TextRank"),
            Patch(color=COL_OTHER, label="Not selected"),
        ]
        ax2.legend(handles=legend, fontsize=10, loc="upper left")
        ax2.set_title("Graphe TextRank — toutes les phrases", fontweight="bold", fontsize=12, pad=12)
        ax2.axis("off")
        plt.tight_layout()
        fig2_path = os.path.join(output_dir, "fig2_textrank_graph.png")
        plt.savefig(fig2_path, dpi=100, bbox_inches="tight")
        paths.append(fig2_path)
        plt.close()

    fig3, ax3 = plt.subplots(figsize=(max(12, n * 0.6), 6))
    bar_colors = get_colors(n, selected_idx)

    bars = ax3.bar(labels, scores_list, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.7)

    if n <= 40:
        for bar, score, c in zip(bars, scores_list, bar_colors):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(scores_list) * 0.015,
                f"{score:.4f}", ha="center", va="bottom",
                fontsize=max(6, 9 - n // 10), fontweight="bold", color=c
            )

    ax3.axhline(
        y=min_selected, color=COL_SELECTED, linestyle="--", linewidth=2,
        label=f"Selection threshold = {min_selected:.4f} (top {sentence_count} sentences)"
    )
    ax3.legend(fontsize=10)

    for i, tick in enumerate(ax3.get_xticklabels()):
        tick.set_color(COL_SELECTED if i in selected_idx else "black")

    ax3.set_title("Score PageRank par phrase — sélection Sumy", fontweight="bold", fontsize=12, pad=12)
    ax3.set_xlabel("Sentence", fontsize=11)
    ax3.set_ylabel("Sumy PageRank score", fontsize=11)
    ax3.set_ylim(0, max(scores_list) * 1.2)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_pagerank_scores.png")
    plt.savefig(fig3_path, dpi=100, bbox_inches="tight")
    paths.append(fig3_path)
    plt.close()

    fig4, ax4 = plt.subplots(figsize=(14, max(6, n * 0.5)))
    bar_colors_h = get_colors(n, selected_idx)

    ax4.barh(range(n), scores_list, color=bar_colors_h, edgecolor="white", linewidth=0.5, height=0.65)

    for i, (score, sent) in enumerate(zip(scores_list, sentences)):
        is_sel = i in selected_idx
        preview = sent[:70] + "..." if len(sent) > 70 else sent

        ax4.text(
            -0.002, i, f"S{i+1}: {preview}",
            ha="right", va="center", fontsize=max(5, 8 - n // 15),
            color=COL_SELECTED if is_sel else "#333333",
            fontweight="bold" if is_sel else "normal"
        )
        ax4.text(
            score + max(scores_list) * 0.01, i, f"{score:.4f}",
            ha="left", va="center", fontsize=max(5, 8 - n // 15),
            fontweight="bold", color=COL_SELECTED if is_sel else COL_OTHER
        )

    ax4.axvline(x=min_selected, color=COL_SELECTED, linestyle="--", linewidth=2, label=f"Threshold = {min_selected:.4f}")
    ax4.text(min_selected + max(scores_list) * 0.01, n * 0.98, "← selected", color=COL_SELECTED, fontsize=10, fontweight="bold", va="top")
    ax4.text(min_selected - max(scores_list) * 0.01, n * 0.98, "not selected →", color=COL_OTHER, fontsize=10, fontweight="bold", va="top", ha="right")

    ax4.set_yticks([])
    ax4.set_xlabel("Sumy PageRank score", fontsize=11)
    ax4.set_xlim(-0.6, max(scores_list) * 1.3)
    ax4.set_ylim(-0.5, n - 0.5)
    ax4.legend(fontsize=10, loc="lower right")
    ax4.set_title("Comparaison complète — toutes les phrases classées", fontweight="bold", fontsize=12, pad=12)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_all_sentences_ranked.png")
    plt.savefig(fig4_path, dpi=100, bbox_inches="tight")
    paths.append(fig4_path)
    plt.close()

    return " ".join(selected_texts), paths


def generate_textrank_summary(text, sentence_count=5, language="french"):
    """Version légère sans génération de graphiques (utile pour l'évaluation rapide)."""
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in sentences)