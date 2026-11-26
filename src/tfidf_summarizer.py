import math
from collections import Counter

from src.preprocessing import split_into_sentences, preprocess_sentences


def compute_idf(cleaned_sentences):
    """
    Calcule l'IDF de chaque mot.

    Formule :
        IDF = log((N + 1) / (df + 1)) + 1

    N  : nombre total de phrases
    df : nombre de phrases contenant le mot

    Retour :
        dict : mot -> score IDF
    """

    total_sentences = len(cleaned_sentences)
    document_frequency = {}

    for words in cleaned_sentences:
        unique_words = set(words)

        for word in unique_words:
            document_frequency[word] = document_frequency.get(word, 0) + 1

    idf_scores = {}

    for word, df in document_frequency.items():
        idf_scores[word] = math.log((total_sentences + 1) / (df + 1)) + 1

    return idf_scores


def compute_tfidf_scores(sentences, language="french"):
    """
    Calcule un score TF-IDF pour chaque phrase.

    Paramètres :
        sentences : list[str]
            Liste des phrases originales.
        language : str
            Langue du texte.

    Retour :
        list[float] : scores normalisés entre 0 et 1.
    """

    cleaned_sentences = preprocess_sentences(sentences, language=language)

    if len(cleaned_sentences) == 0:
        return []

    idf_scores = compute_idf(cleaned_sentences)

    raw_scores = []

    for words in cleaned_sentences:
        if len(words) == 0:
            raw_scores.append(0)
            continue

        word_counts = Counter(words)
        sentence_length = len(words)

        score = 0

        for word, count in word_counts.items():
            tf = count / sentence_length
            idf = idf_scores.get(word, 0)
            score += tf * idf

        raw_scores.append(score)

    normalized_scores = normalize_scores(raw_scores)

    return normalized_scores


def normalize_scores(scores):
    """
    Normalise les scores entre 0 et 1.

    Formule :
        score_normalisé = (score - min) / (max - min)
    """

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0 for _ in scores]

    normalized = [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]

    return normalized


def rank_sentences(sentences, scores):
    """
    Associe chaque phrase à sa position et à son score.
    """

    sentence_data = []

    for index, sentence in enumerate(sentences):
        sentence_data.append({
            "index": index,
            "sentence": sentence,
            "score": scores[index]
        })

    ranked_sentences = sorted(
        sentence_data,
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked_sentences


def generate_tfidf_summary(text, n_sentences=5, language="french"):
    """
    Génère un résumé extractif avec TF-IDF.

    Étapes :
    1. Découper le texte en phrases.
    2. Calculer le score TF-IDF de chaque phrase.
    3. Sélectionner les top-N phrases.
    4. Remettre les phrases sélectionnées dans l'ordre original.

    Retour :
        summary : str
        selected_sentences : list[str]
        scores : list[float]
        ranked_sentences : list[dict]
    """

    sentences = split_into_sentences(text, language=language)

    if len(sentences) == 0:
        return "", [], [], []

    scores = compute_tfidf_scores(sentences, language=language)

    ranked_sentences = rank_sentences(sentences, scores)

    n_sentences = min(n_sentences, len(sentences))

    selected = ranked_sentences[:n_sentences]

    # Très important : remettre les phrases sélectionnées dans l'ordre original.
    selected = sorted(selected, key=lambda item: item["index"])

    selected_sentences = [item["sentence"] for item in selected]

    summary = " ".join(selected_sentences)

    return summary, selected_sentences, scores, ranked_sentences
