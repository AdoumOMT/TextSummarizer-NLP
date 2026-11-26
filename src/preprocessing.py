import re
import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords


def ensure_nltk_resources():
    """
    Vérifie rapidement que les ressources NLTK nécessaires sont disponibles.
    Si elles ne sont pas présentes, les télécharge automatiquement.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def split_into_sentences(text, language="french"):
    """
    Découpe un texte brut en phrases avec NLTK.
    """
    ensure_nltk_resources()

    if not text or not text.strip():
        return []

    sentences = sent_tokenize(text, language=language)

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


def clean_sentence(sentence, language="french", stop_words=None):
    """
    Nettoie une phrase pour le calcul TF-IDF :
    - mise en minuscules
    - suppression de la ponctuation
    - suppression des stopwords et des mots <= 2 caractères
    """
    if stop_words is None:
        ensure_nltk_resources()
        try:
            stop_words = set(stopwords.words(language))
        except OSError:
            stop_words = set(stopwords.words("french"))

    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]+", sentence)

    return [
        word
        for word in words
        if word not in stop_words and len(word) > 2
    ]


def preprocess_sentences(sentences, language="french"):
    """
    Nettoie une liste de phrases en chargeant les stopwords une seule fois.
    """
    ensure_nltk_resources()
    try:
        stop_words = set(stopwords.words(language))
    except OSError:
        stop_words = set(stopwords.words("french"))

    return [
        clean_sentence(sentence, language=language, stop_words=stop_words)
        for sentence in sentences
    ]