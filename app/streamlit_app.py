import sys
from pathlib import Path
import pandas as pd
import fitz  
import streamlit as st
from docx import Document

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.tfidf_summarizer import generate_tfidf_summary
from src.textrank_summarizer import summarizer_text_rank
from src.rouge_evaluator import evaluate_rouge


def read_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def read_pdf(uploaded_file):
    text = ""
    pdf_bytes = uploaded_file.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            text += page.get_text() + "\n"
    return text


def read_docx(uploaded_file):
    document = Document(uploaded_file)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".txt"):
        return read_txt(uploaded_file)
    if file_name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if file_name.endswith(".docx"):
        return read_docx(uploaded_file)
    return ""


st.set_page_config(
    page_title="Résumé de texte",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Masque uniquement le bouton Deploy, le menu 3 points et le footer */
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        [data-testid="stDecoration"] {
            display: none !important;
        }
        footer {
            visibility: hidden !important;
        }
        /* Réduit l'espace vide en haut tout en gardant le bouton '>' de la sidebar */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Paramètres")

    lang_choice = st.selectbox(
        "Langue du document",
        options=["Français", "Anglais"],
        index=0,
    )
    language = "french" if lang_choice == "Français" else "english"

    length_choice = st.select_slider(
        "Taille du résumé",
        options=["Très court", "Moyen", "Détaillé"],
        value="Moyen",
        help="Très court : 3 phrases | Moyen : 5 phrases | Détaillé : 8 phrases",
    )
    length_map = {
        "Très court": 3,
        "Moyen": 5,
        "Détaillé": 8,
    }
    n_sentences = length_map[length_choice]

    st.divider()

    show_perf = st.toggle(
        "Visualiser les performances",
        value=False,
        help="Affiche l'évaluation comparative (scores ROUGE) et les graphiques d'analyse.",
    )


st.title("Résumeur de texte")
st.markdown(
    "Collez un texte ou importez un document (**PDF, Word ou Texte**) "
    "pour en extraire automatiquement l'essentiel."
)

uploaded_file = st.file_uploader(
    "Importer un fichier (optionnel)",
    type=["txt", "pdf", "docx"],
    help="Formats acceptés : .txt, .pdf, .docx",
)

default_text = extract_text_from_file(uploaded_file) if uploaded_file else ""

article_text = st.text_area(
    "Votre texte :",
    value=default_text,
    height=230,
    placeholder="Collez ici l'article, le rapport ou le cours que vous souhaitez résumer...",
)

if article_text.strip():
    st.caption(f"Document prêt : **{len(article_text.split())} mots**")

reference_text = ""
if show_perf:
    st.info("Mode performance activé : ajoutez un résumé de référence ci-dessous pour calculer les scores de précision.")
    uploaded_ref = st.file_uploader(
        "Importer un résumé de référence (.txt, .pdf, .docx) — optionnel",
        type=["txt", "pdf", "docx"],
        key="up_ref",
    )
    default_ref = extract_text_from_file(uploaded_ref) if uploaded_ref else ""
    reference_text = st.text_area(
        "Résumé de référence (pour la comparaison) :",
        value=default_ref,
        height=130,
        placeholder="Collez ici le résumé humain attendu pour mesurer la précision...",
    )


st.write("")
if st.button("Résumer mon texte", type="primary", use_container_width=True):
    if not article_text.strip():
        st.warning("Veuillez d'abord coller un texte ou importer un fichier.")
    else:
        with st.spinner("Rédaction de la synthèse en cours..."):
            summary_1, _, _, _ = generate_tfidf_summary(
                article_text, n_sentences=n_sentences, language=language
            )

            figures_dir = str(ROOT_DIR / "outputs/figures")
            results_dir = str(ROOT_DIR / "outputs/results")
            summary_2, fig_paths = summarizer_text_rank(
                article_text,
                sentence_count=n_sentences,
                language=language,
                output_dir=figures_dir,
            )

            rouge_rows = []
            if reference_text.strip():
                rouge_rows = evaluate_rouge(
                    reference_text,
                    {"Proposition 1 (TF-IDF)": summary_1, "Proposition 2 (TextRank)": summary_2},
                    results_dir=results_dir,
                    figures_dir=figures_dir,
                )

            st.session_state["user_summary"] = {
                "orig_words": max(1, len(article_text.split())),
                "s1": summary_1,
                "s2": summary_2,
                "rouge": rouge_rows,
                "figures": fig_paths,
            }


if "user_summary" in st.session_state:
    res = st.session_state["user_summary"]
    st.divider()
    st.subheader("Votre résumé")

    tab_main, tab_alt = st.tabs(["Synthèse recommandée", "Alternative"])

    with tab_main:
        w2 = len(res["s2"].split())
        red2 = max(0, round((1 - w2 / res["orig_words"]) * 100))
        st.caption(f"Temps de lecture réduit de **{red2}%** ({w2} mots)")
        st.success(res["s2"])
        st.download_button(
            "Télécharger ce résumé (.txt)",
            data=res["s2"],
            file_name="resume.txt",
            mime="text/plain",
            key="dl_main",
        )

    with tab_alt:
        w1 = len(res["s1"].split())
        red1 = max(0, round((1 - w1 / res["orig_words"]) * 100))
        st.caption(f"Temps de lecture réduit de **{red1}%** ({w1} mots)")
        st.info(res["s1"])
        st.download_button(
            "Télécharger cette version (.txt)",
            data=res["s1"],
            file_name="resume_alternatif.txt",
            mime="text/plain",
            key="dl_alt",
        )

    if show_perf:
        st.divider()
        st.subheader("Analyse des performances")

        tab_scores, tab_graphs = st.tabs(
            ["Scores de précision (ROUGE)", "Graphes d'analyse des phrases"]
        )

        with tab_scores:
            if res["rouge"]:
                df_rouge = pd.DataFrame(res["rouge"])
                avg_f1 = df_rouge.groupby("methode")["f1"].mean()
                best = avg_f1.idxmax()

                st.metric(
                    label="Approche la plus fidèle au résumé de référence",
                    value=best,
                    delta=f"Score F1 moyen : {avg_f1[best]:.4f}",
                )

                c_tbl, c_img = st.columns([1.2, 1])
                with c_tbl:
                    st.dataframe(df_rouge, use_container_width=True, hide_index=True)
                with c_img:
                    rouge_img = ROOT_DIR / "outputs/figures/rouge_comparison.png"
                    if rouge_img.exists():
                        st.image(str(rouge_img), use_container_width=True)
            else:
                st.info(
                    "Aucun résumé de référence fourni. Ajoutez-en un en haut puis relancez "
                    "pour afficher le tableau et la courbe des scores ROUGE."
                )

        with tab_graphs:
            if res["figures"]:
                g1, g2 = st.columns(2)
                for idx, path in enumerate(res["figures"]):
                    target = g1 if idx % 2 == 0 else g2
                    with target:
                        if Path(path).exists():
                            st.image(path, use_container_width=True)
            else:
                st.info("Aucun graphique généré pour ce texte.")

        st.markdown("#### Interprétation des résultats")
        w_orig = res["orig_words"]
        w_tr = len(res["s2"].split())
        w_tf = len(res["s1"].split())
        red_tr = max(0, round((1 - w_tr / w_orig) * 100))
        red_tf = max(0, round((1 - w_tf / w_orig) * 100))

        if res["rouge"]:
            df_rouge = pd.DataFrame(res["rouge"])
            avg_f1 = df_rouge.groupby("methode")["f1"].mean()
            best = avg_f1.idxmax()
            other = [m for m in avg_f1.index if m != best][0]
            diff = avg_f1[best] - avg_f1[other]

            st.info(
                f"**Verdict :** **{best}** obtient la meilleure correspondance globale avec le résumé de référence "
                f"(score F1 moyen de **{avg_f1[best]:.4f}**, soit **+{diff:.4f}** par rapport à {other}).\n\n"
                f"- **Compression du texte :** La synthèse recommandée réduit le document initial ({w_orig} mots) "
                f"de **{red_tr}%** ({w_tr} mots), contre **{red_tf}%** ({w_tf} mots) pour l'alternative.\n"
                f"- **Comment lire les scores :** *ROUGE-1* mesure les mots-clés retrouvés, *ROUGE-2* les expressions "
                f"de deux mots consécutifs, et *ROUGE-L* la fidélité de la structure des phrases."
            )
        else:
            st.info(
                f"**Bilan de la synthèse :** Le document original de **{w_orig} mots** a été condensé à "
                f"**{w_tr} mots** pour la synthèse recommandée (**-{red_tr}%**) et **{w_tf} mots** pour l'alternative (**-{red_tf}%**).\n\n"
                f"- **Synthèse recommandée (TextRank) :** Privilégie les phrases les plus centrales et connectées aux idées principales du texte.\n"
                f"- **Alternative (TF-IDF) :** Met en avant les phrases contenant les termes les plus spécifiques et distinctifs du document."
            )
            
