import nltk

resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
]

for resource in resources:
    try:
        nltk.download(resource, quiet=True)
        print(f"[OK] Ressource '{resource}' prête.")
    except Exception as e:
        print(f"[ERREUR] Impossible de télécharger {resource} : {e}")

print("Téléchargement des ressources NLTK terminé.")