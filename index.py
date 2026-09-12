"""
Détecteur d'intention "James" (wake by intent, pas par simple mot-clé)

Idée clé : le mot "James" doit apparaître dans les DEUX classes (positive et
négative) sinon le modèle apprend juste "présence du mot -> wake" et se fait
avoir par des phrases comme "je parlais à James" ou "c'était James au tel".

Classe 1 (wake=1)   : on s'adresse DIRECTEMENT à James, on l'interpelle,
                      on lui donne un ordre / une question.
Classe 0 (wake=0)   : on parle DE James à quelqu'un d'autre, on mentionne
                      son nom sans l'interpeller, ou phrase normale sans
                      rapport.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
import random

MODEL_NAME = "almanach/camembert-base"
OUT_DIR = "./camembert-james-wake"


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

def build_dataset():
    # --- Classe 1 : adresse directe à James (wake) ---
    wake_templates = [
        "James",
        "James !",
        "James ?",
        "James, t'es là ?",
        "James, tu m'entends ?",
        "James tu es là ?",
        "Hé James",
        "Eh James",
        "Salut James",
        "Bonjour James",
        "James, tu peux faire {x} ?",
        "James, peux-tu {x} ?",
        "James tu peux {x}",
        "James, fais {x}",
        "James, lance {x}",
        "James, ouvre {x}",
        "James, mets {x}",
        "James, éteins {x}",
        "James, active {x}",
        "James, dis-moi {x}",
        "James, quelle heure il est ?",
        "James, quel temps fait-il ?",
        "Dis James, {x}",
        "James j'ai besoin de {x}",
        "James, aide-moi avec {x}",
        "OK James",
        "Hey James, {x}",
        "James, arrête",
        "James, stop",
        "James répète",
        "James recommence",
        "James, ça marche pas",
        "James, tu peux répéter ?",
    ]

    x_fillers = [
        "la lumière", "la musique", "le minuteur", "mes mails",
        "la météo", "un rappel", "la liste de courses", "la vidéo",
        "l'alarme", "le chauffage", "la porte", "le calcul",
        "une recherche", "un résumé", "la traduction", "le café",
    ]

    wake_texts = []
    for t in wake_templates:
        if "{x}" in t:
            for x in x_fillers:
                wake_texts.append(t.format(x=x))
        else:
            wake_texts.append(t)

    # --- Classe 0 : mention de James SANS s'adresser à lui, ou phrases neutres ---
    mention_templates = [
        "Je parlais à James tout à l'heure",
        "Non c'était James",
        "J'ai vu James hier",
        "James est parti au travail",
        "Tu as parlé à James ?",
        "James m'a dit que tout allait bien",
        "C'est James qui a fait ça",
        "James n'est pas là",
        "Demande à James",
        "James et moi on est allés au cinéma",
        "James a appelé pendant que tu dormais",
        "Je crois que James a raison",
        "James, c'est mon collègue",
        "Elle s'appelle Jamesa, pas James",
        "James travaille avec Paul",
        "On a mangé avec James et Sophie",
        "James est en retard",
        "Ce n'est pas James, c'est Julien",
        "James adore le café",
        "James m'a envoyé un message",
        "Je vais chez James ce soir",
        "James habite à côté de chez moi",
        "Tu te souviens de James ?",
        "James avait raison sur ce point",
        "Mon frère s'appelle James",
        "James, il est sympa",
        "On parlait de James justement",
        "James viendra demain",
    ]

    neutral_templates = [
        "Salut",
        "Ça va",
        "Bonjour",
        "Comment ça va ?",
        "Il fait beau aujourd'hui",
        "Je vais faire les courses",
        "Qu'est-ce qu'on mange ce soir ?",
        "J'ai fini mon travail",
        "Tu as vu le match hier ?",
        "Je suis fatigué",
        "On se voit demain",
        "Merci beaucoup",
        "De rien",
        "À plus tard",
        "Bonne nuit",
        "Quelle heure est-il ?",
        "Je n'ai pas compris",
        "Peux-tu répéter ?",
        "C'est une bonne idée",
        "Je ne sais pas",
        "On y va ?",
        "Attends une seconde",
        "C'est fini",
        "Il pleut dehors",
        "Je pense que oui",
        "Pas maintenant",
        "Ok, d'accord",
        "Je suis d'accord avec toi",
    ]

    neg_texts = mention_templates + neutral_templates

    # Équilibrage approximatif : on duplique un peu les négatifs pour
    # matcher l'ordre de grandeur des positifs, avec un peu de bruit.
    random.seed(42)
    neg_texts_expanded = neg_texts * 6
    random.shuffle(neg_texts_expanded)
    neg_texts_expanded = neg_texts_expanded[: len(wake_texts)]

    texts = wake_texts + neg_texts_expanded
    labels = [1] * len(wake_texts) + [0] * len(neg_texts_expanded)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# 2. Entraînement
# ---------------------------------------------------------------------------

def train_wake_model():
    print("🎓 Fine-tuning CamemBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    texts, labels = build_dataset()
    print(f"📊 Dataset total : {len(texts)} exemples "
          f"({sum(labels)} wake / {len(labels) - sum(labels)} non-wake)")

    # Split train / eval pour vérifier que ça généralise vraiment
    n_eval = max(20, int(0.1 * len(texts)))
    eval_texts, eval_labels = texts[:n_eval], labels[:n_eval]
    train_texts, train_labels = texts[n_eval:], labels[n_eval:]

    def encode(t):
        return tokenizer(t, truncation=True, padding=True, max_length=64)

    train_ds = Dataset.from_dict({
        **encode(train_texts),
        "labels": train_labels,
    })
    eval_ds = Dataset.from_dict({
        **encode(eval_texts),
        "labels": eval_labels,
    })

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"📈 Eval loss finale : {metrics.get('eval_loss'):.4f}")

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("✅ Modèle sauvegardé !")
    return True


# ---------------------------------------------------------------------------
# 3. Détecteur en usage
# ---------------------------------------------------------------------------

class WakeDetector:
    def __init__(self, threshold: float = 0.6):
        self.tokenizer = AutoTokenizer.from_pretrained(OUT_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR)
        self.model.eval()
        self.is_active = False
        self.threshold = threshold

    def predict_proba(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=-1)[0][1].item()

    def process(self, text: str) -> bool:
        if self.is_active:
            print(f"🎤 IA active → commande : {text}")
            if "éteins" in text.lower() or "stop" in text.lower():
                self.is_active = False
                print("😴 IA désactivée")
            return True

        proba_wake = self.predict_proba(text)
        if proba_wake > self.threshold:
            self.is_active = True
            print(f"🚀 IA activée : '{text}' ({proba_wake:.2f})")
            return True

        print(f"🤐 Ignoré : '{text}' ({proba_wake:.2f})")
        return False


# ---------------------------------------------------------------------------
# 4. Démo interactive
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(os.path.join(OUT_DIR, "config.json")):
        train_wake_model()

    detector = WakeDetector(threshold=0.6)

    # Quelques exemples de validation rapide avant de passer en interactif
    sanity_checks = [
        ("James, t'es là ?", True),
        ("James, tu peux allumer la lumière ?", True),
        ("Je parlais à James tout à l'heure", False),
        ("Non c'était James", False),
        ("Salut, ça va ?", False),
    ]
    print("\n🔍 Vérification rapide :")
    for text, expected in sanity_checks:
        p = detector.predict_proba(text)
        detector.is_active = False  # reset entre chaque check
        verdict = "✅" if (p > detector.threshold) == expected else "⚠️"
        print(f"  {verdict} '{text}' -> proba={p:.2f} (attendu wake={expected})")

    detector.is_active = False
    print("\n🎙️ Parle à ton IA (Ctrl+C pour quitter) :")
    while True:
        try:
            text = input("> ").strip()
            if text:
                detector.process(text)
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break