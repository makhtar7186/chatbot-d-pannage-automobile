"""
train_model.py — Reconstruction complète du modèle AutoExpert depuis zéro
==========================================================================
Ce script :
  1. Charge le dataset enrichi (dataset_chatbot.json)
  2. Prétraite le texte (tokenisation, lemmatisation, bag-of-words)
  3. Construit et entraîne un réseau de neurones profond
  4. Sauvegarde le modèle (.h5), les mots (words.pkl) et les classes (classes.pkl)
  5. Affiche les métriques d'entraînement

Usage :
    python train_model.py
    python train_model.py --dataset mon_dataset.json --epochs 300
"""

import json
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer

# ─── TensorFlow / Keras ────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# ─── Reproductibilité ─────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── NLTK ─────────────────────────────────────────────────────────────────
import os
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
for res, pkg in [('tokenizers/punkt', 'punkt'),
                  ('tokenizers/punkt_tab', 'punkt_tab'),
                  ('corpora/wordnet', 'wordnet')]:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)

lemmatizer = WordNetLemmatizer()

# ─── Fonctions de prétraitement ────────────────────────────────────────────
IGNORE_CHARS = set('!?.,;:"\'-()[]{}')

def tokenize_and_lemmatize(sentence: str) -> list[str]:
    """Tokenise et lemmatise une phrase."""
    tokens = nltk.word_tokenize(sentence.lower())
    return [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in IGNORE_CHARS and len(w) > 1
    ]

def build_vocabulary(intents: dict) -> tuple[list, list, list]:
    """Construit words, classes, et documents depuis les intents."""
    words, classes, documents = [], [], []

    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in classes:
            classes.append(tag)

        for pattern in intent['patterns']:
            tokens = tokenize_and_lemmatize(pattern)
            words.extend(tokens)
            documents.append((tokens, tag))

    words = sorted(set(words))
    classes = sorted(classes)
    return words, classes, documents

def create_training_data(
    words: list, classes: list, documents: list
) -> tuple[np.ndarray, np.ndarray]:
    """Crée les matrices X (bag-of-words) et y (one-hot)."""
    training = []

    for tokens, tag in documents:
        bag = [1 if w in tokens else 0 for w in words]
        label = [1 if c == tag else 0 for c in classes]
        training.append((bag, label))

    # Augmentation : dupliquer chaque exemple 2-3 fois avec légère variation
    augmented = []
    for bag, label in training:
        augmented.append((bag, label))
        # Version avec bruit (flip aléatoire de quelques bits à 0)
        noisy_bag = bag.copy()
        for i in range(len(noisy_bag)):
            if noisy_bag[i] == 0 and random.random() < 0.03:
                noisy_bag[i] = 1
        augmented.append((noisy_bag, label))

    random.shuffle(augmented)
    X = np.array([a[0] for a in augmented], dtype=np.float32)
    y = np.array([a[1] for a in augmented], dtype=np.float32)
    return X, y

# ─── Architecture du modèle ────────────────────────────────────────────────
def build_model(input_size: int, output_size: int) -> tf.keras.Model:
    """
    Réseau de neurones profond avec :
    - BatchNormalization pour une convergence plus stable
    - Dropout progressif pour régulariser
    - Activation GELU (meilleure que ReLU pour le NLP)
    """
    model = Sequential([
        Input(shape=(input_size,)),

        Dense(256, activation='gelu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='gelu'),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64, activation='gelu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(output_size, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

# ─── Visualisation ────────────────────────────────────────────────────────
def plot_training_history(history, output_path: str = "training_history.png"):
    """Génère les courbes d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'entraînement — AutoExpert Model", fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', color='#ff6b35', linewidth=2)
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Val Loss', color='#4ecdc4',
                     linewidth=2, linestyle='--')
    axes[0].set_title('Perte (Loss)')
    axes[0].set_xlabel('Époques')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy',
                 color='#ff6b35', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy',
                     color='#4ecdc4', linewidth=2, linestyle='--')
    axes[1].set_title('Précision (Accuracy)')
    axes[1].set_xlabel('Époques')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Courbes sauvegardées → {output_path}")
    plt.close()

# ─── Évaluation ───────────────────────────────────────────────────────────
def evaluate_model(model, words, classes, intents):
    """Teste le modèle sur quelques phrases et affiche les résultats."""
    test_phrases = [
        ("Mes freins font un bruit bizarre", "brake_issue"),
        ("Voyant moteur allumé que faire", "engine_light"),
        ("Quelle pression pour mes pneus", "tire_pressure"),
        ("Bonjour", "greeting"),
        ("Au revoir merci", "goodbye"),
        ("Ma batterie est morte", "battery_issue"),
        ("Comment préparer ma voiture pour l'hiver", "winter_driving"),
        ("Avantages voiture électrique", "electric_vehicle"),
    ]

    print("\n" + "═" * 60)
    print("  ÉVALUATION SUR PHRASES TEST")
    print("═" * 60)
    correct = 0
    for phrase, expected_tag in test_phrases:
        tokens = tokenize_and_lemmatize(phrase)
        bow = np.array([[1 if w in tokens else 0 for w in words]], dtype=np.float32)
        pred = model.predict(bow, verbose=0)[0]
        predicted_tag = classes[np.argmax(pred)]
        confidence = pred.max()
        status = "✅" if predicted_tag == expected_tag else "❌"
        if predicted_tag == expected_tag:
            correct += 1
        print(f"  {status} '{phrase}'")
        print(f"     → Prédit: {predicted_tag} ({confidence:.1%}) | Attendu: {expected_tag}")
    print(f"\n  Score : {correct}/{len(test_phrases)} ({correct/len(test_phrases):.0%})")
    print("═" * 60)

# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Entraîne le modèle AutoExpert")
    parser.add_argument('--dataset', default='dataset_chatbot.json', help='Chemin du dataset')
    parser.add_argument('--epochs', type=int, default=300, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=8, help='Taille de batch')
    parser.add_argument('--validation_split', type=float, default=0.15,
                        help='Fraction des données pour la validation')
    parser.add_argument('--output_dir', default='.', help='Dossier de sortie des artefacts')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Chargement du dataset ────────────────────────────────────────────
    print(f"\n📂 Chargement du dataset : {args.dataset}")
    with open(args.dataset, encoding='utf-8') as f:
        intents = json.load(f)

    n_intents = len(intents['intents'])
    n_patterns = sum(len(i['patterns']) for i in intents['intents'])
    print(f"   • {n_intents} intentions, {n_patterns} patterns")

    # ── 2. Prétraitement ────────────────────────────────────────────────────
    print("\n🔧 Prétraitement du texte...")
    words, classes, documents = build_vocabulary(intents)
    print(f"   • Vocabulaire : {len(words)} mots")
    print(f"   • Classes : {len(classes)} intentions")
    print(f"   • Documents : {len(documents)} exemples")

    # ── 3. Données d'entraînement ───────────────────────────────────────────
    print("\n📊 Création des données d'entraînement (avec augmentation)...")
    X, y = create_training_data(words, classes, documents)
    print(f"   • X shape : {X.shape}")
    print(f"   • y shape : {y.shape}")

    # ── 4. Construction du modèle ───────────────────────────────────────────
    print("\n🏗️  Construction du modèle...")
    model = build_model(len(words), len(classes))
    model.summary()

    # ── 5. Callbacks ────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if args.validation_split > 0 else 'loss',
            patience=40,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if args.validation_split > 0 else 'loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(output_dir / 'chatbot_model_best.h5'),
            monitor='val_accuracy' if args.validation_split > 0 else 'accuracy',
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── 6. Entraînement ─────────────────────────────────────────────────────
    print(f"\n🚀 Entraînement ({args.epochs} époques max, batch={args.batch_size})...")
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split if args.validation_split > 0 else 0.0,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    final_acc = history.history['accuracy'][-1]
    print(f"\n✅ Entraînement terminé — Accuracy finale : {final_acc:.2%}")

    # ── 7. Sauvegarde des artefacts ─────────────────────────────────────────
    print("\n💾 Sauvegarde des artefacts...")

    model_path = output_dir / 'chatbot_model.h5'
    model.save(str(model_path))
    print(f"   • Modèle        → {model_path}")

    words_path = output_dir / 'words.pkl'
    with open(words_path, 'wb') as f:
        pickle.dump(words, f)
    print(f"   • Vocabulaire   → {words_path}")

    classes_path = output_dir / 'classes.pkl'
    with open(classes_path, 'wb') as f:
        pickle.dump(classes, f)
    print(f"   • Classes       → {classes_path}")

    # ── 8. Courbes ──────────────────────────────────────────────────────────
    plot_training_history(history, str(output_dir / 'training_history.png'))

    # ── 9. Évaluation rapide ────────────────────────────────────────────────
    evaluate_model(model, words, classes, intents)

    print("\n🎉 Pipeline complet ! Tous les artefacts sont dans :", output_dir)
    print("   Lancez maintenant : streamlit run chatbot_app.py")

if __name__ == '__main__':
    main()
