# chatbot_app.py — Version améliorée (NVIDIA NIM API)
import streamlit as st
import numpy as np
import random
import json
import pickle
import nltk
import time
import os
import logging
from datetime import datetime
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

# ─── Configuration du logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Configuration de la page Streamlit ─────────────────────────────────────
st.set_page_config(
    page_title="AutoExpert — Assistant Automobile",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Fond principal — forcer le dark sur TOUS les conteneurs Streamlit ── */
    .stApp,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="block-container"],
    .main .block-container {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
        color: #e8e8f0 !important;
    }

    /* ── Bulles de chat utilisateur ── */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"][data-role="user"] {
        background: transparent !important;
    }

    /* Avatar utilisateur — fond orange */
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #ff6b35, #ff4500) !important;
        border: none !important;
    }

    /* Conteneur texte du message utilisateur */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) 
    [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%) !important;
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 10px 16px !important;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.35) !important;
        display: inline-block !important;
        max-width: 85% !important;
    }

    /* ── Bulles de chat assistant ── */
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #1e2a4a, #243050) !important;
        border: 1px solid #ff6b35 !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stMarkdownContainer"] p {
        color: #d8d8f0 !important;
    }

    /* Fond de la zone de chat */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(90deg, #ff6b35, #f7c59f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #8888aa;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0d0d1f 0%, #1a1a2e 100%) !important;
        border-right: 1px solid #2a2a4a !important;
    }

    /* ── Stats cards ── */
    .stat-card {
        background: #1e2a4a;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #ff6b35;
    }

    .stat-number {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        color: #ff6b35;
        font-weight: 700;
    }

    .stat-label {
        font-size: 0.78rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Intent tags ── */
    .intent-tag {
        background: rgba(255, 107, 53, 0.15);
        color: #ff8c42;
        border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-family: 'Space Mono', monospace;
        margin-left: 8px;
    }

    /* ── Boutons sidebar ── */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b35, #ff4500) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4) !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: #1a1a2e !important;
        color: #e8e8f0 !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #ff6b35 !important;
        box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.2) !important;
    }

    /* ── Slider & checkbox ── */
    .stSlider [data-testid="stTickBar"] { background: #ff6b35 !important; }

    /* ── Confidence badges ── */
    .confidence-high { color: #4ecdc4; }
    .confidence-low  { color: #ff8c42; }

    /* ── Timestamp ── */
    .msg-timestamp {
        font-size: 0.68rem;
        color: #555577;
        margin-top: 4px;
        font-family: 'Space Mono', monospace;
    }

    /* ── Masquer footer Streamlit ── */
    footer { visibility: hidden; }

    /* ── Texte général dans la sidebar ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small {
        color: #c8c8e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── NLTK Setup ──────────────────────────────────────────────────────────────
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for resource in [('tokenizers/punkt', 'punkt'), ('corpora/wordnet', 'wordnet'), ('tokenizers/punkt_tab', 'punkt_tab')]:
    try:
        nltk.data.find(resource[0])
    except LookupError:
        nltk.download(resource[1], download_dir=nltk_data_dir, quiet=True)

# ─── Chargement des composants ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du moteur AutoExpert...")
def load_components():
    try:
        from keras.models import load_model
        from nltk.stem import WordNetLemmatizer

        model = load_model('chatbot_model.h5')
        intents = json.load(open('dataset_chatbot.json', encoding='utf-8'))
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            st.sidebar.warning("⚠️ NVIDIA_API_KEY manquante — mode local activé")

        nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key or "missing",
        ) if api_key else None

        lemmatizer = WordNetLemmatizer()
        return model, intents, words, classes, nvidia_client, lemmatizer

    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        st.error(f"❌ Erreur de chargement : {e}")
        st.stop()

model, intents, words, classes, nvidia_client, lemmatizer = load_components()

# ─── Session State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "intents_detected" not in st.session_state:
    st.session_state.intents_detected = {}
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "last_active_tag" not in st.session_state:
    st.session_state.last_active_tag = None
if "used_causes" not in st.session_state:
    st.session_state.used_causes = {}
if "used_causes_flat" not in st.session_state:
    st.session_state.used_causes_flat = []

# Fenêtre glissante : nb max de tours (1 tour = 1 user + 1 assistant)
MEMORY_WINDOW = 8

# ─── Fonctions du chatbot ─────────────────────────────────────────────────────
def clean_up_sentence(sentence: str) -> list[str]:
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence: str) -> np.ndarray:
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1.0
    return bag

def predict_class(sentence: str, threshold: float = 0.65) -> list[dict]:
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [
        {"intent": classes[i], "probability": float(r)}
        for i, r in enumerate(res)
        if r > threshold
    ]
    return sorted(results, key=lambda x: x["probability"], reverse=True)


def get_intent_response(tag: str) -> str:
    """Retourne une réponse de base aléatoire pour un tag donné."""
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return ""

def get_all_intent_responses(tag: str) -> list[str]:
    """Retourne TOUTES les réponses disponibles pour un tag."""
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return intent['responses']
    return []

def get_already_used_causes() -> list[str]:
    """
    Récupère les causes déjà fournies par le modèle dans cette session.
    Stockées dans session_state.used_causes par tag.
    """
    return st.session_state.get("used_causes_flat", [])

def register_cause(tag: str, cause: str):
    """Enregistre une cause comme déjà utilisée pour ce tag."""
    if "used_causes" not in st.session_state:
        st.session_state.used_causes = {}
    if "used_causes_flat" not in st.session_state:
        st.session_state.used_causes_flat = []
    if tag not in st.session_state.used_causes:
        st.session_state.used_causes[tag] = []
    st.session_state.used_causes[tag].append(cause)
    st.session_state.used_causes_flat.append(cause)

def pick_new_cause(tag: str) -> str | None:
    """
    Choisit une cause que le modèle n'a pas encore proposée pour ce tag.
    Retourne None si toutes les causes ont été épuisées.
    """
    all_causes = get_all_intent_responses(tag)
    used = st.session_state.get("used_causes", {}).get(tag, [])
    remaining = [c for c in all_causes if c not in used]
    if not remaining:
        return None  # toutes les causes épuisées
    chosen = random.choice(remaining)
    register_cause(tag, chosen)
    return chosen

def build_llm_history(cause_hint: str = "", is_followup: bool = False) -> list[dict]:
    """
    Construit la liste de messages envoyée à l'API NVIDIA.
    Le system prompt guide le LLM pour qu'il réponde de façon naturelle
    en intégrant ce que l'utilisateur vient de dire ET la cause suggérée.
    """

    if is_followup:
        system_content = (
            "Tu es AutoExpert, un assistant chaleureux et expert en mécanique automobile. "
            "Tu parles en français, de façon naturelle et fluide, comme un ami mécanicien compétent.\n\n"
            "RÈGLES ABSOLUES :\n"
            "• Lis attentivement CE QUE L'UTILISATEUR VIENT DE DIRE et commence ta réponse "
            "en y faisant référence explicitement (ex: 'Puisque vous n\\'entendez pas de grincement...', "
            "'Vu que vous avez déjà changé le liquide...', 'Si la pédale s\\'enfonce doucement...').\n"
            "• Ne répète JAMAIS mot pour mot la cause technique — intègre-la dans une phrase naturelle.\n"
            "• Enchaîne logiquement depuis la réponse précédente : tu enquêtes avec l'utilisateur, "
            "tu élimines des pistes, tu converges vers la solution.\n"
            "• 3 à 5 phrases maximum. Ton conversationnel, pas de liste à puces.\n"
            "• Termine si possible par une question d'investigation ou une action concrète."
        )
    else:
        system_content = (
            "Tu es AutoExpert, un assistant chaleureux et expert en mécanique automobile. "
            "Tu parles en français, de façon naturelle et fluide, comme un ami mécanicien compétent.\n\n"
            "RÈGLES ABSOLUES :\n"
            "• Reformule la cause technique de façon humaine et accessible — pas de jargon brut.\n"
            "• Commence par situer le problème avant de proposer la cause "
            "(ex: 'Ce type de problème vient souvent de...', 'La première chose à vérifier, c\\'est...').\n"
            "• 3 à 5 phrases maximum. Ton conversationnel, pas de liste à puces.\n"
            "• Termine par une question pour mieux cerner le problème ou confirmer la piste."
        )

    if cause_hint:
        system_content += (
            f"\n\n--- PISTE TECHNIQUE À EXPLORER ---\n"
            f"{cause_hint}\n"
            f"Utilise cette piste comme fil conducteur de ta réponse, "
            f"mais exprime-la de façon naturelle en tenant compte du contexte de la conversation."
            f"\n--- FIN DE LA PISTE ---"
        )

    windowed = st.session_state.conversation_history[-(MEMORY_WINDOW * 2):]
    return [{"role": "system", "content": system_content}] + windowed


def call_llm(history: list[dict]) -> str:
    """Appel direct à l'API NVIDIA NIM avec historique structuré."""
    response = nvidia_client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=history,
        temperature=0.75,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
#
#  Tour 1 — Nouveau problème :
#    Modèle → cause #1 → LLM présente la piste naturellement + pose une question
#
#  Tour 2+ — Message de suivi (l'utilisateur précise, contredit, confirme) :
#    Modèle → cause #2 (différente) → LLM commence par "Puisque vous dites que..."
#    en s'appuyant sur ce que l'utilisateur vient de dire, puis explore la nouvelle piste
#
#  Aucune intention connue :
#    LLM répond librement avec l'historique (salutations, hors-sujet poli…)
#
# ──────────────────────────────────────────────────────────────────────────────

def chatbot_response(msg: str) -> tuple[str, str | None, float | None, str | None]:
    """
    Retourne (réponse, tag, confiance, cause_utilisée).
    """
    if not msg.strip():
        return (
            "Je n'ai pas compris votre message. Pouvez-vous reformuler ?",
            None, None, None
        )

    # ── 1. Prédiction du modèle Keras ────────────────────────────────────────
    predictions  = predict_class(msg, threshold=0.60)
    current_tag  = predictions[0]["intent"]      if predictions else None
    current_conf = predictions[0]["probability"] if predictions else None
    last_tag     = st.session_state.get("last_active_tag")
    active_tag   = current_tag or last_tag

    # Est-ce un message de suivi sur le même sujet ?
    is_followup = (last_tag is not None) and (current_tag == last_tag or current_tag is None)

    # ── 2. Ajout du message utilisateur à l'historique LLM ───────────────────
    st.session_state.conversation_history.append({
        "role": "user",
        "content": msg,
    })

    # ── 3. Sélection de la cause à explorer ──────────────────────────────────
    cause_used = None

    if active_tag:
        st.session_state.intents_detected[active_tag] = (
            st.session_state.intents_detected.get(active_tag, 0) + 1
        )

        if not is_followup:
            # Nouveau sujet : réinitialiser les causes pour ce tag
            if "used_causes" not in st.session_state:
                st.session_state.used_causes = {}
            st.session_state.used_causes[active_tag] = []

        cause_used = pick_new_cause(active_tag)
        if cause_used is None:
            cause_used = (
                "Toutes les causes courantes ont été explorées. "
                "Recommande chaleureusement une visite chez un professionnel "
                "pour un diagnostic approfondi avec les outils adéquats."
            )

        st.session_state.last_active_tag = active_tag

    # ── 4. Appel LLM ─────────────────────────────────────────────────────────
    if nvidia_client:
        history = build_llm_history(cause_hint=cause_used or "", is_followup=is_followup)
        try:
            answer = call_llm(history)
        except Exception as e:
            logger.error(f"NVIDIA API error: {e}")
            answer = cause_used or "Je rencontre une difficulté technique. Veuillez réessayer."
    else:
        answer = cause_used or (
            "Je suis spécialisé en automobile. "
            "Posez-moi une question sur la mécanique ou l'entretien !"
        )

    # ── 5. Sauvegarder la réponse ─────────────────────────────────────────────
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": answer,
    })

    return answer, active_tag, current_conf, cause_used

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 AutoExpert")
    st.markdown("<span style='color:#8888aa;font-size:0.85rem'>Assistant Automobile IA</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.total_questions}</div>
            <div class="stat-label">Questions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        duration = (datetime.now() - st.session_state.session_start).seconds // 60
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{duration}m</div>
            <div class="stat-label">Durée</div>
        </div>""", unsafe_allow_html=True)

    # Indicateur mémoire
    mem_turns = len(st.session_state.conversation_history) // 2
    mem_max   = MEMORY_WINDOW
    mem_pct   = min(mem_turns / mem_max, 1.0)
    mem_color = "#4ecdc4" if mem_pct < 0.75 else "#ff8c42"
    st.markdown(f"""
    <div class="stat-card" style="margin-top:4px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span class="stat-label">🧠 Mémoire active</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.8rem;color:{mem_color}">
                {mem_turns}/{mem_max} tours
            </span>
        </div>
        <div style="background:#0f0f1a;border-radius:4px;height:6px;margin-top:6px;overflow:hidden">
            <div style="width:{mem_pct*100:.0f}%;height:100%;background:{mem_color};
                        border-radius:4px;transition:width 0.4s ease"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Top intents
    if st.session_state.intents_detected:
        st.markdown("### 🏷️ Sujets abordés")
        sorted_intents = sorted(st.session_state.intents_detected.items(), key=lambda x: x[1], reverse=True)[:5]
        for tag, count in sorted_intents:
            label = tag.replace("_", " ").title()
            st.markdown(f"<small>• **{label}** — {count}x</small>", unsafe_allow_html=True)

    st.markdown("---")

    # Suggestions
    st.markdown("### 💡 Questions fréquentes")
    suggestions = [
        "Comment entretenir mes freins ?",
        "Voyant moteur allumé, que faire ?",
        "Pression idéale des pneus ?",
        "Changer l'huile moteur",
        "Voiture qui ne démarre pas",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state.pending_input = s
            st.rerun()

    st.markdown("---")

    # Réglages
    st.markdown("### ⚙️ Réglages")
    confidence_threshold = st.slider("Seuil de confiance", 0.5, 0.95, 0.65, 0.05,
                                      help="Confiance minimale pour utiliser une intention détectée")
    show_debug = st.checkbox("Afficher les métadonnées", value=False)

    st.markdown("---")
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.total_questions = 0
        st.session_state.intents_detected = {}
        st.session_state.session_start = datetime.now()
        st.session_state.last_active_tag = None
        st.session_state.used_causes = {}
        st.session_state.used_causes_flat = []
        st.rerun()

# ─── Header principal ─────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚗 AutoExpert</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Votre assistant intelligent pour tous vos problèmes automobiles</div>', unsafe_allow_html=True)

# ─── Historique de conversation ───────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        timestamp = message.get("timestamp", "")

        # Badge d'intention si disponible
        if message["role"] == "assistant" and message.get("intent") and show_debug:
            intent_label = message["intent"].replace("_", " ").title()
            confidence = message.get("confidence", 0)
            badge_color = "confidence-high" if confidence > 0.8 else "confidence-low"
            st.markdown(
                f'<span class="intent-tag">🏷️ {intent_label} · '
                f'<span class="{badge_color}">{confidence:.0%}</span></span>',
                unsafe_allow_html=True
            )

        st.markdown(content)
        if timestamp and show_debug:
            st.markdown(f'<div class="msg-timestamp">{timestamp}</div>', unsafe_allow_html=True)

# ─── Gestion de l'input suggéré (sidebar) ─────────────────────────────────────
pending = st.session_state.pop("pending_input", None)

# ─── Input de l'utilisateur ───────────────────────────────────────────────────
user_input = st.chat_input("Posez votre question automobile...") or pending

if user_input:
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp,
    })
    st.session_state.total_questions += 1

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner(""):
            response, detected_intent, confidence, cause_used = chatbot_response(user_input)

        # Affichage animé caractère par caractère
        full_response = ""
        for char in response:
            full_response += char
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.012)
        message_placeholder.markdown(full_response)

        # Badge debug
        if show_debug:
            debug_parts = []
            if detected_intent:
                intent_label = detected_intent.replace("_", " ").title()
                badge_color = "confidence-high" if (confidence or 0) > 0.8 else "confidence-low"
                debug_parts.append(
                    f'🏷️ {intent_label} · <span class="{badge_color}">{(confidence or 0):.0%}</span>'
                )
            if cause_used:
                short_cause = cause_used[:60] + "…" if len(cause_used) > 60 else cause_used
                debug_parts.append(f'🔧 Guide: <em>{short_cause}</em>')
            if debug_parts:
                st.markdown(
                    f'<span class="intent-tag">{" &nbsp;|&nbsp; ".join(debug_parts)}</span>',
                    unsafe_allow_html=True
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "timestamp": datetime.now().strftime("%H:%M"),
        "intent": detected_intent,
        "confidence": confidence,
        "cause": cause_used,
    })

# ─── Message d'accueil (conversation vide) ────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #555577;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🔧</div>
        <div style="font-size: 1.1rem; color: #8888aa; margin-bottom: 0.5rem;">
            Bonjour ! Je suis <strong style="color:#ff8c42">AutoExpert</strong>.
        </div>
        <div style="font-size: 0.9rem;">
            Posez-moi vos questions sur l'entretien, la mécanique ou la conduite.
        </div>
    </div>
    """, unsafe_allow_html=True)