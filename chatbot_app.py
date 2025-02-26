# chatbot_app.py
import streamlit as st
import numpy as np
import random
import json
import pickle
import nltk
import time
from tensorflow.keras.models import load_model  
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.stem import WordNetLemmatizer
from langchain_groq import ChatGroq
import os


# Définir un dossier personnalisé pour les ressources NLTK
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Télécharger les ressources NLTK si elles ne sont pas déjà installées
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)

# Initialisation de Streamlit
st.title("🤖 Chatbot Expert Automobile")
st.markdown("Posez-moi vos questions sur les problèmes mécaniques!")



# Chargement des modèles et données
@st.cache_resource
def load_components():
    # Modèle Keras
    model = load_model('chatbot_model.h5')
    
    # Données d'intentions
    intents = json.load(open('dataset_chatbot.json', encoding='utf-8'))
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    

    llm_model = ChatGroq(
        temperature=0.3,
        model_name="mixtral-8x7b-32768",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return model, intents, words, classes, llm_model

model, intents, words, classes, llm_model = load_components()
lemmatizer = WordNetLemmatizer()

# Gestion de l'historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fonctions du chatbot (adaptées du notebook)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.75]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def chatbot_response(msg):
    ints = predict_class(msg)
    if ints and float(ints[0]['probability']) > 0.7:
        for intent in intents['intents']:
            if intent['tag'] == ints[0]['intent']:
                prompt = f"""Tu es un assistant expert en automobile. Voici le contexte des derniers échanges :
                        {random.choice(intent['responses'])},
                        {st.session_state.messages[-5:]}

                        Nouvelle question : {msg}

                        Réponds en t'appuyant sur le contexte fourni. Si tu ne peux pas répondre précisément ou si la question dépasse tes compétences, indique : "Je ne sais pas." :"""
                response= llm_model.invoke(prompt)
                if isinstance(response, dict) and 'content' in response:
                            return "OK"
                return response.content
            
    else:
        # Générer une réponse contextuelle avec le LLM
        prompt = f"""Contexte de la conversation :
        {st.session_state.messages[-10:]}

        Nouvelle question : {msg}
        
        En tant qu'expert automobile, fournissez une réponse précise et utile si la question est en rapport avec l'automobile , une salutation ou des information sur moi .
        sinon repond par je ne sait pas .

        :"""
        response = llm_model.invoke(prompt)
        if isinstance(response, dict) and 'content' in response:
                    return "OK"
        return response.content

# Interface utilisateur
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["question"])

if prompt := st.chat_input("Tapez votre message..."):
    st.session_state.messages.append({"role": "user", "question": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Le chatbot réfléchit..."):
        response = chatbot_response(prompt)
    
    with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for char in response:
                full_response += char
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "question": response})
