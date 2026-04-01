from openai import OpenAI
import os 
import dotenv
dotenv.load_dotenv()
class LLMService:

    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key= os.getenv("GROQ_API_KEY")
        )

  
    def generate_reponse(self, context, base_response,msg):
        prompt = f"""Tu es AutoExpert, un assistant expert en automobile parlant français.
Contexte des derniers échanges :
{context}

Contexte métier fourni : {base_response}

Question actuelle : {msg}

Instructions :
- Réponds de façon claire, structurée et professionnelle en français.
- Appuie-toi sur le contexte métier ci-dessus pour personnaliser ta réponse.
- Si la question dépasse tes compétences automobiles, dis-le honnêtement.
- Sois concis mais complet (3-5 phrases max si possible).
- N'invente pas de prix ou de spécifications non vérifiées.

Réponse :"""
        
   
        response = self.client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": "Tu es AutoExpert, un assistant expert en automobile parlant français."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return self._parse_response(response.choices[0].message.content.strip())

