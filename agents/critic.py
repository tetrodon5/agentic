from openai import OpenAI


class CriticAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, question: str, researcher_answer: str):
        response = self.client.responses.create(
            model="gpt-5.6-luna" ,
            input=(
                "Tu es un agent critique scientifique spécialisé dans le vin, "
                "la viticulture et l'œnologie. "
                "Ton rôle est de vérifier la solidité scientifique d'une réponse. "
                "Identifie les affirmations trop certaines, les approximations, "
                "les points nécessitant des sources, les contradictions éventuelles "
                "et les nuances importantes manquantes. "
                "Ne réécris pas toute la réponse : produis une critique concise et utile.\n\n"
                f"Question initiale : {question}\n\n"
                f"Réponse du Researcher Agent :\n{researcher_answer}"
            )
        )

        return {
            "text": response.output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }