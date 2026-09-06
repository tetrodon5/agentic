from openai import OpenAI


class OrchestratorAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, question: str):
        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=(
                "Tu es l'orchestrateur scientifique de la Cité du Vin. "
                "Analyse la question et retourne un plan très compact pour les autres agents.\n\n"
                "Format obligatoire :\n"
                "RESEARCH: 3 à 5 points maximum\n"
                "CRITIC: 2 à 4 risques ou points de vigilance maximum\n"
                "AUDIENCE: niveau de vulgarisation en une ligne\n\n"
                "Pas d'introduction, pas de conclusion, pas de développement.\n\n"
                f"Question : {question}"
            )
        )

        return {
            "text": response.output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }