from openai import OpenAI


class ResearcherAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, question: str, research_plan: str):
        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=(
                "Tu es un agent de recherche scientifique spécialisé dans le vin, "
                "la viticulture et l'œnologie. "
                "Tu dois suivre le plan de recherche fourni par l'orchestrateur. "
                "Analyse la question et fournis une réponse scientifique concise, "
                "factuelle et prudente.\n\n"
                f"Question : {question}\n\n"
                f"Plan de l'orchestrateur :\n{research_plan}"
            )
        )

        return {
            "text": response.output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }