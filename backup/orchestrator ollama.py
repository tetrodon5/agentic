import requests


class OrchestratorAgent:

    def __init__(self, client=None):
        self.client = client

    def run(self, question: str):

        prompt = (
            "Tu es l'orchestrateur d'un écosystème scientifique spécialisé "
            "dans le vin, la viticulture et l'œnologie.\n\n"
            "Ta mission est de préparer un plan de traitement très compact "
            "pour les agents suivants.\n\n"
            "FORMAT OBLIGATOIRE :\n"
            "RESEARCH: 3 à 5 points maximum\n"
            "CRITIC: 2 à 4 points de vigilance maximum\n"
            "AUDIENCE: une ligne\n\n"
            "Pas d'introduction, pas de conclusion, pas de développement.\n\n"
            f"QUESTION :\n{question}"
        )

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        text = data.get(
            "response",
            ""
        )

        input_tokens = data.get(
            "prompt_eval_count",
            0
        )

        output_tokens = data.get(
            "eval_count",
            0
        )

        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (
                input_tokens
                + output_tokens
            ),
        }