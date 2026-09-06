from openai import OpenAI


class CommunicatorAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, question: str, researcher_answer: str, critic_answer: str):
        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=(
                "Tu es un agent de médiation scientifique pour la Cité du Vin. "
                "À partir de la recherche et de la critique scientifique ci-dessous, "
                "produis une réponse finale claire, fiable, pédagogique et nuancée. "
                "Corrige les points signalés par le critique et évite les affirmations "
                "trop certaines lorsqu'elles ne sont pas suffisamment établies.\n\n"
                f"Question : {question}\n\n"
                f"Recherche :\n{researcher_answer}\n\n"
                f"Critique :\n{critic_answer}"
            )
        )

        return {
            "text": response.output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }