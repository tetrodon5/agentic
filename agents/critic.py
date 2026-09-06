from openai import OpenAI


class CriticAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(
        self,
        question: str,
        researcher_answer: str,
        rag_context: str = ""
    ):
        rag_prompt = ""
        if rag_context:
            rag_prompt = (
                "\n\nSOURCES DOCUMENTAIRES RAG FOURNIES AU CHERCHEUR :\n"
                f"{rag_context}\n\n"
                "Compare les affirmations de la réponse aux passages fournis. "
                "Signale les contradictions avec ces passages et distingue "
                "une affirmation non étayée d'une affirmation fausse."
            )
        else:
            rag_prompt = (
                "\n\nAucun appui documentaire suffisamment pertinent n'a été trouvé. "
                "Vérifie que la réponse signale cette limite et ne présente pas "
                "les connaissances générales ou la mémoire générée comme "
                "des affirmations étayées par les documents. "
                "L'absence d'appui documentaire ne signifie pas qu'une affirmation est fausse."
            )

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
                f"{rag_prompt}"
            )
        )

        return {
            "text": response.output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
