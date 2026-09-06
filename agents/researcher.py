from openai import OpenAI


class ResearcherAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(
        self,
        question: str,
        research_plan: str,
        rag_context: str,
        memory_context: str
    ):
        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=(
                "Tu es un agent de recherche scientifique spécialisé dans le vin, "
                "la viticulture et l'œnologie.\n\n"

                "Tu dois suivre le plan fourni par l'orchestrateur.\n\n"

                "RÈGLE DE PRIORITÉ DES SOURCES :\n"
                "1. Les sources documentaires RAG sont prioritaires.\n"
                "2. La mémoire générée peut compléter l'analyse mais elle n'est pas "
                "considérée comme une source scientifique validée.\n"
                "3. Si la mémoire contredit les documents, privilégie les documents.\n"
                "4. Si les documents ne permettent pas de confirmer une affirmation, "
                "indique clairement qu'elle relève de connaissances générales ou "
                "d'une mémoire générée non validée.\n"
                "5. Ne présente jamais une mémoire générée comme une source documentaire.\n\n"

                f"QUESTION UTILISATEUR :\n{question}\n\n"

                f"PLAN DE L'ORCHESTRATEUR :\n"
                f"{research_plan}\n\n"

                f"SOURCES DOCUMENTAIRES RAG :\n"
                f"{rag_context}\n\n"

                f"MÉMOIRE GÉNÉRÉE NON VALIDÉE :\n"
                f"{memory_context}\n"
            )
        )

        return {
            "text": response.output_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }