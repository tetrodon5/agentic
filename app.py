from dotenv import load_dotenv
from openai import OpenAI
import os

from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.communicator import CommunicatorAgent


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

orchestrator = OrchestratorAgent(client)
researcher = ResearcherAgent(client)
critic = CriticAgent(client)
communicator = CommunicatorAgent(client)

question = input("\nPosez votre question scientifique : ")


# 1. Orchestrator
orchestrator_result = orchestrator.run(question)

print("\n=== ORCHESTRATOR AGENT ===")
print(orchestrator_result["text"])


# 2. Researcher
research_result = researcher.run(
    question,
    orchestrator_result["text"]
)

print("\n=== RESEARCHER AGENT ===")
print(research_result["text"])


# 3. Critic
critic_result = critic.run(
    question,
    research_result["text"]
)

print("\n=== CRITIC AGENT ===")
print(critic_result["text"])


# 4. Communicator
communicator_result = communicator.run(
    question,
    research_result["text"],
    critic_result["text"]
)

print("\n=== COMMUNICATOR AGENT ===")
print(communicator_result["text"])


# Consommation totale
total_tokens = (
    orchestrator_result["total_tokens"]
    + research_result["total_tokens"]
    + critic_result["total_tokens"]
    + communicator_result["total_tokens"]
)

print("\n=== CONSOMMATION ===")
print("Orchestrator :", orchestrator_result["total_tokens"], "tokens")
print("Researcher :", research_result["total_tokens"], "tokens")
print("Critic :", critic_result["total_tokens"], "tokens")
print("Communicator :", communicator_result["total_tokens"], "tokens")
print("Total du run :", total_tokens, "tokens")