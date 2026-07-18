import os
from dotenv import load_dotenv
from openai import OpenAI

from crewai import Agent, Crew, Task, LLM


load_dotenv()


def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment/.env")
    return OpenAI(api_key=api_key)


# This helper uses the exact connection style you shared.
def quick_openai_check() -> str:
    client = build_openai_client()

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are an expert problem solver."
            },
            {
                "role": "user",
                "content": (
                    "If a shop gives a 20% discount on a $ 5000 item, "
                    "what is the final price?\n"
                    "Only provide: Explanation and Final Answer."
                )
            }
        ]
    )

    return response.output_text


def build_crew() -> Crew:
    # CrewAI LLM config for OpenAI gpt-4o-mini
    llm = LLM(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    researcher = Agent(
        role="Research Agent",
        goal="Identify the key facts and assumptions for the user question.",
        backstory="You break down questions into clean, usable facts.",
        llm=llm,
        verbose=True,
    )

    solver = Agent(
        role="Solver Agent",
        goal="Compute the answer from the facts provided by the Research Agent.",
        backstory="You do the math carefully and show concise reasoning.",
        llm=llm,
        verbose=True,
    )

    reviewer = Agent(
        role="Reviewer Agent",
        goal="Check correctness and return a final user-ready response.",
        backstory="You validate logic and produce the final clean answer.",
        llm=llm,
        verbose=True,
    )

    task1 = Task(
        description=(
            "Break down this question into facts and approach: "
            "If a shop gives a 20% discount on a  $ 5000 item, "
            "what is the final price?"
        ),
        expected_output="A short fact list and formula to apply.",
        agent=researcher,
    )

    task2 = Task(
        description=(
            "Using the previous analysis, calculate the final price "
            "step-by-step in 2-3 lines."
        ),
        expected_output="A correct calculation with numeric result.",
        agent=solver,
        context=[task1],
    )

    task3 = Task(
        description=(
            "Review the solution and provide final output in this format:\n"
            "Explanation: <1-2 lines>\n"
            "Final Answer: <amount in ?>"
        ),
        expected_output="Final polished answer with correct amount.",
        agent=reviewer,
        context=[task2],
    )

    return Crew(
        agents=[researcher, solver, reviewer],
        tasks=[task1, task2, task3],
        verbose=True,
    )


def main() -> None:
    print("=== OpenAI direct call check (your style) ===")
    print(quick_openai_check())

    print("\n=== CrewAI multi-agent run (3 agents, gpt-4o-mini) ===")
    crew = build_crew()
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
