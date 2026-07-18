import os
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END


load_dotenv()


class AgentState(TypedDict):
    question: str
    research_output: str
    solver_output: str
    final_output: str


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


def build_langchain_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment/.env")

    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    )


def build_chains():
    llm = build_langchain_llm()
    parser = StrOutputParser()

    researcher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Research Agent. Identify key facts and assumptions for the user question. Be concise.",
            ),
            (
                "human",
                "Break down this question into facts and approach: {question}",
            ),
        ]
    )

    solver_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Solver Agent. Compute the answer from the facts provided by the Research Agent. Be concise and accurate.",
            ),
            (
                "human",
                "Question: {question}\n\nResearch analysis:\n{research_output}\n\nCalculate the final price step-by-step in 2-3 lines.",
            ),
        ]
    )

    reviewer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Reviewer Agent. Check correctness and return final user-ready response.",
            ),
            (
                "human",
                "Question: {question}\n\nSolver output:\n{solver_output}\n\nReturn final output exactly in this format:\nExplanation: <1-2 lines>\nFinal Answer: <amount in ?>",
            ),
        ]
    )

    researcher_chain = researcher_prompt | llm | parser
    solver_chain = solver_prompt | llm | parser
    reviewer_chain = reviewer_prompt | llm | parser

    return researcher_chain, solver_chain, reviewer_chain


def build_graph():
    researcher_chain, solver_chain, reviewer_chain = build_chains()

    def researcher_node(state: AgentState) -> AgentState:
        print("[Node Call] Research Agent")
        output = researcher_chain.invoke({"question": state["question"]})
        print("[Node Done] Research Agent")
        return {
            **state,
            "research_output": output,
        }

    def solver_node(state: AgentState) -> AgentState:
        print("[Node Call] Solver Agent")
        output = solver_chain.invoke(
            {
                "question": state["question"],
                "research_output": state["research_output"],
            }
        )
        print("[Node Done] Solver Agent")
        return {
            **state,
            "solver_output": output,
        }

    def reviewer_node(state: AgentState) -> AgentState:
        print("[Node Call] Reviewer Agent")
        output = reviewer_chain.invoke(
            {
                "question": state["question"],
                "solver_output": state["solver_output"],
            }
        )
        print("[Node Done] Reviewer Agent")
        return {
            **state,
            "final_output": output,
        }

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("researcher", researcher_node)
    graph_builder.add_node("solver", solver_node)
    graph_builder.add_node("reviewer", reviewer_node)

    graph_builder.set_entry_point("researcher")
    graph_builder.add_edge("researcher", "solver")
    graph_builder.add_edge("solver", "reviewer")
    graph_builder.add_edge("reviewer", END)

    return graph_builder.compile()


def run_langgraph_multiagent(question: str) -> str:
    graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "research_output": "",
        "solver_output": "",
        "final_output": "",
    }

    final_state = graph.invoke(initial_state)
    return final_state["final_output"]


def main() -> None:
    print("=== OpenAI direct call check (your style) ===")
    print(quick_openai_check())

    question = "If a shop gives a 20% discount on a $ 5000 item, what is the final price?"

    print("\n=== LangGraph multi-agent run (3 nodes, gpt-4o-mini) ===")
    result = run_langgraph_multiagent(question)
    print(result)


if __name__ == "__main__":
    main()
