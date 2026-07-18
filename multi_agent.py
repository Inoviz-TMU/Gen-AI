import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# ---- shared external memory (stand-in for a real KV / vector store) ----
memory_store = {}

def write_memory(key, content):
    """Write strategy: store a distilled note, keyed for exact lookup later."""
    memory_store[key] = content

def read_memory(key):
    """Retrieval strategy: exact lookup by topic key."""
    return memory_store.get(key, "")


# ---- Agent 1: Researcher ----
def researcher_agent(topic):
    print(f"[Agent Call] Researcher agent started for topic: {topic}")
    response = client.responses.create(
        model=MODEL,
        instructions=(
            "You are a research agent. Gather 3-4 concise, factual "
            "bullet points on the given topic. Be brief and precise."
        ),
        input=f"Research topic: {topic}",
    )
    findings = response.output_text

    # Summarise-then-save: compress before writing to shared memory
    summary_response = client.responses.create(
        model=MODEL,
        instructions="Compress the following notes into a 2-sentence summary.",
        input=findings,
    )
    write_memory(topic, summary_response.output_text)
    print(f"[Agent Done] Researcher agent completed for topic: {topic}")
    return findings


# ---- Agent 2: Writer ----
def writer_agent(topic):
    print(f"[Agent Call] Writer agent started for topic: {topic}")
    notes = read_memory(topic)  # exact lookup, no re-research needed
    if not notes:
        print(f"[Agent Skip] Writer agent found no notes for topic: {topic}")
        return "No research found in memory for this topic yet."

    response = client.responses.create(
        model=MODEL,
        instructions=(
            "You are a writing agent. Using only the provided research "
            "notes, write a short, polished paragraph for the end user."
        ),
        input=f"Research notes:\n{notes}\n\nWrite the final summary.",
    )
    print(f"[Agent Done] Writer agent completed for topic: {topic}")
    return response.output_text


# ---- orchestration ----
if __name__ == "__main__":
    topic = "Molecular biology concepts"

    print(f"[Orchestrator] Starting workflow for topic: {topic}")
    researcher_agent(topic)              # writes to memory_store
    final_answer = writer_agent(topic)   # reads from memory_store
    print("[Orchestrator] Workflow completed")

    print(final_answer)