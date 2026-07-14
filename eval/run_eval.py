"""
Week 4: Runs the hand-labeled eval set through LumosAI's retrieval + generation
pipeline, then scores the results using RAGAS.

Metrics computed:
  - faithfulness: does the answer stay grounded in the retrieved context,
                   or does it say things not actually supported by it?
  - answer_relevancy: does the answer actually address the question asked?
  - context_precision: of the chunks retrieved, how many were actually useful?
  - context_recall: did retrieval surface what was needed to answer fully?

Usage:
    python eval/run_eval.py
"""

import sys
import os
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

from retrieval.hybrid_search import HybridRetriever
from eval.eval_set import EVAL_SET

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GEN_MODEL = "llama-3.1-8b-instant"  # was llama-3.3-70b-versatile
TOP_K = 5

SYSTEM_PROMPT = """You are LumosAI, a codebase Q&A assistant. Answer the user's \
question using ONLY the provided code context below. Always cite the specific \
file and line numbers you used in your answer. If the context doesn't contain \
enough information to answer confidently, say so explicitly instead of guessing."""


def build_context_block(chunks) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"--- {chunk.citation()} ---\n{chunk.content}\n")
    return "\n".join(parts)

def generate_with_retry(messages, max_retries=5):
    """Calls Groq with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GEN_MODEL,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            wait_seconds = 15 * (attempt + 1)
            print(f"    Rate limited, waiting {wait_seconds}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)
    raise RuntimeError("Exceeded max retries due to persistent rate limiting.")

def run_pipeline_on_eval_set():
    """Runs every eval question through retrieval + generation, collecting
    everything RAGAS needs: question, generated answer, retrieved contexts
    (as plain strings), and our hand-written ground truth.
    """
    retriever = HybridRetriever()

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_SET, 1):
        question = item["question"]
        print(f"[{i}/{len(EVAL_SET)}] {question}")

        chunks = retriever.search(question, top_k=TOP_K)
        context_texts = [f"{c.citation()}\n{c.content}" for c in chunks]
        context_block = build_context_block(chunks)

        response = generate_with_retry(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context from the codebase:\n\n{context_block}\n\nQuestion: {question}",
                },
            ]
        )
        answer = response

        questions.append(question)
        answers.append(answer)
        contexts_list.append(context_texts)
        ground_truths.append(item["ground_truth"])

        time.sleep(8)  # small pause between questions to stay under the TPM limit

    retriever.close()

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )


def run_ragas_eval(dataset: Dataset):
    """Scores the dataset using RAGAS, with Groq as the judge LLM (instead
    of the default OpenAI) and a local sentence-transformers model for the
    embedding-based parts of the metrics.
    """
    judge_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    run_config = RunConfig(max_workers=1, timeout=300, max_retries=15, max_wait=30)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    return results


if __name__ == "__main__":
    print("Running pipeline on eval set...\n")
    dataset = run_pipeline_on_eval_set()

    print("\nRunning RAGAS evaluation (this calls the LLM several more times, be patient)...\n")
    results = run_ragas_eval(dataset)

    print("\n=== RAGAS Results ===")
    print(results)

    # Save raw results for later comparison (e.g. before/after chunking changes)
    results_df = results.to_pandas()
    os.makedirs("eval/results", exist_ok=True)
    results_df.to_csv("eval/results/eval_results.csv", index=False)
    print("\nSaved detailed per-question results to eval/results/eval_results.csv")