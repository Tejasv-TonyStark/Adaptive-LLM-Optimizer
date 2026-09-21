"""Run actual routing checks; --live runs the full RAG pipeline, never canned answers."""
from Evaluation.benchmark import main
if __name__ == "__main__":
    main(default_suite="rag")
