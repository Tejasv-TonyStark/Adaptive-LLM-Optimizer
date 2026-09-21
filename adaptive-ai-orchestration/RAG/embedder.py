from Execution.Bedrock_client import get_embedding
def embed_chunks(chunks):
    # Fail the build rather than silently publish an incomplete corpus.
    return [{**chunk, "embedding": get_embedding(chunk["text"])} for chunk in chunks]
def embed_query(query):
    return get_embedding(query)
