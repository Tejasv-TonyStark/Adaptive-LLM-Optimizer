# Execution/Bedrock_client.py

import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────
# MODEL IDs
# nova-micro → Amazon Nova Micro  (1B,  fast, cheap)
# llama3-8b  → Llama 3.1 8B      (8B,  balanced)
# haiku      → Llama 3.3 70B     (70B, powerful, complex)
# ──────────────────────────────────────────

MODEL_IDS = {
    "nova-micro": "amazon.nova-micro-v1:0",
    "llama3-8b":  "us.meta.llama3-1-8b-instruct-v1:0",
    "haiku":      "us.meta.llama3-3-70b-instruct-v1:0"
}

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION             = os.getenv("AWS_REGION", "us-east-1")


def get_bedrock_client():
    """Returns a Bedrock runtime client using credentials from .env"""
    return boto3.client(
        "bedrock-runtime",
        region_name          = REGION,
        aws_access_key_id    = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key= os.getenv("AWS_SECRET_ACCESS_KEY")
    )


def invoke_nova(client, prompt: str) -> dict:
    """
    Calls Amazon Nova Micro.
    Used for: fast mode, low complexity queries.

    Returns:
        dict with keys: text, input_tokens, output_tokens
        Nova includes usage metadata in every response — always exact counts.
    """
    body = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ]
    }

    response = client.invoke_model(
        modelId     = MODEL_IDS["nova-micro"],
        body        = json.dumps(body),
        contentType = "application/json",
        accept      = "application/json"
    )

    result = json.loads(response["body"].read())

    text         = result["output"]["message"]["content"][0]["text"]
    # Nova always returns usage — safe to access directly
    input_tokens  = result.get("usage", {}).get("inputTokens")
    output_tokens = result.get("usage", {}).get("outputTokens")

    return {
        "text":          text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }


def invoke_llama(client, prompt: str, model_key: str = "llama3-8b") -> dict:
    """
    Calls Llama models — both 8B and 70B use the same request format.
    Used for: reasoning (8B) and complex/high queries (70B).

    Returns:
        dict with keys: text, input_tokens, output_tokens
        Llama returns prompt_token_count + generation_token_count.
    """
    body = {
        "prompt":      prompt,
        "max_gen_len": 512,
        "temperature": 0.7
    }

    response = client.invoke_model(
        modelId     = MODEL_IDS[model_key],
        body        = json.dumps(body),
        contentType = "application/json",
        accept      = "application/json"
    )

    result = json.loads(response["body"].read())

    text          = result["generation"]
    # Llama returns separate token count fields
    input_tokens  = result.get("prompt_token_count")
    output_tokens = result.get("generation_token_count")

    return {
        "text":          text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }


def invoke_model(model: str, prompt: str) -> dict:
    """
    Master routing function — calls the correct model.

    Mapping:
        nova-micro → Amazon Nova Micro  (1B)
        llama3-8b  → Llama 3.1 8B      (8B)
        haiku      → Llama 3.3 70B     (70B)

    Args:
        model:  nova-micro / llama3-8b / haiku
        prompt: full prompt string

    Returns:
        dict — { text, input_tokens, output_tokens }
        input_tokens / output_tokens are None if model didn't return them.
    """
    client = get_bedrock_client()

    if model == "nova-micro":
        return invoke_nova(client, prompt)
    elif model == "llama3-8b":
        return invoke_llama(client, prompt, model_key="llama3-8b")
    elif model == "haiku":
        return invoke_llama(client, prompt, model_key="haiku")
    else:
        return invoke_llama(client, prompt, model_key="llama3-8b")


def get_embedding(text: str) -> list[float]:
    """
    Generates embeddings using Titan Text Embeddings V2.
    Used by RAG system for document and query vectorization.

    Returns:
        list of floats — 1024 dimension embedding vector
    """
    client = get_bedrock_client()

    body = {"inputText": text}

    response = client.invoke_model(
        modelId     = EMBEDDING_MODEL_ID,
        body        = json.dumps(body),
        contentType = "application/json",
        accept      = "application/json"
    )

    result = json.loads(response["body"].read())
    return result["embedding"]


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = "What is Python? Answer in 2 sentences."

    print("\n── Bedrock Client Test ──\n")

    for model in ["nova-micro", "llama3-8b", "haiku"]:
        print(f"Testing {model}...")
        try:
            result = invoke_model(model, test_prompt)
            print(f"✅ {model}: {result['text'][:100]}...")
            print(f"   Tokens — input: {result['input_tokens']}, output: {result['output_tokens']}")
        except Exception as e:
            print(f"❌ {model} failed: {e}")
        print()

    print("Testing embeddings...")
    try:
        embedding = get_embedding("What is Python?")
        print(f"✅ Embedding generated — vector size: {len(embedding)}")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
