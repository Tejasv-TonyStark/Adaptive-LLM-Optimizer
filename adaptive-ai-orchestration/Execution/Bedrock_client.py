"""Bedrock adapters with bounded SDK timeouts and explicit generation settings."""
import json
import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
MODEL_IDS = {
    "nova-micro": "amazon.nova-micro-v1:0",
    "llama3-8b": "us.meta.llama3-1-8b-instruct-v1:0",
    "llama3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
}
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
def normalize_max_output_tokens(value=None):
    return max(1, min(int(value if value is not None else 256), 2048))
@lru_cache(maxsize=1)
def get_bedrock_client():
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"),
        config=Config(connect_timeout=5, read_timeout=45,
                      retries={"mode": "standard", "total_max_attempts": 1}))
def invoke_model(model, prompt, max_output_tokens=None, temperature=0.2):
    if model not in MODEL_IDS:
        raise ValueError(f"Unknown model: {model}")
    cap = normalize_max_output_tokens(max_output_tokens)
    if model == "nova-micro":
        body = {"messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": cap, "temperature": temperature}}
    else:
        formatted = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" +
                     prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        body = {"prompt": formatted, "max_gen_len": cap, "temperature": temperature}
    response = get_bedrock_client().invoke_model(modelId=MODEL_IDS[model],
        body=json.dumps(body), contentType="application/json", accept="application/json")
    result = json.loads(response["body"].read())
    if model == "nova-micro":
        return dict(text="".join(p.get("text", "") for p in result["output"]["message"]["content"]),
                    input_tokens=result.get("usage", {}).get("inputTokens"),
                    output_tokens=result.get("usage", {}).get("outputTokens"),
                    stop_reason=result.get("stopReason"))
    return dict(text=result["generation"], input_tokens=result.get("prompt_token_count"),
                output_tokens=result.get("generation_token_count"), stop_reason=result.get("stop_reason"))
def get_embedding_result(text):
    response = get_bedrock_client().invoke_model(modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}), contentType="application/json", accept="application/json")
    data = json.loads(response["body"].read())
    return dict(embedding=data["embedding"], input_tokens=data.get("inputTextTokenCount"),
                output_tokens=0, text="")
def get_embedding(text):
    return get_embedding_result(text)["embedding"]
