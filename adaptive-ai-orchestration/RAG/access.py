"""Optional deployment-owned source ACL. Never accept permissions from the caller.

Without RAG_ACCESS_POLICY the deployment explicitly uses one shared corpus.
With a policy, unlisted files/users are denied. File errors fail closed.
"""
import json
import os
from pathlib import Path

def allowed_sources(user_id):
    policy_path = os.getenv("RAG_ACCESS_POLICY")
    if not policy_path:
        return None
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or any(
        not isinstance(source, str) or not isinstance(users, list)
        or any(not isinstance(user, str) for user in users)
        for source, users in policy.items()
    ):
        raise ValueError("Invalid source access policy")
    return {source for source, users in policy.items()
            if user_id is not None and ("*" in users or str(user_id) in users)}
