"""
Simple Qwen (DashScope) OpenAI-compatible API test.

Requirements:
  pip install -U openai

Env vars (recommended):
  export DASHSCOPE_API_KEY="your_key"
  # intl (Singapore) endpoint:
  export OPENAI_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  # or CN (Beijing) endpoint:
  # export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

  # Model suggestion:
  export OPENAI_MODEL="qwen-turbo-latest"
  # (If your account supports other Qwen models, you can replace it.)
"""

import os
import sys
from dotenv import load_dotenv

loaded = load_dotenv()  # loads GEMINI_API_KEY
if not loaded:
    print("❌ Error: .env file not found!")
elif not os.getenv("DASHSCOPE_API_KEY"):
    print("❌ Error: DASHSCOPE_API_KEY not found in .env!")
else:
    print("✅ Environment loaded successfully. API Key detected.")


def build_client():
    try:
        from openai import OpenAI  # openai>=1.0.0
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `openai`. Install: pip install -U openai"
        ) from e

    base_url = os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = os.getenv("DASHSCOPE_API_KEY", "")

    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    return OpenAI(api_key=api_key, base_url=base_url)


def chat(client, prompt: str) -> str:
    """
    This is the same style as your KeeperLLMClient._chat:
      client.chat.completions.create(model=..., messages=[...])
    """
    model = os.getenv("OPENAI_MODEL", "qwen-turbo-latest")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )

    # Standard OpenAI-compatible response shape
    return (resp.choices[0].message.content or "").strip()


def test_1():
    prompt = "Say hello and confirm you are reachable. Reply in one sentence."
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    client = build_client()
    print(
        "Base URL:",
        os.getenv(
            "OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        ),
    )
    print("Model:", os.getenv("OPENAI_MODEL", "qwen-turbo-latest"))
    print("\n--- Prompt ---")
    print(prompt)

    try:
        out = chat(client, prompt)
    except Exception as e:
        print("\n[ERROR] API call failed:")
        print(repr(e))
        print("\nCommon fixes:")
        print("1) Ensure `pip show openai` is >= 1.0.0")
        print("2) Ensure DASHSCOPE_API_KEY is correct and exported")
        print("3) Try switching OPENAI_BASE_URL between intl and CN endpoints")
        print("4) Try model = qwen-turbo-latest / qwen-plus-latest if available")
        raise

    print("\n--- Response ---")
    print(out)


if __name__ == "__main__":
    test_1()
