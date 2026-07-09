import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
print(f"Loaded API key starting with: {key[:10] if key else 'None'}")

from langchain_google_genai import ChatGoogleGenerativeAI

models = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

for model_name in models:
    print(f"\n--- Testing model: {model_name} ---")
    try:
        # Set max_retries=0 so it fails immediately if rate limited
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=0.0, max_retries=0)
        res = llm.invoke("Say hello")
        print(f"Success! Response: {res.content}")
    except Exception as e:
        print(f"Failed: {e}")
