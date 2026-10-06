import os
from dotenv import load_dotenv
from openai import OpenAI

# Read variables from the .env file into the environment
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

# Fail early with a clear message instead of a confusing API error
if not api_key or "your" in api_key:
    raise SystemExit("OPENAI_API_KEY is missing or still a placeholder in .env")
if not model or "your" in model or "put-a-model" in model:
    raise SystemExit("OPENAI_MODEL is missing or still a placeholder in .env")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model=model,
    input="Reply with exactly these three words: setup works.",
)

print("Model used:", model)
print("Reply:", response.output_text)