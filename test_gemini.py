import os
from dotenv import load_dotenv

load_dotenv()
from google import genai

key = os.getenv("GEMINI_API_KEY", "")
print("Key loaded:", bool(key), "| length:", len(key))
client = genai.Client(api_key=key)

print("\nModels your key can see:")
names = []
for m in client.models.list():
    n = m.name.replace("models/", "")
    actions = getattr(m, "supported_actions", None) or ["generateContent"]
    skip = any(w in n for w in ["image", "tts", "live", "audio", "embedding", "veo", "imagen", "robotics"])
    if ("gemini" in n or "gemma" in n) and "generateContent" in actions and not skip:
        names.append(n)
        print("  -", n)

preferred = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash"]
to_test = [m for m in preferred if m in names] or [m for m in names if "flash" in m][:3]

print("\nTesting models...\n")
for m in to_test:
    try:
        r = client.models.generate_content(model=m, contents="Say hello in one short sentence.")
        print(f"[OK]   {m} -> {r.text!r}")
    except Exception as e:
        print(f"[FAIL] {m} -> {str(e)[:200]}")