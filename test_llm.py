from config import GEMINI_MODELS, GROQ_MODELS, get_llm, has_key

if has_key("gemini"):
    print("Gemini models your key can use:")
    for m in GEMINI_MODELS[:12]:
        print("  -", m)
    print("\nTesting the top 2 Gemini models through CrewAI...\n")
    for m in GEMINI_MODELS[:2]:
        try:
            out = get_llm("gemini", m).call("Say hello in one short sentence.")
            print(f"[OK]   {m} -> {out!r}")
        except Exception as e:
            print(f"[FAIL] {m} -> {str(e)[:250]}")
else:
    print("No GEMINI_API_KEY found in .env")

if has_key("groq"):
    print("\nGroq backup models:", GROQ_MODELS[:3])