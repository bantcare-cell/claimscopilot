"""Learning experiment (not part of the app): first raw LLM call via Groq.

Written before any framework — plain SDK call to understand the messages
array, system vs user roles, and how a system prompt enforces guardrails
(the off-topic question below gets politely declined).
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are ClaimsCopilot, an assistant for a health insurance claims team. Answer only questions about health insurance claims and policies, in plain, simple language. If a question is not about claims or insurance, politely say that it's outside what you can help with. Never give medical advice."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print(response.choices[0].message.content)
