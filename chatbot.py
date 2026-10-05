"""
Sentiment-Aware Chatbot
-----------------------
Integrates a traditional Python chatbot with the HuggingFace Inference API
for real-time sentiment analysis using the cardiffnlp/twitter-roberta-base-sentiment-latest model.

Origin: Extended from the echo bot pattern introduced in MSCS 5220 Topic 5.
Reason: Assignment requires integrating a cloud AI service into a traditional chatbot.

Author: [Your Name]
Course: MSCS 5220 - Human-Computer Interaction
Institution: University of the Cumberlands
Date: October 2026
"""

import os
from dotenv import load_dotenv
load_dotenv()
import sys
import random
import time
import requests
from config import DefaultConfig

# ── Startup: validate config before accepting user input ─────────────────────

CONFIG = DefaultConfig()

if not CONFIG.HF_API_TOKEN:
    print(
        "\n[ERROR] HF_API_TOKEN environment variable is not set.\n"
        "Create a free account at https://huggingface.co, generate an\n"
        "API token under Settings > Access Tokens, then run:\n\n"
        "  export HF_API_TOKEN=hf_your_token_here   (macOS/Linux)\n"
        "  SET HF_API_TOKEN=hf_your_token_here       (Windows)\n"
    )
    sys.exit(1)

# ── Sentiment model endpoint ──────────────────────────────────────────────────

HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# ── Response pools (chosen at random to avoid repetition) ────────────────────

POSITIVE_RESPONSES = [
    "That's great to hear! Your message has a positive vibe to it ({score:.0%} confident).",
    "Love the energy! I'm picking up positive sentiment in what you said ({score:.0%} confidence).",
    "Sounds like things are going well — your message reads as positive ({score:.0%} confidence).",
    "I can feel the good vibes! Your message registers as positive ({score:.0%} confident).",
]

NEGATIVE_RESPONSES = [
    "I'm sorry to hear that. Your message carries a negative tone ({score:.0%} confident). "
    "I hope things improve soon.",
    "That sounds rough. I'm detecting a negative sentiment in what you shared ({score:.0%} confidence). "
    "Is there anything I can help with?",
    "I hear you — your message registers as negative ({score:.0%} confident). "
    "Sometimes just putting it into words helps.",
    "It sounds like things are a bit tough right now. "
    "Your message has a negative tone ({score:.0%} confidence). Hang in there.",
]

NEUTRAL_RESPONSES = [
    "Got it — your message reads as fairly neutral ({score:.0%} confidence).",
    "That seems pretty balanced in tone. I'm reading neutral sentiment ({score:.0%} confident).",
    "Noted! Your message comes across as neutral ({score:.0%} confidence).",
    "Your message has a neutral tone ({score:.0%} confident). "
    "Feel free to share more if you'd like.",
]

CAPABILITIES_TEXT = """\
Here's what I can do:
  • Analyze the sentiment of anything you type — positive, negative, or neutral
  • Report the model's confidence score alongside the detected sentiment
  • Respond to greetings
  • List my capabilities (type 'help' or '?')
  • Exit cleanly (type 'exit', 'quit', or 'bye')

I understand natural, conversational text — including informal language and sarcasm.
Try me with anything from "I aced my exam!" to "Ugh, nothing is going right today."
"""

GREETINGS = {"hi", "hello", "hey", "howdy", "good morning", "good afternoon", "good evening"}
HELP_COMMANDS = {"help", "?", "capabilities", "what can you do", "/help"}
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "/exit", "/quit"}

# ── Core AI function ──────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> tuple[str, float] | None:
    """
    Send text to the HuggingFace Inference API and return (label, score).
    Returns None on recoverable errors; raises on unrecoverable ones.
    """
    headers = {"Authorization": f"Bearer {CONFIG.HF_API_TOKEN}"}
    payload = {"inputs": text}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
    except requests.exceptions.Timeout:
        print("Bot: [The request timed out. Please try again.]\n")
        return None
    except requests.exceptions.ConnectionError:
        print("Bot: [I couldn't reach the AI service. Check your internet connection.]\n")
        return None

    if response.status_code == 429:
        print("Bot: [Rate limit reached on the free tier. Please wait a moment and try again.]\n")
        return None

    if response.status_code == 503:
        # Model is loading — HuggingFace returns an estimated_time field
        try:
            wait = response.json().get("estimated_time", 20)
        except Exception:
            wait = 20
        print(f"Bot: [The AI model is warming up — this happens after periods of inactivity. "
              f"I'll be ready in about {wait:.0f} seconds. Please retry.]\n")
        return None

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Bot: [The AI service returned an error: {e}. Please try again.]\n")
        return None

    data = response.json()

    # Response is a list of label-score dicts (possibly nested in a list)
    if not isinstance(data, list):
        print("Bot: [Unexpected response from the AI service. Please try again.]\n")
        return None

    predictions = data[0] if data and isinstance(data[0], list) else data

    if not predictions or not isinstance(predictions[0], dict):
        print("Bot: [Could not parse the AI response. Please try again.]\n")
        return None

    top = max(predictions, key=lambda x: x.get("score", 0))
    return top.get("label", "neutral"), top.get("score", 0.0)


def build_response(label: str, score: float) -> str:
    """Map a sentiment label to a human-friendly response string."""
    label_lower = label.lower()
    if "pos" in label_lower:
        template = random.choice(POSITIVE_RESPONSES)
    elif "neg" in label_lower:
        template = random.choice(NEGATIVE_RESPONSES)
    else:
        template = random.choice(NEUTRAL_RESPONSES)
    return template.format(score=score)


# ── Input dispatcher ──────────────────────────────────────────────────────────

def process_input(user_input: str) -> str:
    """
    Route user input to the appropriate handler.
    Returns the bot's response string, or 'EXIT' to signal shutdown.
    """
    text = user_input.strip()

    if not text:
        return "Please type something — I'm here when you're ready."

    lower = text.lower()

    if lower in EXIT_COMMANDS:
        return "EXIT"

    if lower in HELP_COMMANDS:
        return CAPABILITIES_TEXT

    if lower in GREETINGS:
        return (
            "Hello! I'm your sentiment-aware chatbot, powered by a RoBERTa model from HuggingFace.\n"
            "Type anything and I'll analyze the emotional tone of your message.\n"
            "Type 'help' to see my full capabilities."
        )

    # Malformed / trivially short input
    if len(text) < 3:
        return (
            "That's a little short for me to analyze meaningfully. "
            "Try a full sentence or phrase."
        )

    # Route to AI sentiment analysis
    result = analyze_sentiment(text)
    if result is None:
        return ""  # Error already printed inside analyze_sentiment

    label, score = result
    return build_response(label, score)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Sentiment-Aware Chatbot")
    print("  Powered by HuggingFace AI (cardiffnlp/twitter-roberta)")
    print("=" * 60)
    print("Type 'help' to see capabilities, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        response = process_input(user_input)

        if response == "EXIT":
            print("Bot: Goodbye! Take care.\n")
            break

        if response:
            print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
