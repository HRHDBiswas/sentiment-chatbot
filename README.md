# Sentiment-Aware Chatbot

A traditional Python chatbot extended with a cloud-based AI language service
for real-time sentiment analysis.

**Course:** MSCS 5220 – Human-Computer Interaction  
**Institution:** University of the Cumberlands  
**Assignment:** Integrate Traditional Chatbot with AI Service  

---

## Origin

This project is based on the echo bot pattern introduced in Topic 5 of the course.
The original echo bot was copied as a starting point, and then extended with:
- A `DefaultConfig` class for environment-variable-backed configuration
- Integration with the HuggingFace Inference API for sentiment analysis
- Curated response pools mapped to detected sentiment categories
- Input validation, error handling, and graceful degradation

---

## AI Service Used

**HuggingFace Inference API** (free tier)  
**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
A fine-tuned RoBERTa model trained on Twitter data for three-class sentiment
classification: Positive, Negative, and Neutral.

HuggingFace was chosen over Azure AI Language because its free tier requires
no credit card and the REST API is straightforward to integrate.

---

## Features

- Detects sentiment (positive / negative / neutral) with confidence scores
- Responds with contextually appropriate, human-sounding messages
- Handles greetings, capability queries, and exit commands
- Validates API token at startup and provides clear error guidance
- Handles rate limiting, model cold starts, and network errors gracefully
- Rejects trivially short or empty inputs with a helpful prompt

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/[your-github-username]/sentiment-chatbot.git
cd sentiment-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your HuggingFace API token

Create a free account at https://huggingface.co and generate an API token
at https://huggingface.co/settings/tokens.

**macOS / Linux:**
```bash
export HF_API_TOKEN=hf_your_token_here
```

**Windows (Command Prompt):**
```cmd
SET HF_API_TOKEN=hf_your_token_here
```

### 4. Run the chatbot

```bash
python chatbot.py
```

---

## Usage

| Input | Bot behavior |
|---|---|
| `hello` / `hi` / `hey` | Greeting response |
| `help` / `?` | Lists capabilities |
| Any sentence or phrase | Sentiment analysis + response |
| Very short input (< 3 chars) | Asks for more input |
| `exit` / `quit` / `bye` | Exits cleanly |

---

## Example Session

```
============================================================
  Sentiment-Aware Chatbot
  Powered by HuggingFace AI (cardiffnlp/twitter-roberta)
============================================================
Type 'help' to see capabilities, or 'exit' to quit.

You: hello
Bot: Hello! I'm your sentiment-aware chatbot, powered by a RoBERTa model from HuggingFace.
Type anything and I'll analyze the emotional tone of your message.
Type 'help' to see my full capabilities.

You: I just got a promotion at work!
Bot: That's great to hear! Your message has a positive vibe to it (97% confident).

You: Oh great, another Monday
Bot: I'm sorry to hear that. Your message carries a negative tone (89% confident).
I hope things improve soon.

You: The meeting is at 3pm
Bot: Got it — your message reads as fairly neutral (82% confidence).

You: exit
Bot: Goodbye! Take care.
```

---

## Project Structure

```
sentiment-chatbot/
├── chatbot.py          # Main bot script and interaction loop
├── config.py           # DefaultConfig class (environment variable management)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
└── README.md           # This file
```

---

## Known Limitations

- **Stateless**: each message is analyzed independently; no conversational memory
- **Terminal-only**: no web front end; runs in a command-line session
- **Small response pool**: responses can repeat in longer sessions
- **No intent routing**: all non-command input goes to sentiment analysis
- **Free-tier rate limits**: heavy use may trigger 429 errors

---

## Future Work

- Add conversational memory (rolling window of recent turns)
- Implement confidence-aware response intensity scaling
- Add intent classification for multi-category routing
- Build a Flask/FastAPI web interface
- Expand response pools or use LLM-generated dynamic responses

---

## AI Disclosure

Claude (Anthropic) was used to assist with drafting documentation and refining response text.
The implementation, architecture decisions, and testing were done by the author.
