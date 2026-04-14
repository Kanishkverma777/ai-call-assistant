"""
Sales Call Analyzer (CLI version)
Transcribes a customer audio file and provides AI-powered
intent/sentiment analysis with sales recommendations.
"""

import logging
import re
import sys

from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
AUDIO_FILE = "customer.wav"
LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.0

# ============================================================================
# Data Models
# ============================================================================
class IntentOutput(BaseModel):
    intent: str
    sentiment: str
    entities: List[str]
    confidence: float = 0.5  # 0.0 to 1.0, how confident the model is
    recommendation: str = ""  # AI-generated sales coaching tip

# ============================================================================
# ASR (Automatic Speech Recognition)
# ============================================================================
class ASR:
    def __init__(self):
        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(seg.text for seg in segments)

# ============================================================================
# Text Cleaning
# ============================================================================
def clean_text(text: str) -> str:
    text = text.lower()
    # NOTE: "like" intentionally excluded — too ambiguous
    # (e.g. "I like your product" would become "I your product")
    text = re.sub(r"\b(uh|um|you know|I mean|so basically)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ============================================================================
# Decision Logic (rule-based, no LLM)
# ============================================================================
def decide_action(intent_data: IntentOutput) -> str:
    if intent_data.intent == "pricing_objection":
        if intent_data.sentiment == "negative":
            return "Empathize with concern, then explain ROI before discount"
        return "Explain pricing structure clearly"

    if intent_data.intent == "complaint":
        return "Acknowledge issue and ask clarifying question"

    if intent_data.intent == "purchase_intent":
        return "Move to close and discuss onboarding"

    if intent_data.intent == "interest":
        return "Provide more details about features and benefits"

    return "Provide general clarification"

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    logger.info("Initializing ASR model...")
    asr = ASR()

    logger.info("Transcribing audio: %s", AUDIO_FILE)
    raw_text = asr.transcribe(AUDIO_FILE)
    logger.info("RAW TRANSCRIPT: %s", raw_text)

    cleaned_text = clean_text(raw_text)
    logger.info("CLEANED TEXT: %s", cleaned_text)

    # --- Setup LLM chain ---
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    parser = PydanticOutputParser(pydantic_object=IntentOutput)

    intent_prompt = ChatPromptTemplate.from_template("""You are an expert B2B sales call analyst and coach.

## Examples:
Text: "The price is way too high for our budget"
→ intent: pricing_objection, sentiment: negative, entities: ["price", "budget"], confidence: 0.95, recommendation: "Empathize with the budget concern first, then pivot to demonstrating ROI."

Text: "That sounds interesting, can you tell me more about the API?"
→ intent: interest, sentiment: positive, entities: ["API"], confidence: 0.85, recommendation: "Share specific API documentation and offer a live demo."

Text: "We've been having issues with your support response times"
→ intent: complaint, sentiment: negative, entities: ["support", "response times"], confidence: 0.90, recommendation: "Acknowledge the frustration and escalate with a concrete resolution timeline."

Text: "Yes, let's go ahead and sign the contract"
→ intent: purchase_intent, sentiment: positive, entities: ["contract"], confidence: 0.95, recommendation: "Move swiftly to close. Send the contract for e-signature today."

## Now classify this text AND provide a coaching recommendation:
Text: {text}

Rules:
- intent MUST be one of: pricing_objection, interest, complaint, purchase_intent, other
- sentiment MUST be one of: positive, neutral, negative
- entities: extract 1-5 specific nouns or phrases the customer mentioned
- confidence: a float between 0.0 and 1.0 indicating how confident you are
- recommendation: ONE short, actionable coaching tip for the sales agent (2-3 sentences max)
- If unsure, use intent="other" and a low confidence score

{format_instructions}
""")

    intent_chain = intent_prompt | llm | parser

    logger.info("Extracting intent, sentiment & recommendation...")

    try:
        intent_result = intent_chain.invoke({
            "text": cleaned_text,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("INTENT RESULT: %s", intent_result)
    except Exception as e:
        logger.warning("Intent parsing failed: %s", e)
        # Attempt to get raw model response and parse it; fall back to defaults.
        try:
            raw_resp = (intent_prompt | llm).invoke({
                "text": cleaned_text,
                "format_instructions": parser.get_format_instructions()
            })
            logger.debug("RAW MODEL RESPONSE: %s", raw_resp)
            raw_resp_text = None
            if hasattr(raw_resp, "content"):
                raw_resp_text = raw_resp.content
            elif isinstance(raw_resp, str):
                raw_resp_text = raw_resp
            else:
                raw_resp_text = str(raw_resp)

            try:
                intent_result = parser.parse(raw_resp_text)
                logger.info("Parsed intent from raw response: %s", intent_result)
            except Exception as e2:
                logger.error("Parser failed on raw response: %s", e2)
                intent_result = IntentOutput(intent="other", sentiment="neutral", entities=[])
        except Exception as e3:
            logger.error("LLM invocation failed while recovering intent: %s", e3)
            intent_result = IntentOutput(intent="other", sentiment="neutral", entities=[])

    # --- Decision Logic ---
    action = decide_action(intent_result)

    # --- Display Results ---
    print("\n" + "=" * 55)
    print("  📊 SALES CALL ANALYSIS REPORT")
    print("=" * 55)
    print(f"  Customer said:    {raw_text}")
    print(f"  Detected intent:  {intent_result.intent}")
    print(f"  Sentiment:        {intent_result.sentiment}")
    print(f"  Confidence:       {int(intent_result.confidence * 100)}%")
    print(f"  Keywords:         {', '.join(intent_result.entities) if intent_result.entities else 'None'}")
    print(f"  Sales action:     {action}")
    print(f"  AI Recommendation: {intent_result.recommendation}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
