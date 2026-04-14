"""
Live Audio Sales Call Analyzer
Streams audio from the microphone, transcribes in chunks,
and provides real-time intent/sentiment analysis with sales recommendations.
"""

import queue
import tempfile
import re

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_SECONDS = 3
LLM_MODEL = "mistral"
LLM_TEMPERATURE = 0.0


# ============================================================================
# Data Models
# ============================================================================
class LiveIntent(BaseModel):
    intent: str
    sentiment: str
    entities: List[str]


# ============================================================================
# Text Cleaning
# ============================================================================
def clean_text(text: str) -> str:
    """Remove filler words and extra spaces."""
    text = text.lower()
    # NOTE: "like" intentionally excluded — too ambiguous
    text = re.sub(r"\b(uh|um|you know|I mean|so basically)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================================
# Decision Logic
# ============================================================================
def decide_live_action(intent_data: LiveIntent) -> str | None:
    """Return a sales action recommendation based on detected intent."""
    actions = {
        "pricing_objection": "💰 Empathize, then explain ROI before offering a discount",
        "complaint": "⚠️ Acknowledge the issue and ask a clarifying question",
        "purchase_intent": "✅ Move to close — discuss onboarding and next steps",
        "interest": "👂 Provide more details about features and benefits",
    }
    return actions.get(intent_data.intent)


# ============================================================================
# Live Audio Processor
# ============================================================================
class LiveAudioProcessor:
    """Captures microphone audio and analyzes it in real-time."""

    def __init__(self):
        self._audio_queue: queue.Queue = queue.Queue()
        self._buffer = np.zeros((0, 1), dtype="float32")
        self._stream: sd.InputStream | None = None

        # Load models lazily (not at import time)
        print("[LiveAudio] Loading Whisper model...")
        self.whisper = WhisperModel("small", device="cpu", compute_type="int8")

        print(f"[LiveAudio] Connecting to Ollama ({LLM_MODEL})...")
        self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        self.parser = PydanticOutputParser(pydantic_object=LiveIntent)

        self.prompt = ChatPromptTemplate.from_template("""
You are an intent and sentiment classifier for sales calls.

Text:
{text}

Classify:
- intent (pricing_objection, interest, complaint, purchase_intent, other)
- sentiment (positive, neutral, negative)
- entities (keywords mentioned)

{format_instructions}
""")
        self.chain = self.prompt | self.llm | self.parser

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"[Audio] Warning: {status}")
        self._audio_queue.put(indata.copy())

    def _get_audio_chunk(self) -> np.ndarray | None:
        """Collect audio from queue and return a chunk when enough is buffered."""
        while not self._audio_queue.empty():
            self._buffer = np.vstack([self._buffer, self._audio_queue.get()])

        required_frames = BUFFER_SECONDS * SAMPLE_RATE
        if len(self._buffer) >= required_frames:
            chunk = self._buffer[:required_frames]
            self._buffer = self._buffer[required_frames:]
            return chunk
        return None

    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe an audio chunk using Whisper."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio_chunk, SAMPLE_RATE)
            segments, _ = self.whisper.transcribe(f.name)
            return " ".join(seg.text for seg in segments)

    def _analyze_intent(self, text: str) -> LiveIntent:
        """Extract intent and sentiment from text using the LLM."""
        try:
            return self.chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            print(f"[Analysis] Intent parsing failed: {e}")
            return LiveIntent(intent="other", sentiment="neutral", entities=[])

    def start(self):
        """Start listening to the microphone and analyzing in real-time."""
        print("\n" + "=" * 50)
        print("  🎙️  Live Sales Call Analyzer")
        print("  Listening... (Ctrl+C to stop)")
        print("=" * 50 + "\n")

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self._audio_callback,
        )
        self._stream.start()

        try:
            while True:
                chunk = self._get_audio_chunk()
                if chunk is None:
                    continue

                transcript = self._transcribe_chunk(chunk)
                if not transcript.strip():
                    continue

                cleaned = clean_text(transcript)
                print(f"\n📝 Heard: \"{cleaned}\"")

                intent = self._analyze_intent(cleaned)
                print(f"   Intent: {intent.intent} | Sentiment: {intent.sentiment}")

                if intent.entities:
                    print(f"   Keywords: {', '.join(intent.entities)}")

                recommendation = decide_live_action(intent)
                if recommendation:
                    print(f"   💡 TIP: {recommendation}")

        except KeyboardInterrupt:
            print("\n\n[LiveAudio] Stopped. Goodbye! 👋")
        finally:
            self.stop()

    def stop(self):
        """Stop the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    processor = LiveAudioProcessor()
    processor.start()
