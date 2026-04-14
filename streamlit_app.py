import streamlit as st
import tempfile
import os
from pathlib import Path
import re

import numpy as np
import sounddevice as sd

# Import the processing components
from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
import soundfile as sf

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="TECHNICAL_MONOLITH",
    page_icon=":material/memory:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar styling overrides */
[data-testid="stSidebar"] {
    background-color: #06080A;
    border-right: 1px solid #1A1F2B;
}

/* Base custom text classes */
.tech-header {
    font-family: 'JetBrains Mono', monospace;
    color: #64748B;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.tech-title {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
    font-size: 1.5rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* Metrics and Cards */
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
# Sidebar removed per user request. Hardcoding default LLM model parameters here.
model_choice = "mistral"
temperature = 0.0

# ============================================================================
# DATA MODELS
# ============================================================================
class IntentOutput(BaseModel):
    intent: str
    sentiment: str
    entities: List[str]
    confidence: float = 0.5  # 0.0 to 1.0, how confident the model is
    recommendation: str = ""  # AI-generated sales coaching tip

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached)"""
    return WhisperModel("small", device="cpu", compute_type="int8")

@st.cache_resource
def load_llm(model_name: str = "mistral", temp: float = 0.0):
    """Load LLM model (cached)"""
    return ChatOllama(model=model_name, temperature=temp)

def transcribe_audio(audio_path: str) -> str:
    """Convert audio file to text"""
    model = load_whisper_model()
    segments, _ = model.transcribe(audio_path)
    return " ".join(seg.text for seg in segments)

def clean_text(text: str) -> str:
    """Remove filler words and extra spaces"""
    text = text.lower()
    # NOTE: "like" intentionally excluded — too ambiguous
    # (e.g. "I like your product" would become "I your product")
    text = re.sub(r"\b(uh|um|you know|I mean|so basically)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_intent(cleaned_text: str, llm_model_name: str, temp: float) -> IntentOutput:
    """Extract intent and sentiment using AI"""
    llm = load_llm(llm_model_name, temp)
    parser = PydanticOutputParser(pydantic_object=IntentOutput)
    
    intent_prompt = ChatPromptTemplate.from_template("""You are an expert B2B sales call analyst and coach.

## Examples:
Text: "The price is way too high for our budget"
→ intent: pricing_objection, sentiment: negative, entities: ["price", "budget"], confidence: 0.95, recommendation: "Empathize with the budget concern first, then pivot to demonstrating ROI and total cost of ownership vs competitors."

Text: "That sounds interesting, can you tell me more about the API?"
→ intent: interest, sentiment: positive, entities: ["API"], confidence: 0.85, recommendation: "Share specific API documentation and offer a live demo to maintain the customer's momentum."

Text: "We've been having issues with your support response times"
→ intent: complaint, sentiment: negative, entities: ["support", "response times"], confidence: 0.90, recommendation: "Acknowledge the frustration sincerely and escalate to your support lead with a concrete resolution timeline."

Text: "Yes, let's go ahead and sign the contract"
→ intent: purchase_intent, sentiment: positive, entities: ["contract"], confidence: 0.95, recommendation: "Move swiftly to close. Send the contract for e-signature today and outline the onboarding timeline."

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
    
    try:
        result = intent_chain.invoke({
            "text": cleaned_text,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        # Warn the user instead of silently returning defaults
        st.warning(f"️ AI analysis failed — using fallback defaults. Error: {e}")
        return IntentOutput(intent="other", sentiment="neutral", entities=[])

def decide_action(intent_data: IntentOutput) -> str:
    """Decide what sales action to take"""
    if intent_data.intent == "pricing_objection":
        if intent_data.sentiment == "negative":
            return "PRICING_CONCERN: Review ROI metrics before discount."
        return "PRICING_QUERY: Detail pricing structure and value proposition."
    
    if intent_data.intent == "complaint":
        return "COMPLAINT_DETECTED: Execute escalation protocol."
    
    if intent_data.intent == "purchase_intent":
        return "READY_TO_BUY: Initiate closing procedure."
    
    if intent_data.intent == "interest":
        return "INTERESTED: Provide feature specifics."
    
    return "GENERAL: Maintain dialog."

def get_recommendation(intent_data: IntentOutput, llm_model_name: str, temp: float) -> str:
    """Get AI-powered sales recommendation"""
    llm = load_llm(llm_model_name, temp)
    
    recommendation_prompt = ChatPromptTemplate.from_template("""
You are an expert sales coach. Based on the customer's intent and sentiment, 
provide ONE short, actionable recommendation for the sales agent (2-3 sentences max).

Intent: {intent}
Sentiment: {sentiment}
Key words: {entities}

Recommendation:""")
    
    recommendation_chain = recommendation_prompt | llm
    
    try:
        result = recommendation_chain.invoke({
            "intent": intent_data.intent,
            "sentiment": intent_data.sentiment,
            "entities": ", ".join(intent_data.entities) if intent_data.entities else "none"
        })
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        st.warning(f"️ Recommendation generation failed. Error: {e}")
        return "Unable to generate recommendation at this time."


def record_audio(duration_seconds: float, samplerate: int = 16000) -> str:
    """Record audio from the local microphone and save to a temp WAV file.

    Returns the path to the temporary WAV file.
    """
    num_frames = int(duration_seconds * samplerate)
    recording = sd.rec(num_frames, samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        sf.write(tmp_file.name, recording, samplerate)
        return tmp_file.name


def analyze_audio_file(audio_path: str, model_choice: str, temperature: float, delete_after: bool = False) -> None:
    """Run the full analysis pipeline on a given audio file and render results in the UI."""

    try:
        with st.spinner("Processing audio... This may take a moment."):
            # Step 1: Transcribe
            with st.status("Transcribing audio...", expanded=True):
                raw_text = transcribe_audio(audio_path)
                st.write("Transcription complete.")
                st.info(f"**Raw Transcript:** {raw_text}")

            # Step 2: Clean text
            with st.status("Cleaning text filter...", expanded=True):
                cleaned_text = clean_text(raw_text)
                st.write("Text normalized.")
                st.info(f"**Cleaned Text:** {cleaned_text}")

            # Step 3: Extract intent + recommendation (single LLM call)
            with st.status("Running semantic analysis...", expanded=True):
                intent_result = extract_intent(cleaned_text, model_choice, temperature)
                st.write("Analysis complete.")

            # Step 4: Decision logic (rule-based, no LLM call needed)
            with st.status("Applying ruleset...", expanded=True):
                action = decide_action(intent_result)
                # Use AI recommendation from the combined prompt;
                # fall back to a separate call only if empty
                recommendation = intent_result.recommendation
                if not recommendation.strip():
                    recommendation = get_recommendation(intent_result, model_choice, temperature)
                st.write("Action plan generated.")

        # ========== DISPLAY RESULTS ==========
        st.markdown("---")
        st.subheader("ANALYSIS_RESULTS")

        # Intent, Sentiment, Keywords & Confidence Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="INTENT",
                value=intent_result.intent.upper(),
            )

        with col2:
            sentiment_emoji = {
                "positive": ":material/thumb_up:",
                "negative": ":material/thumb_down:",
                "neutral": ":material/remove:",
            }
            emoji = sentiment_emoji.get(intent_result.sentiment, "")
            st.metric(
                label="SENTIMENT",
                value=f"{emoji} {intent_result.sentiment.upper()}",
            )

        with col3:
            st.metric(
                label="KEYWORDS",
                value=len(intent_result.entities),
            )

        with col4:
            confidence_pct = int(intent_result.confidence * 100)
            st.metric(
                label="CONFIDENCE",
                value=f"{confidence_pct}%",
            )

        st.markdown("---")

        # Detailed Results
        st.subheader("DETAILED_METRICS")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Detected Information")
            st.markdown(f"""
            **Intent:** `{intent_result.intent}`

            **Sentiment:** `{intent_result.sentiment}`

            **Confidence:** `{int(intent_result.confidence * 100)}%`

            **Keywords/Entities:**
            """)
            if intent_result.entities:
                for entity in intent_result.entities:
                    st.write(f"• {entity}")
            else:
                st.write("• No entities detected")

        with col2:
            st.markdown("#### Customer's Message")
            st.text_area(
                "Original transcript:",
                value=raw_text,
                height=120,
                disabled=True,
                label_visibility="collapsed",
            )

        st.markdown("---")

        # Sales Recommendation
        st.subheader("AGENT_DIRECTIVE")

        st.markdown(
            f"""
            <div style="background-color: #000000; color: #ffffff; padding: 20px; border-left: 5px solid #1f77b4; border-radius: 5px;">
            <h4 style="margin-top: 0;">Recommended Action:</h4>
            <p style="font-size: 16px; font-weight: bold;">{action}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### AI-Generated Talking Points")
        st.info(recommendation)

        # Export results
        st.markdown("---")
        st.subheader("EXPORT_DATA")

        # Create summary text
        summary = f"""
SALES CALL ANALYSIS REPORT
{'='*50}

TRANSCRIPT:
{raw_text}

CLEANED TEXT:
{cleaned_text}

ANALYSIS RESULTS:
- Intent: {intent_result.intent}
- Sentiment: {intent_result.sentiment}
- Confidence: {int(intent_result.confidence * 100)}%
- Keywords: {', '.join(intent_result.entities) if intent_result.entities else 'None'}

RECOMMENDED ACTION:
{action}

SALES COACHING TIP:
{recommendation}

{'='*50}
Generated by Sales Call Intelligence System
        """

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="DOWNLOAD LOG",
                icon=":material/download:",
                data=summary,
                file_name="call_analysis.txt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f" Error processing audio: {str(e)}")
        st.info(
            " **Troubleshooting:**\n- Make sure Ollama is running (`ollama serve`)\n- Check that mistral model is installed (`ollama pull mistral`)\n- Try again with a shorter audio file",
        )
    finally:
        if delete_after and os.path.exists(audio_path):
            os.unlink(audio_path)

# ============================================================================
# MAIN APP
# ============================================================================
st.title("TECHNICAL_MONOLITH")
st.markdown("**SYSTEM_STATUS / ACTIVE** | TRANSCRIPT_LIBRARY")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs([":material/troubleshoot: SYSTEM_ANALYSIS", ":material/info: DOCS", ":material/bolt: QUICKSTART"])

# ============================================================================
# TAB 1: ANALYZE CALL
# ============================================================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Audio File")
        st.markdown("Supported formats: `.wav`, `.mp3`, `.m4a`, etc.")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Upload a recording of the sales call"
        )
    
    with col2:
        st.subheader("Or Use Sample")
        use_sample = st.checkbox("Use sample audio (customer.wav)", value=False)

        st.markdown("---")
        st.subheader("Or Record with Mic")
        audio_bytes = st.audio_input("Record Voice Node")
    
    st.markdown("---")
    
    # Process uploaded file or sample
    if uploaded_file or use_sample:
        if st.button("EXECUTE_ANALYSIS", icon=":material/play_arrow:", type="primary", use_container_width=True):
            # Get audio file path
            if use_sample:
                audio_path = "customer.wav"
                if not os.path.exists(audio_path):
                    st.error(" Sample file 'customer.wav' not found. Please upload an audio file instead.")
                    st.stop()
                analyze_audio_file(audio_path, model_choice, temperature, delete_after=False)
            else:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    audio_path = tmp_file.name
                analyze_audio_file(audio_path, model_choice, temperature, delete_after=True)
    else:
        st.info(" Upload an audio file or select the sample to get started!")

    # Process microphone recording
    if audio_bytes is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes.getbuffer())
                mic_audio_path = tmp_file.name
            analyze_audio_file(mic_audio_path, model_choice, temperature, delete_after=True)
        except Exception as e:
            st.error(f" Error analyzing microphone recording: {str(e)}")

# ============================================================================
# TAB 2: HOW IT WORKS
# ============================================================================
with tab2:
    st.subheader(" How This System Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Step-by-Step Process")
        st.markdown("""
        1. **Upload Audio** ️
           - Upload a customer call recording
        
        2. **Transcribe** 
           - AI converts speech to text
           - Uses faster-whisper model
        
        3. **Clean Text** 
           - Removes filler words (um, uh, like)
           - Removes extra spaces
        
        4. **Analyze** 
           - AI identifies customer intent
           - Detects sentiment (positive/negative/neutral)
           - Extracts key entities/keywords
        
        5. **Decide** 
           - System determines best sales action
           - Generates coaching recommendations
        
        6. **Results** 
           - Display findings and recommendations
        """)
    
    with col2:
        st.markdown("#### Intent Categories")
        st.markdown("""
        **pricing_objection**
        Customer concerned about cost
        
        **complaint**
        Customer unhappy with something
        
        **interest**
        Customer wants more information
        
        **purchase_intent**
        Customer ready to buy
        
        **other**
        Anything else
        """)
    
    st.markdown("---")
    
    st.markdown("#### Sentiment Analysis")
    st.markdown("""
    | Sentiment | Meaning | Example |
    |-----------|---------|---------|
    | **Positive**  | Customer happy/interested | "That sounds great!" |
    | **Neutral**  | Customer just informing | "OK, let me check" |
    | **Negative**  | Customer upset/hesitant | "Price is too high" |
    """)
    
    st.markdown("---")
    
    st.markdown("#### Technology Stack")
    st.markdown("""
    - **Speech-to-Text:** faster-whisper (OpenAI's Whisper)
    - **AI Model:** Mistral (via Ollama)
    - **Data Parsing:** Pydantic
    - **Framework:** LangChain
    - **Interface:** Streamlit
    """)

# ============================================================================
# TAB 3: QUICK START
# ============================================================================
with tab3:
    st.subheader(" Getting Started")
    
    st.markdown("#### Prerequisites")
    st.markdown("""
    Before using this app, make sure you have:
    
    1. **Ollama installed** - Download from [ollama.ai](https://ollama.ai)
    2. **Mistral model** - Run: `ollama pull mistral`
    3. **Python packages** - Already installed via requirements.txt
    """)
    
    st.markdown("---")
    
    st.markdown("#### Running This App")
    st.code("""
# Terminal 1 - Start Ollama
ollama serve

# Terminal 2 - Start Streamlit app
cd /Users/[YourUsername]/Desktop/AI_ML/Project001
source venv/bin/activate
streamlit run streamlit_app.py
    """, language="bash")
    
    st.markdown("---")
    
    st.markdown("#### Troubleshooting")
    
    with st.expander(" App hangs or says 'Connection refused'"):
        st.markdown("""
        **Problem:** Ollama is not running
        
        **Solution:**
        ```bash
        # In a separate terminal, run:
        ollama serve
        ```
        Then refresh this app.
        """)
    
    with st.expander(" Bad results / wrong intent detected"):
        st.markdown("""
        **Problem:** Model might be too weak or wrong
        
        **Solution:**
        1. Check the model in sidebar - should be "mistral"
        2. Try pulling a better model:
           ```bash
           ollama pull neural-chat
           ```
        3. Change model in sidebar settings
        """)
    
    with st.expander(" Audio file won't upload"):
        st.markdown("""
        **Problem:** File format not supported
        
        **Solution:** Convert to supported format:
        - .wav (recommended)
        - .mp3
        - .m4a
        - .ogg
        - .flac
        """)
    
    st.markdown("---")
    
    st.markdown("#### Sample Use Cases")
    st.markdown("""
     **Sales Training** - Review calls and improve responses
    
     **Quality Assurance** - Ensure agents follow best practices
    
     **Performance Analysis** - Track how agents handle objections
    
     **Customer Insights** - Understand what customers want
    
     **Compliance** - Verify standard sales processes are followed
    """)
    
    st.markdown("---")
    
    st.success("""
    **Ready to analyze your first call?** Go to the "Analyze Call" tab and upload an audio file!
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px; padding: 20px;">
 Sales Call Intelligence System | All processing done locally (no cloud) | Privacy first
</div>
""", unsafe_allow_html=True)
