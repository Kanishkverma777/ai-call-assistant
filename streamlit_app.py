import streamlit as st
import tempfile
import os
from pathlib import Path
import re

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
    page_title="Sales Call Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .intent-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin: 10px 0;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 15px;
        border-left: 5px solid #28a745;
        border-radius: 5px;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 15px;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    st.subheader("Model Settings")
    model_choice = st.selectbox(
        "Choose LLM Model",
        ["tinyllama", "mistral", "neural-chat", "llama2"],
        help="Choose which AI model to use. Tinyllama is fast and lightweight."
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower = more consistent, Higher = more creative"
    )
    
    st.markdown("---")
    st.subheader("About")
    st.info("""
    **Sales Call Intelligence System**
    
    Analyzes customer conversations to extract intent and sentiment, providing real-time sales recommendations.
    
    All processing happens locally - no data sent to cloud!
    """)

# ============================================================================
# DATA MODELS
# ============================================================================
class IntentOutput(BaseModel):
    intent: str
    sentiment: str
    entities: List[str]

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
    text = re.sub(r"\b(uh|um|you know|like)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_intent(cleaned_text: str, llm_model_name: str, temp: float) -> IntentOutput:
    """Extract intent and sentiment using AI"""
    llm = load_llm(llm_model_name, temp)
    parser = PydanticOutputParser(pydantic_object=IntentOutput)
    
    intent_prompt = ChatPromptTemplate.from_template("""
You are an intent and sentiment classifier for sales calls.

Text:
{text}

Classify:
- intent (pricing_objection, interest, complaint, purchase_intent, other)
- sentiment (positive, neutral, negative)
- entities (keywords mentioned)

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
        # Fallback to defaults
        return IntentOutput(intent="other", sentiment="neutral", entities=[])

def decide_action(intent_data: IntentOutput) -> str:
    """Decide what sales action to take"""
    if intent_data.intent == "pricing_objection":
        if intent_data.sentiment == "negative":
            return "🔴 PRICING CONCERN: Empathize with concern, then explain ROI before offering discount"
        return "💰 PRICING QUESTION: Explain pricing structure clearly and highlight value proposition"
    
    if intent_data.intent == "complaint":
        return "⚠️ COMPLAINT DETECTED: Acknowledge issue, empathize, and ask clarifying questions"
    
    if intent_data.intent == "purchase_intent":
        return "✅ READY TO BUY: Move to close and discuss onboarding details"
    
    if intent_data.intent == "interest":
        return "👂 INTERESTED: Provide more details about features and benefits"
    
    return "ℹ️ GENERAL: Provide clarification and continue building rapport"

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
        return "Unable to generate recommendation at this time."

# ============================================================================
# MAIN APP
# ============================================================================
st.title("🎯 Sales Call Intelligence System")
st.markdown("*Real-time AI analysis of customer conversations*")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Analyze Call", "📚 How It Works", "⚡ Quick Start"])

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
    
    # Process button
    if uploaded_file or use_sample:
        if st.button("🎙️ Analyze Call", type="primary", use_container_width=True):
            with st.spinner("Processing audio... This may take a moment."):
                try:
                    # Get audio file path
                    if use_sample:
                        audio_path = "customer.wav"
                        if not os.path.exists(audio_path):
                            st.error("❌ Sample file 'customer.wav' not found. Please upload an audio file instead.")
                            st.stop()
                    else:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            audio_path = tmp_file.name
                    
                    # Step 1: Transcribe
                    with st.status("🎙️ Transcribing audio...", expanded=True):
                        raw_text = transcribe_audio(audio_path)
                        st.write("✅ Transcription complete")
                        st.info(f"**Raw Transcript:** {raw_text}")
                    
                    # Step 2: Clean text
                    with st.status("🧹 Cleaning text...", expanded=True):
                        cleaned_text = clean_text(raw_text)
                        st.write("✅ Text cleaned")
                        st.info(f"**Cleaned Text:** {cleaned_text}")
                    
                    # Step 3: Extract intent
                    with st.status("🔍 Analyzing intent & sentiment...", expanded=True):
                        intent_result = extract_intent(cleaned_text, model_choice, temperature)
                        st.write("✅ Analysis complete")
                    
                    # Step 4: Decision logic
                    with st.status("💡 Generating recommendation...", expanded=True):
                        action = decide_action(intent_result)
                        recommendation = get_recommendation(intent_result, model_choice, temperature)
                        st.write("✅ Recommendation generated")
                    
                    # ========== DISPLAY RESULTS ==========
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Intent & Sentiment Cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="🎯 Intent Detected",
                            value=intent_result.intent.upper(),
                        )
                    
                    with col2:
                        sentiment_emoji = {
                            "positive": "😊",
                            "negative": "😞",
                            "neutral": "😐"
                        }
                        emoji = sentiment_emoji.get(intent_result.sentiment, "❓")
                        st.metric(
                            label="😊 Sentiment",
                            value=f"{emoji} {intent_result.sentiment.upper()}",
                        )
                    
                    with col3:
                        st.metric(
                            label="🏷️ Keywords Found",
                            value=len(intent_result.entities),
                        )
                    
                    st.markdown("---")
                    
                    # Detailed Results
                    st.subheader("📋 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Detected Information")
                        st.markdown(f"""
                        **Intent:** `{intent_result.intent}`
                        
                        **Sentiment:** `{intent_result.sentiment}`
                        
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
                            label_visibility="collapsed"
                        )
                    
                    st.markdown("---")
                    
                    # Sales Recommendation
                    st.subheader("💼 Sales Agent Recommendation")
                    
                    st.markdown(f"""
                    <div style="background-color: #000000; color: #ffffff; padding: 20px; border-left: 5px solid #1f77b4; border-radius: 5px;">
                    <h4 style="margin-top: 0;">Recommended Action:</h4>
                    <p style="font-size: 16px; font-weight: bold;">{action}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### AI-Generated Talking Points")
                    st.info(recommendation)
                    
                    # Export results
                    st.markdown("---")
                    st.subheader("📥 Export Results")
                    
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
                            label="📄 Download as Text",
                            data=summary,
                            file_name="call_analysis.txt",
                            mime="text/plain"
                        )
                    
                    # Clean up temp file
                    if not use_sample and os.path.exists(audio_path):
                        os.unlink(audio_path)
                
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                    st.info("💡 **Troubleshooting:**\n- Make sure Ollama is running (`ollama serve`)\n- Check that mistral model is installed (`ollama pull mistral`)\n- Try again with a shorter audio file")
    else:
        st.info("👆 Upload an audio file or select the sample to get started!")

# ============================================================================
# TAB 2: HOW IT WORKS
# ============================================================================
with tab2:
    st.subheader("🔧 How This System Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Step-by-Step Process")
        st.markdown("""
        1. **Upload Audio** 🎙️
           - Upload a customer call recording
        
        2. **Transcribe** 📝
           - AI converts speech to text
           - Uses faster-whisper model
        
        3. **Clean Text** 🧹
           - Removes filler words (um, uh, like)
           - Removes extra spaces
        
        4. **Analyze** 🔍
           - AI identifies customer intent
           - Detects sentiment (positive/negative/neutral)
           - Extracts key entities/keywords
        
        5. **Decide** 💡
           - System determines best sales action
           - Generates coaching recommendations
        
        6. **Results** 📊
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
    | **Positive** 😊 | Customer happy/interested | "That sounds great!" |
    | **Neutral** 😐 | Customer just informing | "OK, let me check" |
    | **Negative** 😞 | Customer upset/hesitant | "Price is too high" |
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
    st.subheader("⚡ Getting Started")
    
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
    
    with st.expander("🔴 App hangs or says 'Connection refused'"):
        st.markdown("""
        **Problem:** Ollama is not running
        
        **Solution:**
        ```bash
        # In a separate terminal, run:
        ollama serve
        ```
        Then refresh this app.
        """)
    
    with st.expander("🔴 Bad results / wrong intent detected"):
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
    
    with st.expander("🔴 Audio file won't upload"):
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
    ✅ **Sales Training** - Review calls and improve responses
    
    ✅ **Quality Assurance** - Ensure agents follow best practices
    
    ✅ **Performance Analysis** - Track how agents handle objections
    
    ✅ **Customer Insights** - Understand what customers want
    
    ✅ **Compliance** - Verify standard sales processes are followed
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
🚀 Sales Call Intelligence System | All processing done locally (no cloud) | Privacy first
</div>
""", unsafe_allow_html=True)
