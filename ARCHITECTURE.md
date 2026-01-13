# 🏗️ System Architecture & Workflow

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION LAYER                        │
├─────────────────────────────────────────────────────────────────────┤
│  Streamlit Web App (streamlit_app.py)                               │
│  ├─ Upload Audio File Interface                                    │
│  ├─ Real-time Progress Tracking                                    │
│  ├─ Results Display with Charts                                    │
│  └─ Configuration Sidebar (Model, Temperature)                     │
└────────────────────────┬──────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE LAYER                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐    ┌──────────────────┐                       │
│  │  Audio Input    │    │  Transcription   │                       │
│  │  .wav, .mp3,    │───▶│  (faster-whisper)│                       │
│  │  .m4a, etc.     │    │  Speech → Text   │                       │
│  └─────────────────┘    └─────────┬────────┘                       │
│                                    │                                 │
│                                    ▼                                 │
│                         ┌──────────────────┐                        │
│                         │  Text Cleaning   │                        │
│                         │  Remove: um, uh  │                        │
│                         │  Clean spacing   │                        │
│                         └─────────┬────────┘                        │
│                                    │                                 │
│                                    ▼                                 │
│                    ┌────────────────────────────┐                   │
│                    │  Intent & Sentiment        │                   │
│                    │  Analysis (AI Model)       │                   │
│                    │  ├─ Intent Detection       │                   │
│                    │  ├─ Sentiment Analysis     │                   │
│                    │  └─ Entity Extraction      │                   │
│                    └─────────┬──────────────────┘                   │
│                              │                                       │
│                              ▼                                       │
│                    ┌────────────────────────────┐                   │
│                    │  Decision Logic            │                   │
│                    │  Generate Recommendations  │                   │
│                    │  Create Sales Action Plan  │                   │
│                    └─────────┬──────────────────┘                   │
│                              │                                       │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ✅ Transcription Result                                             │
│  ✅ Cleaned Text                                                     │
│  ✅ Intent Detected (pricing_objection, complaint, etc.)            │
│  ✅ Sentiment (positive, neutral, negative)                         │
│  ✅ Keywords/Entities Found                                         │
│  ✅ Sales Action Recommendation                                     │
│  ✅ AI Coaching Tips                                                │
│  ✅ Export as Text File                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STREAMLIT APPLICATION                         │
│                    (Web Interface & UI)                           │
└──────────────────┬───────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Audio Handler   │  │  LLM Manager     │
│  - Upload       │  │  - Model Config  │
│  - Save Temp    │  │  - Temperature   │
│  - Cleanup      │  │  - Parameters    │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Processing Core      │
        │  (newrec.py logic)    │
        └────────┬──────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌────────┐  ┌───────────┐  ┌──────────┐
│Whisper │  │  LangChain│  │ Pydantic │
│ (ASR)  │  │  (Chains) │  │ (Parser) │
└────────┘  └───────────┘  └──────────┘
    │            │            │
    └────────────┼────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Ollama Backend  │
        │  (Local AI)      │
        │  Model: mistral  │
        └──────────────────┘
```

---

## Technology Stack

```
FRONTEND LAYER
├─ Streamlit              (Web framework)
├─ HTML/CSS               (Custom styling)
└─ Python UI Components   (Tabs, buttons, metrics)

PROCESSING LAYER
├─ faster-whisper         (Speech → Text)
├─ pydantic               (Data validation)
├─ langchain              (AI orchestration)
├─ langchain-ollama       (Local LLM integration)
└─ Python regex           (Text cleaning)

AI/ML LAYER
├─ Ollama                 (Local AI engine)
├─ mistral model          (Language model)
└─ Whisper model          (Speech recognition)

DATA LAYER
├─ Audio files            (.wav, .mp3, .m4a, etc.)
├─ In-memory processing   (No databases)
└─ Temp file storage      (Automatic cleanup)

UTILITIES
├─ soundfile              (Audio I/O)
├─ pydub                  (Audio manipulation)
├─ numpy                  (Numerical operations)
└─ pandas                 (Data handling)
```

---

## Request/Response Flow

### User Request: Upload Audio

```
User uploads audio file
         │
         ▼
Streamlit receives file
         │
         ▼
Save to temp location
         │
         ▼
Call transcribe_audio()
         │
         ▼
faster-whisper processes
         │
         ▼
Return text transcript
         │
         ▼
Call clean_text()
         │
         ▼
Remove filler words
         │
         ▼
Call extract_intent()
         │
         ▼
Create LangChain prompt
         │
         ▼
Send to Ollama (mistral)
         │
         ▼
AI analyzes and responds
         │
         ▼
Parser validates response
         │
         ▼
Return structured data
         │
         ▼
Call decide_action()
         │
         ▼
Return recommended action
         │
         ▼
Get additional coaching
         │
         ▼
Display all results in UI
```

---

## Intent Detection Logic

```
Customer says: "The price feels too high for us right now"
              │
              ▼
        Extract Intent: "pricing_objection"
              │
              ▼
        Detect Sentiment: "negative"
              │
              ▼
        Find Entities: ["price", "high"]
              │
              ▼
        Match to Rules:
        IF intent == "pricing_objection" AND sentiment == "negative"
        THEN recommend: "Empathize with concern, then explain ROI"
              │
              ▼
        Return Action & Coaching Tips
```

---

## File Relationships

```
streamlit_app.py (Main GUI)
    ├─ Uses: ChatOllama (LLM)
    ├─ Uses: WhisperModel (ASR)
    ├─ Uses: PydanticOutputParser
    ├─ Uses: ChatPromptTemplate
    ├─ Uses: IntentOutput (data model)
    ├─ Imports: faster_whisper
    ├─ Imports: langchain_ollama
    ├─ Imports: pydantic
    └─ Imports: streamlit

newrec.py (Command-line version)
    ├─ Same logic as streamlit_app
    ├─ No GUI
    └─ Direct output to terminal

recommender.py (Advanced version)
    ├─ Live audio streaming
    ├─ Real-time processing
    └─ More complex setup
```

---

## Data Models

### IntentOutput (Pydantic Model)
```python
class IntentOutput(BaseModel):
    intent: str              # "pricing_objection", "complaint", etc.
    sentiment: str           # "positive", "neutral", "negative"
    entities: List[str]      # ["price", "cost", "high"]
```

---

## Processing Timeline

```
Time  Action                          Typically Takes
────  ──────────────────────────────  ─────────────────
 0s   User uploads file               Instant
 1s   Save to temp location           < 1s
 2s   Whisper transcribes audio       3-10s (depends on audio length)
 3s   Clean text                      < 1s
 4s   Send to AI for analysis         5-15s (mistral thinking time)
 5s   Parse response                  < 1s
 6s   Generate recommendation         2-5s
 7s   Display results                 Instant
 ────────────────────────────────────────────────────
     Total: ~15-30 seconds
```

---

## Error Handling Flow

```
Try to process
     │
     ├─ Success? ─── YES ──→ Return results
     │
     └─ NO (Exception caught)
        │
        ├─ Try recovery path
        │  └─ Get raw model response
        │     └─ Manual parsing
        │        └─ Success? ──→ Return parsed results
        │        │
        │        └─ NO ────┐
        │                  │
        └─ Return defaults (intent="other", sentiment="neutral")
```

---

## Security & Privacy

```
User's Audio File
     │
     ├─ Never sent to cloud
     │
     ├─ Process locally only
     │  └─ Whisper (local)
     │  └─ Mistral (local via Ollama)
     │
     ├─ Store temporarily
     │  └─ Auto-deleted after processing
     │
     └─ Display results only to user
```

---

## Scalability Considerations

### Current Setup (Single User)
- Process one file at a time
- Suitable for: Training, testing, single agent

### Future Scaling Options
- Add queue system for multiple requests
- Database for storing analysis history
- Multi-worker processing
- API endpoints for integration
- Real-time dashboards
- Batch processing

---

**All components work together seamlessly to provide real-time AI analysis of sales calls!** 🚀
