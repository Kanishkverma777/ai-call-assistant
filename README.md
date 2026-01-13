# 🎯 Sales Call Intelligence System

A real-time AI-powered system that analyzes customer conversations to extract intent, sentiment, and provides personalized sales recommendations.

## 📖 What This Project Does

This application listens to sales calls (audio files), converts speech to text, analyzes customer sentiment and intent, and gives sales agents real-time recommendations on how to respond effectively.

**Use Case:** A sales agent is on a call with a customer who says *"The price feels too high for us right now."* The system instantly detects this is a **pricing objection** with **negative sentiment** and recommends: *"Empathize with concern, then explain ROI before discount."*

---

## 🚀 Key Features

✅ **Speech-to-Text Conversion** - Converts audio to text using AI  
✅ **Intent Detection** - Identifies customer intent (pricing objection, complaint, interest, etc.)  
✅ **Sentiment Analysis** - Detects customer emotion (positive, neutral, negative)  
✅ **Smart Recommendations** - Gives sales agents specific next steps  
✅ **Works Offline** - Uses local AI models (no cloud required)  
✅ **Web Interface** - Modern GUI with Streamlit  

---

## 📋 How It Works (Simple Explanation)

```
Audio File → Speech-to-Text → Clean Text → AI Analysis → Results
     ↓              ↓              ↓            ↓            ↓
customer.wav   "I like it but..."   Remove filler   Extract Intent   Show Recommendation
                                    & noise         & Sentiment
```

**Step by Step:**
1. **Audio Input** - Takes customer's audio file (`.wav`, `.mp3`, etc.)
2. **Transcription** - AI converts speech to text
3. **Text Cleaning** - Removes "um", "uh", "like" and extra spaces
4. **Analysis** - AI determines what customer wants and how they feel
5. **Decision Logic** - System decides what sales agent should do
6. **Recommendation** - AI generates specific talking points

---

## 🔧 Installation (Mac Users)

### Prerequisites
- **macOS** (10.14 or newer)
- **Python 3.8+** already installed on Mac

### Step 1: Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install PortAudio (required for audio processing)
```bash
brew install portaudio
```

### Step 3: Navigate to Project
```bash
cd /Users/[YourUsername]/Desktop/AI_ML/Project001
```

### Step 4: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
You should see `(venv)` at the start of your terminal line.

### Step 5: Install Python Packages
```bash
pip install -r requirements.txt
pip install streamlit
```

### Step 6: Install Ollama (Local AI Engine)
Download from: **https://ollama.ai** (choose Mac version)

OR use Homebrew:
```bash
brew install ollama
```

### Step 7: Pull AI Model
```bash
ollama pull mistral
```

---

## ▶️ How to Run

### Option A: Command Line (Direct)
```bash
# Terminal Tab 1 - Start Ollama (keep running)
ollama serve

# Terminal Tab 2 - Run the analysis
cd /Users/[YourUsername]/Desktop/AI_ML/Project001
source venv/bin/activate
python newrec.py
```

### Option B: Streamlit Web Interface (Recommended) ⭐
```bash
# Terminal Tab 1 - Start Ollama (keep running)
ollama serve

# Terminal Tab 2 - Run Streamlit app
cd /Users/[YourUsername]/Desktop/AI_ML/Project001
source venv/bin/activate
streamlit run streamlit_app.py
```

Then open your browser to: `http://localhost:8501`

---

## 📁 Project Structure

```
Project001/
├── newrec.py              # Main script (command line version)
├── streamlit_app.py       # Web interface version (Streamlit)
├── recommender.py         # Advanced version (live audio streaming)
├── requirements.txt       # Python packages needed
├── README.md              # This file
└── customer.wav           # Sample audio file for testing
```

---

## 🎓 File Explanations

| File | Purpose |
|------|---------|
| **newrec.py** | Main script - processes one audio file at a time |
| **streamlit_app.py** | Web GUI - upload files and see results visually |
| **recommender.py** | Advanced - listens to live microphone input |
| **requirements.txt** | List of all Python packages to install |

---

## 🧪 Testing & Examples

### Test with Sample File
```bash
# Make sure Ollama is running first!
python newrec.py
```

### Expected Output
```
Transcribing audio...
RAW TRANSCRIPT: Honestly, I like what you're offering, but the price feels too high for us right now.
CLEANED TEXT: honestly i what you're offering but price feels too high for us right now
Extracting intent and sentiment...
INTENT RESULT: intent='pricing_objection' sentiment='negative' entities=['price', 'high']
DECISION: Empathize with concern, then explain ROI before discount
Sales recommendation: [AI-generated tips]
```

---

## ❓ Troubleshooting

### Error: `ModuleNotFoundError: No module named 'ollama'`
**Solution:** Make sure you installed Ollama (the app, not pip)
```bash
# Check if Ollama is installed
which ollama

# If not, download from https://ollama.ai
```

### Error: `Connection refused` or script hangs
**Solution:** Ollama is not running. Open another terminal and run:
```bash
ollama serve
```

### Error: `portaudio.h not found`
**Solution:** Install PortAudio
```bash
brew install portaudio
```

### Script runs but gives wrong results
**Solution:** Make sure you're using the `mistral` model:
```bash
ollama pull mistral
```

### Streamlit won't start
**Solution:** Install Streamlit
```bash
pip install streamlit
```

---

## 🏗️ How the AI Works (Technical Details)

- **Speech Recognition:** `faster-whisper` (OpenAI's Whisper model)
- **AI Analysis:** `mistral` model via Ollama (runs locally, no internet required)
- **Data Parsing:** `Pydantic` (ensures output format is correct)
- **Framework:** `LangChain` (orchestrates AI pipeline)

---

## 📊 Intent Categories

The system can detect:
- **pricing_objection** - Customer concerned about cost
- **complaint** - Customer unhappy with something
- **interest** - Customer wants to know more
- **purchase_intent** - Customer ready to buy
- **other** - Anything else

---

## 💡 Use Cases

✅ Sales coaching - Train agents with real feedback  
✅ Call analytics - Understand customer patterns  
✅ Agent performance - Track how agents handle objections  
✅ Compliance - Ensure standard sales processes  

---

## 🔐 Privacy & Security

- **No cloud:** All data stays on your computer
- **No internet required:** Works completely offline
- **No account needed:** Just install and use

---

## 📝 Next Steps (Optional Enhancements)

- [ ] Add live call recording
- [ ] Export reports to PDF
- [ ] Dashboard with analytics
- [ ] Multi-language support
- [ ] Custom training data

---

## 📧 Support

If you encounter issues:
1. Check **Troubleshooting** section above
2. Make sure both Ollama and Python venv are running
3. Verify all packages installed: `pip list`

---

## 📜 License

This project is open source and free to use.

---

**Happy Selling! 🚀**