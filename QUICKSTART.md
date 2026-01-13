# Quick Setup Guide for Mac

## 🚀 Fast Installation (5 minutes)

### Step 1: Open Terminal and Navigate to Project
```bash
cd /Users/[YourUsername]/Desktop/AI_ML/Project001
```

### Step 2: Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
*(You should see `(venv)` at the start of your terminal)*

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama
**Option A - Manual Download:**
- Go to https://ollama.ai
- Download Mac version
- Install it

**Option B - Using Homebrew:**
```bash
brew install ollama
```

### Step 5: Pull AI Model
```bash
ollama pull mistral
```

---

## ▶️ Running the App

### Step 1: Start Ollama (Keep This Running)
```bash
ollama serve
```
Leave this terminal open!

### Step 2: Open New Terminal Tab (Cmd+T)
```bash
cd /Users/[YourUsername]/Desktop/AI_ML/Project001
source venv/bin/activate
streamlit run streamlit_app.py
```

### Step 3: Open Browser
Your browser should automatically open to `http://localhost:8501`

If not, open it manually and go to: **http://localhost:8501**

---

## 📊 What You'll See

1. **Analyze Call Tab** - Upload audio and see results
2. **How It Works Tab** - Learn how the system works
3. **Quick Start Tab** - Help & troubleshooting

---

## ✅ Quick Test

To test if everything works:

1. Make sure Ollama is running (`ollama serve`)
2. Run: `streamlit run streamlit_app.py`
3. On the Streamlit page, check "Use sample audio (customer.wav)"
4. Click "Analyze Call"
5. You should see analysis results!

---

## 🆘 If Something Goes Wrong

**Ollama not working?**
```bash
# Check if installed
which ollama

# Check if model is downloaded
ollama list

# Pull the model if missing
ollama pull mistral
```

**Streamlit won't install?**
```bash
pip install streamlit --upgrade
```

**Python packages missing?**
```bash
pip install -r requirements.txt --upgrade
```

---

**That's it! You're all set! 🎉**
