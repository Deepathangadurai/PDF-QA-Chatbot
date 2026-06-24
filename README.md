# PDF-QA-Chatbot
An end-to-end RAG (Retrieval-Augmented Generation) chatbot built with Python and Streamlit that allows users to upload multiple PDFs and ask questions grounded strictly in the document context using FAISS vector storage and the Google Gemini API.

## 🚀 Key Features

* **Multi-File Processing:** Seamlessly uploads, extracts, and merges text across multiple PDF documents simultaneously.
* **Semantic Chunking:** Leverages sliding-window character splitting (`RecursiveCharacterTextSplitter`) to preserve paragraph context and eliminate data fragmentation.
* **Local Vector Indexing:** Generates embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` and indexes them into a high-performance **FAISS** vector store.
* **Contextual Guardrails:** Combines retrieved document segments with strict prompt engineering sent to `gemini-2.5-flash` to enforce factual grounding and minimize AI hallucinations.
* **Persistent Chat History:** Maintains full conversation memory across user prompts using Streamlit session state management.

---

## 🛠️ Tech Stack & Architecture

| Component                | Technology                                    |
| ------------------------ | --------------------------------------------- |
| Frontend UI              | Streamlit                                     |
| Document Parsing         | pypdf                                         |
| Orchestration & Chunking | LangChain, langchain-text-splitters           |
| Vector Database          | FAISS (Facebook AI Similarity Search)         |
| Embeddings Model         | HuggingFace Transformers (`all-MiniLM-L6-v2`) |
| LLM Integration          | Google GenAI SDK (`gemini-2.5-flash`)         |

---

## 📁 Repository Structure

```text
PDF_QA_Chatbot/
│
├── app.py               # Streamlit application interface & layout management
├── utils.py             # Core RAG pipeline (parsing, chunking, embeddings, LLM)
├── requirements.txt     # Python package dependencies
├── .env                 # Local environment file for secret keys (Do NOT commit)
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## 💻 Local Setup & Installation

Follow these steps to configure and run the application locally.

### Prerequisites

* Python **3.10, 3.11, or 3.12**
* A valid **Google Gemini API Key** (available from Google AI Studio)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Deepathangadurai/PDF-QA-Chatbot.git
cd PDF-QA-Chatbot
```

---

### Step 2: Create a Virtual Environment

Create and activate an isolated virtual environment.

#### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

Update `pip` and install the required packages.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Windows Users:**
> If you encounter a `c10.dll` loading error while launching PyTorch or SentenceTransformers, install the latest Microsoft Visual C++ Redistributable package and restart your terminal.

---

### Step 4: Configure Environment Variables

Create a file named `.env` in the project root directory and add your Gemini API key.

```env
GEMINI_API_KEY=your_actual_google_gemini_api_key_here
```

---

## 🚦 Running the Application

With the virtual environment activated, start the Streamlit application.

```bash
streamlit run app.py
```

The application will launch locally and open automatically in your default browser.

```text
http://localhost:8501
```

---

## 📝 How to Use the Chatbot

### 1️⃣ Upload Documents

Open the **Document Control Panel** in the sidebar and upload one or more PDF files.

### 2️⃣ Process Documents

Click the **"Process Documents"** button and wait for the success confirmation.

### 3️⃣ Ask Questions

Type your questions in the chat input box. The assistant will answer strictly based on the uploaded documents.

---

## 🔒 Security Notice

* The `.gitignore` file is configured to exclude:

  * `.env`
  * `venv/`
  * `__pycache__/`

* **Never commit or share your Gemini API key publicly.**

* Always store sensitive credentials in environment variables.

---


