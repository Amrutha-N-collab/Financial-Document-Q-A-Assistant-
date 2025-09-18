
# Financial Document Q&A Assistant

A comprehensive Streamlit web application that processes financial documents (PDF and Excel formats) and provides an interactive question-answering system using natural language queries with local Ollama integration.

## 📋 Features

### Document Processing
- ✅ Accepts PDF and Excel file uploads containing financial statements
- ✅ Extracts text and numerical data from financial documents
- ✅ Supports common financial document types (Income statements, Balance sheets, Cash flow statements)
- ✅ Handles various document layouts and formats with advanced parsing

### Question-Answering System
- ✅ Implements natural language processing to understand user questions
- ✅ Provides accurate responses based on uploaded document content
- ✅ Supports conversational interactions with follow-up questions
- ✅ Extracts and presents specific financial metrics when requested

### Technical Implementation
- ✅ Uses Streamlit for the web application interface
- ✅ Deploys using Ollama with local Small Language Models (SLMs)
- ✅ Hosts the application locally (no cloud deployment required)
- ✅ Implements proper error handling and user feedback

### User Interface
- ✅ Clean, intuitive web interface for document upload
- ✅ Interactive chat interface for asking questions
- ✅ Displays extracted financial information in readable format
- ✅ Provides clear feedback on processing status and results

## 🚀 Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
Install Ollama:

Download from https://ollama.ai

Install and add to system PATH

Download a language model:

bash
ollama pull mistral
# or
ollama pull llama2
Start Ollama service:

bash
ollama serve
💻 Usage
Run the application:

bash
streamlit run financial_qa_assistant.py
Upload a financial document:

PDF files (text-based, not scanned images)

Excel files (.xlsx, .xls) with financial data

Ask questions naturally:

"What was the total revenue last year?"

"How much net income was reported?"

"What are the total assets and liabilities?"

"Show me the profit margin trends"

🎯 Supported Document Types
Income Statements

Balance Sheets

Cash Flow Statements

Financial reports and statements

Budgets and forecasts

🔧 Technical Details
Frontend: Streamlit web framework

NLP: Ollama with local LLMs (Mistral, Llama2, etc.)

PDF Processing: pdfplumber with table extraction

Excel Processing: pandas with financial pattern recognition

API: RESTful communication with Ollama HTTP API

📊 Extracted Financial Metrics
Revenue, Sales, Income

Net Income, Gross Profit, Operating Income

Total Assets, Current Assets, Fixed Assets

Total Liabilities, Current Liabilities, Long-term Debt

Equity, Shareholders' Equity

Cash Flow from Operations, Investing, Financing

🤝 Contributing
This project is designed as a demonstration of financial document processing with local AI. Feel free to extend functionality or improve the parsing algorithms.

text

## Step 4: Installation and Setup

1. **Create and activate conda environment:**
   ```bash
   conda create -n financial-qa python=3.9
   conda activate financial-qa
Install dependencies:

bash
pip install -r requirements.txt
Install and setup Ollama:

Download from https://ollama.ai

Install and ensure it's in your system PATH

Start the Ollama service:

bash
ollama serve
Download a model:

bash
ollama pull mistral
Run the application:

bash
streamlit run financial_qa_assistant.py
Step 5: Testing with Sample Data
Create a sample Excel file with financial data or use any PDF financial report. The application will:

Process the document and extract financial data

Identify the document type automatically

Allow natural language questions about the content

Provide responses using Ollama for advanced understanding

Fall back to rule-based responses if Ollama is unavailable

This implementation meets all your requirements with robust error handling, proper document processing, and a clean user interface.

