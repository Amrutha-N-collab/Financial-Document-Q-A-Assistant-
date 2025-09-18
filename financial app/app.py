import streamlit as st
import pandas as pd
import pdfplumber
import openpyxl
import re
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import requests

# Set page configuration
st.set_page_config(
    page_title="Financial Document Q&A Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "document_type" not in st.session_state:
    st.session_state.document_type = None
if "ollama_available" not in st.session_state:
    st.session_state.ollama_available = False
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "mistral"
if "mode" not in st.session_state:
    st.session_state.mode = "document_qa"  # "free_chat" or "document_qa"

# Check Ollama connection
def check_ollama_connection():
    try:
        # Try to connect to Ollama API
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            if "models" in models_data and models_data["models"]:
                models = [model["name"] for model in models_data["models"]]
                st.session_state.ollama_available = True
                st.session_state.ollama_model = models[0]
                return True, f"Connected to Ollama. Available models: {', '.join(models)}"
        
        return False, "Ollama is running but no models are available. Please pull a model first."
    except Exception as e:
        return False, f"Ollama is not available: {str(e)}"

# Function to generate response using Ollama API
def generate_response_with_ollama(query, context=None):
    try:
        if context:
            # For document Q&A mode
            prompt = f"""
            You are a financial analyst assistant. Based on the following financial document context, answer the user's question.
            
            Financial Document Context:
            {json.dumps(context, indent=2)}
            
            User Question: {query}
            
            Provide a concise, accurate answer based only on the financial document. 
            If the information is not available in the document, say so.
            """
        else:
            # For free chat mode
            prompt = f"""
            You are a helpful AI assistant. Please answer the user's question.
            
            User Question: {query}
            
            Provide a helpful and accurate response.
            """
        
        # Generate response using Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": st.session_state.ollama_model,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error generating response: {response.text}"
            
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Please make sure Ollama is running."

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                # Try to extract tables from the page
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += " | ".join(str(cell) if cell is not None else "" for cell in row) + "\n"
                    text += "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return text

# Function to extract data from Excel
def extract_data_from_excel(uploaded_file):
    try:
        # Read the Excel file
        excel_data = pd.read_excel(uploaded_file, sheet_name=None)
        
        # Process each sheet
        processed_data = {}
        for sheet_name, df in excel_data.items():
            # Clean the dataframe
            df = df.dropna(how='all').reset_index(drop=True)
            
            # Convert DataFrame to a more readable format
            processed_data[sheet_name] = {
                "columns": list(df.columns),
                "data": df.values.tolist(),
                "head": df.head(10).to_dict('list')  # First 10 rows for display
            }
        
        return processed_data
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return None

# Function to identify document type and extract financial data
def process_financial_document(uploaded_file, file_type):
    try:
        if file_type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
            
            # Try to identify the document type based on content
            document_type = "Unknown"
            financial_data = {}
            
            # Check for income statement indicators
            income_keywords = ["revenue", "sales", "income statement", "gross profit", "operating income", "net income"]
            if any(keyword in text.lower() for keyword in income_keywords):
                document_type = "Income Statement"
                financial_data = extract_financial_data_from_text(text, "income")
            
            # Check for balance sheet indicators
            balance_keywords = ["balance sheet", "assets", "liabilities", "equity", "current assets", "fixed assets"]
            if any(keyword in text.lower() for keyword in balance_keywords):
                document_type = "Balance Sheet"
                financial_data = extract_financial_data_from_text(text, "balance")
            
            # Check for cash flow indicators
            cash_flow_keywords = ["cash flow", "operating activities", "investing activities", "financing activities"]
            if any(keyword in text.lower() for keyword in cash_flow_keywords):
                document_type = "Cash Flow Statement"
                financial_data = extract_financial_data_from_text(text, "cash_flow")
            
            return {
                "document_type": document_type,
                "text": text,
                "financial_data": financial_data
            }
        
        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            data = extract_data_from_excel(uploaded_file)
            
            if not data:
                return None
                
            # Try to identify document type based on sheet names
            document_type = "Unknown"
            sheet_names = list(data.keys())
            
            income_sheets = ["income", "revenue", "profit", "p&l"]
            balance_sheets = ["balance", "assets", "liabilities"]
            cash_flow_sheets = ["cash flow", "cash"]
            
            for sheet_name in sheet_names:
                if any(keyword in sheet_name.lower() for keyword in income_sheets):
                    document_type = "Income Statement"
                    break
                elif any(keyword in sheet_name.lower() for keyword in balance_sheets):
                    document_type = "Balance Sheet"
                    break
                elif any(keyword in sheet_name.lower() for keyword in cash_flow_sheets):
                    document_type = "Cash Flow Statement"
                    break
            
            return {
                "document_type": document_type,
                "sheets": data,
                "financial_data": extract_financial_data_from_excel(data)
            }
        
        return None
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# Function to extract financial data from text
def extract_financial_data_from_text(text, doc_type):
    financial_data = {}
    
    # Common financial patterns
    patterns = {
        "revenue": r"(revenue|sales|total revenue|total sales)[^\d]*([\d,]+\.?\d*)",
        "net_income": r"(net income|net profit|total comprehensive income)[^\d]*([\d,]+\.?\d*)",
        "gross_profit": r"(gross profit)[^\d]*([\d,]+\.?\d*)",
        "operating_income": r"(operating income|operating profit)[^\d]*([\d,]+\.?\d*)",
        "total_assets": r"(total assets)[^\d]*([\d,]+\.?\d*)",
        "total_liabilities": r"(total liabilities)[^\d]*([\d,]+\.?\d*)",
        "equity": r"(total equity|shareholders'? equity)[^\d]*([\d,]+\.?\d*)",
        "cash_flow_operations": r"(cash flow from operations|net cash provided by operating activities)[^\d]*([\d,]+\.?\d*)"
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the last match (often the most recent or total)
            value = parse_currency(matches[-1][1])
            if value:
                financial_data[key] = value
    
    return financial_data

# Function to extract financial data from Excel
def extract_financial_data_from_excel(data):
    financial_data = {}
    
    # Simple extraction for demonstration
    for sheet_name, sheet_data in data.items():
        # Look for common financial terms in the first row (headers)
        headers = sheet_data.get("columns", [])
        for i, header in enumerate(headers):
            header_lower = str(header).lower()
            
            # Map headers to financial metrics
            header_mappings = {
                "revenue": ["revenue", "sales", "income"],
                "net_income": ["net income", "net profit", "comprehensive income"],
                "gross_profit": ["gross profit"],
                "operating_income": ["operating income", "operating profit"],
                "total_assets": ["total assets", "assets"],
                "total_liabilities": ["total liabilities", "liabilities"],
                "equity": ["equity", "shareholders equity"],
                "cash_flow_operations": ["cash flow from operations", "operating activities"]
            }
            
            for metric, keywords in header_mappings.items():
                if any(keyword in header_lower for keyword in keywords):
                    # Try to get the first numeric value in this column
                    for value in sheet_data.get("data", []):
                        if i < len(value) and is_numeric(value[i]):
                            financial_data[metric] = parse_currency(value[i])
                            break
    
    return financial_data

# Helper function to parse currency values
def parse_currency(value):
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove commas and dollar signs
        value = value.replace(',', '').replace('$', '').replace('(', '-').replace(')', '').strip()
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    return 0.0

# Helper function to check if a value is numeric
def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# Function to generate response (fallback if Ollama is not available)
def generate_response_fallback(query, processed_data=None):
    if processed_data:
        # Document Q&A mode
        response = "I've analyzed the financial document. "
        
        if processed_data.get("document_type") != "Unknown":
            response += f"This appears to be a {processed_data['document_type']}. "
        
        # Check for specific financial metrics in the query
        query_lower = query.lower()
        financial_data = processed_data.get("financial_data", {})
        
        # Map query terms to financial metrics
        query_mappings = {
            "revenue": ["revenue", "sales", "income"],
            "net_income": ["net income", "profit", "earnings"],
            "gross_profit": ["gross profit"],
            "operating_income": ["operating income", "operating profit"],
            "total_assets": ["assets", "total assets"],
            "total_liabilities": ["liabilities", "debt"],
            "equity": ["equity", "shareholders"],
            "cash_flow_operations": ["cash flow", "operating cash"]
        }
        
        answered = False
        for metric, keywords in query_mappings.items():
            if any(keyword in query_lower for keyword in keywords) and metric in financial_data:
                value = financial_data[metric]
                response += f"The {metric.replace('_', ' ')} is ${value:,.2f}. "
                answered = True
        
        # If no specific financial metrics were found in the query
        if not answered:
            if financial_data:
                response += "I found the following financial metrics: " + ", ".join(
                    [f"{k.replace('_', ' ')} (${v:,.2f})" for k, v in financial_data.items()]
                ) + ". Ask me about any of these metrics."
            else:
                response += "I couldn't extract specific financial data from the document. Please try asking about specific metrics like revenue, net income, or assets."
        
        return response
    else:
        # Free chat mode fallback
        simple_responses = {
            "2+2": "2 + 2 = 4",
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! How can I assist you?",
            "help": "I can help answer questions about financial documents or have a general conversation. What would you like to know?",
            "thank": "You're welcome! Is there anything else I can help with?",
        }
        
        query_lower = query.lower()
        for key, response in simple_responses.items():
            if key in query_lower:
                return response
        
        return "I'm sorry, I can only provide limited responses without Ollama. Please make sure Ollama is running for full functionality."

# Main application
def main():
    st.title("📊 Financial Document Q&A Assistant")
    st.markdown("Choose between Free Chat with Ollama or Document Q&A for uploaded files.")
    
    # Mode selection
    mode = st.radio("Choose Mode:", ["💬 Free Chat", "📂 Document Q&A"], horizontal=True)
    
    if mode == "💬 Free Chat":
        st.session_state.mode = "free_chat"
    else:
        st.session_state.mode = "document_qa"
    
    # Check Ollama connection
    if not st.session_state.ollama_available:
        with st.spinner("Checking Ollama connection..."):
            success, message = check_ollama_connection()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.warning(message)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        if st.session_state.ollama_available:
            try:
                # Get available models
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    if "models" in models_data and models_data["models"]:
                        models = [model["name"] for model in models_data["models"]]
                        
                        # Model selection
                        selected_model = st.selectbox(
                            "Select Ollama Model",
                            models,
                            index=0
                        )
                        st.session_state.ollama_model = selected_model
            except:
                st.error("Could not retrieve models from Ollama")
        
        if st.session_state.mode == "document_qa":
            st.header("Document Upload")
            uploaded_file = st.file_uploader(
                "Choose a financial document",
                type=["pdf", "xlsx", "xls"],
                help="Supported formats: PDF, Excel"
            )
            
            if uploaded_file is not None:
                # Process the uploaded file
                with st.spinner("Processing document..."):
                    processed_data = process_financial_document(uploaded_file, uploaded_file.type)
                    
                    if processed_data:
                        st.session_state.processed_data = processed_data
                        st.session_state.document_type = processed_data.get("document_type", "Unknown")
                        
                        st.success(f"Document processed successfully! Detected type: {st.session_state.document_type}")
                        
                        # Display basic document information
                        st.subheader("Document Information")
                        st.write(f"**Type:** {st.session_state.document_type}")
                        st.write(f"**File Name:** {uploaded_file.name}")
                        
                        # Show extracted financial data if available
                        if processed_data.get("financial_data"):
                            st.subheader("Extracted Financial Data")
                            for key, value in processed_data["financial_data"].items():
                                st.write(f"**{key.replace('_', ' ').title()}:** ${value:,.2f}")
                    else:
                        st.error("Failed to process the document. Please try another file.")
        
        st.markdown("---")
        st.markdown("### How to Use")
        
        if st.session_state.mode == "document_qa":
            st.markdown("""
            1. Upload a financial document (PDF or Excel)
            2. Wait for the document to be processed
            3. Start asking questions in the chat interface
            4. Ask about revenue, expenses, profits, or other financial metrics
            """)
            
            st.markdown("### Supported Questions")
            st.markdown("""
            - What was the total revenue?
            - How much net income was reported?
            - What are the total assets?
            - What is the gross profit?
            - How much cash flow from operations?
            """)
        else:
            st.markdown("""
            1. Select an Ollama model (if available)
            2. Start asking questions in the chat interface
            3. The AI will respond using the selected model
            """)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Document Preview")
        
        if st.session_state.mode == "free_chat":
            st.info("Free Chat Mode – no document required.")
        elif st.session_state.processed_data is None:
            st.info("Please upload a document to get started.")
        else:
            if "text" in st.session_state.processed_data:
                # For PDF documents, show extracted text preview
                st.subheader("Extracted Text Preview")
                text_preview = st.session_state.processed_data["text"][:1000] + "..." if len(st.session_state.processed_data["text"]) > 1000 else st.session_state.processed_data["text"]
                st.text_area("Text Preview", text_preview, height=300, label_visibility="collapsed")
            
            if "sheets" in st.session_state.processed_data:
                # For Excel documents, show sheet preview
                st.subheader("Sheet Preview")
                sheet_names = list(st.session_state.processed_data["sheets"].keys())
                selected_sheet = st.selectbox("Select Sheet", sheet_names)
                
                if selected_sheet:
                    sheet_data = st.session_state.processed_data["sheets"][selected_sheet]
                    df_preview = pd.DataFrame(sheet_data["head"])
                    st.dataframe(df_preview)
    
    with col2:
        st.header("Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.ollama_available:
                        if st.session_state.mode == "document_qa" and st.session_state.processed_data:
                            # Prepare context for LLM
                            context = {
                                "document_type": st.session_state.processed_data.get("document_type", "Unknown"),
                                "financial_data": st.session_state.processed_data.get("financial_data", {}),
                                "text_preview": st.session_state.processed_data.get("text", "")[:2000] if "text" in st.session_state.processed_data else "",
                                "sheets_available": list(st.session_state.processed_data.get("sheets", {}).keys()) if "sheets" in st.session_state.processed_data else []
                            }
                            response = generate_response_with_ollama(prompt, context)
                        else:
                            # Free chat mode or no document
                            response = generate_response_with_ollama(prompt)
                    else:
                        # Fallback without Ollama
                        if st.session_state.mode == "document_qa" and st.session_state.processed_data:
                            response = generate_response_fallback(prompt, st.session_state.processed_data)
                        else:
                            response = generate_response_fallback(prompt)
                
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()