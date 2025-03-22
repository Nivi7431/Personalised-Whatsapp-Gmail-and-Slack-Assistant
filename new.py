import streamlit as st 
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

background_image_path = os.path.abspath("bgm.jpeg")

def set_background():
    page_bg_img = '''
    <style>
    .stApp {
        background: url("bgm.jpeg") no-repeat center fixed;
        background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Initialize LLM
LLM = Ollama(model="llama3")

# Load Gmail, WhatsApp & Slack PDFs
gmail_loader = PyPDFLoader("Email_data.pdf.pdf")
whatsapp_loader = PyPDFLoader("whatsapp_chat_.pdf")
slack_loader = PyPDFLoader("slack_meeting_messages.pdf") 

gmail_pdf = gmail_loader.load()
whatsapp_pdf = whatsapp_loader.load()
slack_pdf = slack_loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
gmail_documents = text_splitter.split_documents(gmail_pdf)
whatsapp_documents = text_splitter.split_documents(whatsapp_pdf)
slack_documents = text_splitter.split_documents(slack_pdf)

# Create vector databases
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
gmail_db = Chroma.from_documents(gmail_documents, embeddings, persist_directory="./gmail_db")
whatsapp_db = Chroma.from_documents(whatsapp_documents, embeddings, persist_directory="./whatsapp_db")
slack_db = Chroma.from_documents(slack_documents, embeddings, persist_directory="./slack_db")

# Initialize session state
if "assistant_mode" not in st.session_state:
    st.session_state.assistant_mode = "Gmail"

if "history" not in st.session_state:
    st.session_state.history = []

def set_mode(mode):
    st.session_state.assistant_mode = mode
    st.session_state.history = []

# Prompts for AI assistants
Gmail_Prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Gmail personal assistant named RUBY. 
    - Summarize long email threads 
    - Suggest quick replies 
    - Automatically categorize emails based on Urgent, Follow-up,
      Low Priority
    - Extract key discussion topics 
    - Answer only to the question, Dont give anything irrelevant.
    - If you don't know anything, reply that you don't know.
    - Track & remind about unread messages"""),
    ("human", "{input}")
])

whatsapp_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a WhatsApp personal assistant named RUBY. 
    - Automate routine responses 
    - Summarize long chats 
    - Set up reminders 
    - Answer only to the question, Dont give anything irrelevant.
    - If you don't know anything, reply that you don't know.
    - Handle customer service queries efficiently"""),
    ("human", "{input}")
])

slack_prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a Slack personal assistant named RUBY.
    - Extract important updates from multiple channels
    - Deliver a summary of key discussions and action points
    - Identify actionable messages and suggest task creation
    - Answer only to the question, Dont give anything irrelevant.
    - If you don't know anything, reply that you don't know.
    - Enable quick lookup of past messages and files."""),
    ("human","{input}")
])

Chain_Gmail = Gmail_Prompt | LLM | StrOutputParser()
Chain_whatsapp = whatsapp_prompt | LLM | StrOutputParser()
slack_chain = slack_prompt | LLM | StrOutputParser()

def process_query():
    user_input = st.session_state.user_input.strip()
    
    if not user_input:
        return
    
    # Retrieve relevant context from the vector database
    if st.session_state.assistant_mode == "Gmail":
        retrieved_docs = gmail_db.similarity_search(user_input, k=5)
        chain = Chain_Gmail
        
        # Categorization only for Gmail
        urgent = []
        follow_UP = []
        low_priority_keywords = []

        urgent = {"urgent", "asap", "immediate attention", "high priority", "action required", 
                           "critical", "deadline", "emergency", "important", "time-sensitive", "immediate Response Needed, Critical Issue","Server Downtime Alert"}
        followup_keywords = {"follow-up", "reminder", "checking in", "status update", "follow through", 
                             "revisit", "awaiting response", "next steps", "pending action", "update required","review needed"}
        low_priority_keywords = {"low priority", "no rush", "when convenient", "fyi", "just an update", 
                                 "optional", "take your time", "for your reference", "general information", "non-urgent","Registration Open","Upcoming Event","General Inquiry","Information Update"}
        
        for doc in retrieved_docs:
            email_content = doc.page_content.lower()
            subject_line = ""
            
            lines = doc.page_content.split("\n")
            for line in lines:
                if line.lower().startswith("subject:"):
                    subject_line = line.strip()
                    break

            if not subject_line:
                continue

            if any(keyword in email_content for keyword in urgent):
                urgent.append(subject_line)
            elif any(keyword in email_content for keyword in followup_keywords):
                follow_UP.append(subject_line)
            elif any(keyword in email_content for keyword in low_priority_keywords):
                low_priority_keywords.append(subject_line)
        
        response = ""
        if urgent:
            response += "**Urgent Emails:**\n" + "\n".join(urgent) + "\n\n"
        if follow_UP:
            response += "**Follow-up Emails:**\n" + "\n".join(follow_UP) + "\n\n"
        if low_priority_keywords:
            response += "**Low Priority Emails:**\n" + "\n".join(low_priority_keywords) + "\n\n"
        
        if not response:
            response = "No emails found"
        
    else:
        retrieved_docs = whatsapp_db.similarity_search(user_input, k=5) if st.session_state.assistant_mode == "WhatsApp" else slack_db.similarity_search(user_input, k=5)
        chain = Chain_whatsapp if st.session_state.assistant_mode == "WhatsApp" else slack_chain
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt_input = f"""
        Based on the following messages:
        {context}
        
        User Query: {user_input}
        
        Please summarize or respond to the query appropriately.
    
     )
        
        """
        response = chain.invoke({"input": prompt_input})
    
    # Store messages in history
    st.session_state.history.append({"origin": "Human", "message": user_input})
    st.session_state.history.append({"origin": "RUBY", "message": response})



set_background()

# UI Design
st.title("🤖 RUBY - Your Personal Assistant")

# Display buttons to switch modes
col1, col2, col3 = st.columns(3)
col1.button("📧 Your Gmail Assistant", on_click=set_mode, args=("Gmail",))
col2.button("💬 Your WhatsApp Assistant", on_click=set_mode, args=("WhatsApp",))
col3.button("💼 Your Slack Assistant", on_click=set_mode, args=("Slack",))
st.subheader(f"**Currently Active:** {st.session_state.assistant_mode} Assistant")

# Display chat history
for chat in st.session_state.history:
    st.markdown(f"**{chat['origin']}:** {chat['message']}")

# Input field for user query
st.text_input("Ask RUBY anything:", key="user_input", on_change=process_query)
