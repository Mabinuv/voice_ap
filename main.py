import assemblyai as aai

from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from openai import OpenAI

from dotenv import load_dotenv
import os
import logging

import ssl
import certifi
import openai

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent
)

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile


ssl_context = ssl.create_default_context(cafile=certifi.where())

load_dotenv()
logging.basicConfig(level=logging.INFO)


# API Keys from .env
ASSEMBLY_API_KEY = os.getenv("aai_pi_key")
OPENAI_API_KEY = os.getenv("openAI_api_key")
ELEVENLABS_API_KEY = os.getenv("elevenlabs_api_key")

# Global state
vector_store = None
qa_chain = None

# PDF processing + RAG setup
def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever
    )
    return vectordb, chain

class ai_assistant:
    def __init__(self):
        openai.api_key = os.getenv("openAI_api_key")
        self.openai_client = openai
        self.client = ElevenLabs(api_key=os.getenv('elevenlabs_api_key'))
     
        self.full_transcript = [
            {
                "role": "system",
                "content": (
                    "You are customer support assistance for a bank."
                    "You are supposed be polite and act very understanding to cater the users and give appropriate answers to bank users."
                ),
            }
        ]

  
    
    def generate_ai_response(self, user_text):

        self.full_transcript.append({"role": "user", "content": user_text})
        print(f"\nUser: {user_text}", end="\r\n")

        if qa_chain:
            ai_response = qa_chain.run(user_text)
        else:
            ai_response = "No document uploaded yet. Please upload a PDF."
        
        # response = self.openai_client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=self.full_transcript,
        # )

        # ai_response = response.choices[0].message.content

        self.full_transcript.append({"role": "assistant", "content": ai_response})
        print(f"\nAI Customer Assist: {ai_response}", end="\r\n")
        self.generate_audio(ai_response)


    def generate_audio(self, text):

        audio_stream = self.client.text_to_speech.stream(
            text=text,
            voice_id="NaKPQmdr7mMxXuXrNeFC",
            model_id="eleven_multilingual_v2"
        )

        stream(audio_stream)

     # AssemblyAI event handlers
    def on_begin(self, client, event: BeginEvent):
        print(f"🟢 Session started: {event.id}")

    def on_turn(self, client, event: TurnEvent):
        print(f"🗣️ {event.transcript}")
        if event.end_of_turn:
            self.generate_ai_response(event.transcript)

            if not event.turn_is_formatted:
                self.client_instance.set_params(
                    StreamingSessionParameters(format_turns=True)
                )

    def on_terminated(self, client, event: TerminationEvent):
        print(f"🔴 Session ended. Duration: {event.audio_duration_seconds}s")

    def on_error(self, client, error: StreamingError):
        print(f"❌ Error: {error}")

    def start_transcription(self):
        self.client_instance = StreamingClient(
            StreamingClientOptions(api_key=ASSEMBLY_API_KEY)
        )

        # Bind events
        self.client_instance.on(StreamingEvents.Begin, self.on_begin)
        self.client_instance.on(StreamingEvents.Turn, self.on_turn)
        self.client_instance.on(StreamingEvents.Termination, self.on_terminated)
        self.client_instance.on(StreamingEvents.Error, self.on_error)

        # Start streaming
        self.client_instance.connect(
            StreamingParameters(sample_rate=16000, format_turns=True)
        )
        try:
            self.client_instance.stream(
                aai.extras.MicrophoneStream(sample_rate=16000)
            )
        finally:
            self.client_instance.disconnect(terminate=True)
    
     
# Streamlit UI
st.title("📞 Voice + Chat Bank Assistant (RAG + Stream)")
st.markdown("Upload a PDF document for the assistant to answer based on its content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        # global vector_store, qa_chain
        vector_store, qa_chain = process_pdf(tmp_file.name)
    st.success("✅ Document uploaded and RAG pipeline ready.")

assistant = ai_assistant()

if st.button("🎙️ Start Voice Assistant"):
    assistant.generate_audio("Welcome to Federal Bank. How may I help you today?")
    assistant.start_transcription()

user_input = st.text_input("💬 Ask a question:")
if user_input:
    assistant.generate_ai_response(user_input)

