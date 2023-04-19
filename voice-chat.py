import os
import gradio as gr
from dotenv import load_dotenv, find_dotenv
import openai
from revChatGPT.V3 import Chatbot

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe(audio_file):
    # print(audio_file)
    prompt = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. The author of this tool is Somto Muotoe. Contributors: Ireoluwa Enoch Adedugbe; Abiola Aderiye"
    with open(audio_file, "rb") as f:

        response = openai.Audio.transcribe("whisper-1", f, prompt=prompt)
        text = response['text']
    print(text)
    return text

def chat_with_gpt(prompt):
    chatbot = Chatbot(api_key=os.getenv("OPENAI_API_KEY"))
    return chatbot.ask(prompt)

def transcribe_and_chat(audio_file):
    text = transcribe(audio_file)
    gpt_response = chat_with_gpt(text)
    return text, gpt_response


app = gr.Interface(
    fn=transcribe_and_chat,
    inputs=[
        gr.Audio(source="microphone", type="filepath"), 
        # gr.Textbox(lines=2, label="Prompt"),
    ],
    outputs=[
        gr.Textbox(lines=2, label="Transcription"),
        gr.Textbox(lines=2, label="Response"), 
    ],
        
    title="Voice Chat with GPT-4",
    description="Chat with GPT-4 Using your Voice!",
    # article="<h1>Voice Chat with GPT-4</h1><p>Chat with GPT-4 Using your Voice!</p>",
    # examples=["Hi, how are you?", "Is the earth the only planet with living things?"],
    cache_examples=True

)

app.launch(server_port=8080, share=True)