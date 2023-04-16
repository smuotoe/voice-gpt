import os
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from revChatGPT.V3 import Chatbot

load_dotenv(find_dotenv())


def chat_with_gpt(prompt):
    chatbot = Chatbot(api_key=os.getenv("OPENAI_API_KEY"))
    return chatbot.ask(prompt)


app = gr.Interface(
    fn=chat_with_gpt,
    inputs=gr.Textbox(lines=2, label="Prompt"),
    outputs=gr.Textbox(lines=2, label="Response"),
    title="Chat with GPT-4",
    description="Chat with GPT-4",
)
app.launch(server_port=8080)