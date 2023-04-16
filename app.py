import os
import gradio as gr
from dotenv import load_dotenv, find_dotenv
import openai

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe(audio_file, state=""):
    # print(audio_file)
    prompt = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
    with open(audio_file, "rb") as f:

        response = openai.Audio.transcribe("whisper-1", f, prompt=prompt)
        text = response['text']
    state += f"{text} "
    print(text)
    return state, state

app = gr.Interface(fn=transcribe, 
                   inputs=[gr.Audio(source="microphone", type="filepath"), "state"], 
                   outputs=["textbox", "state"])
app.launch(server_port=8080)