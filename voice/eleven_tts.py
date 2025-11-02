import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from elevenlabs import ElevenLabs, play, save
from config import get_api_keys
from elevenlabs import stream
from flask import Response

keys = get_api_keys()         
API_KEY = keys["ELEVENLABS_API_KEY"]
VOICE_ID = "Aw6HJXoWrSqqRcFGkIYP"
#TEXT = "Its-a me, Mario! i love you you are the best yahoo! woooooww"
MODEL_ID= "eleven_v3"

elevenlabs = ElevenLabs(api_key=API_KEY)

def generate_tts(text):
    audio_stream = elevenlabs.text_to_speech.stream(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
    )
    return Response(audio_stream, mimetype="audio/wav")