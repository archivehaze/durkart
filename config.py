from dotenv import load_dotenv
import os

load_dotenv()
def get_api_keys():
    return {
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY")
    }