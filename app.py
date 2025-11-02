from flask import Flask, render_template, request, send_file, jsonify
from config import get_api_keys
from elevenlabs import ElevenLabs, stream
from voice.mouth_logic import generate_mouth_states
from voice.eleven_tts import generate_tts


from voice.eleven_tts import elevenlabs, VOICE_ID, MODEL_ID
import io
import tempfile

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# Load API key
keys = get_api_keys()
API_KEY = keys["ELEVENLABS_API_KEY"]
VOICE_ID = "Aw6HJXoWrSqqRcFGkIYP"
MODEL_ID = "eleven_v3"

client = ElevenLabs(api_key=API_KEY)

# Route to return TTS audio for hardcoded text
@app.route('/speak')
def speak():
    # Hardcoded text for now
    text = "Its-a me, Mario! This is a test! I love eating ice cream and ring around the rosies, princess peach is predicted to win Woohoo!"
    audio_stream = elevenlabs.text_to_speech.stream(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID
    )

    # Save the stream to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with open(tmp_file.name, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)
    
    # Serve the file to the browser
    return send_file(tmp_file.name, mimetype="audio/wav")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        if not text:
            return render_template("index.html", error="Please enter some text.")

        # Generate audio stream
        audio_stream = client.text_to_speech.stream(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
        )

        # Save audio to in-memory file
        audio_bytes = io.BytesIO()
        for chunk in audio_stream:
            audio_bytes.write(chunk)
        audio_bytes.seek(0)

        return send_file(
            audio_bytes,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="tts.wav"
        )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)