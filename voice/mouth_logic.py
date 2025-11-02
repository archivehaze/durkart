from pydub import AudioSegment
import numpy as np

def generate_mouth_states(audio_path, chunk_ms=50, threshold_factor=0.2):
    audio = AudioSegment.from_wav(audio_path)
    samples = np.array(audio.get_array_of_samples())

    chunk_size = int(audio.frame_rate * chunk_ms / 1000)
    chunks = [samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size)]
    
    # Compute average volume for each chunk
    volumes = [np.abs(chunk).mean() for chunk in chunks]
    
    # Determine mouth state for each chunk
    threshold = max(volumes) * threshold_factor
    mouth_states = ['talking' if v > threshold else 'shut' for v in volumes]
    
    return mouth_states