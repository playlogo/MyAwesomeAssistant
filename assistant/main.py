import os
from dotenv import load_dotenv

load_dotenv()


import openwakeword
from openwakeword.model import Model

import soundcard  # Awesome package we'll use for mic input!
import soundfile
import numpy
import time

from groq import Groq
import io

# Parameters
sensitivity = 0.3  # The higher the number - The more certain the model needs to be that it actually heard you say Alexa - Try tweaking it
sample_rate = 16000
chunk_size = 1280
record_duration = 3  # How long to record after a activation!

# Download the "alexa" model (no it's not from amazon :D)
openwakeword.utils.download_models(model_names=["alexa"])

# And load it
model = Model(wakeword_models=["alexa"])

# Create our sound input
mic = soundcard.default_microphone()

# New!! - Create a groq client - Replace this api key with your own (this one isn't valid!)
client = Groq(api_key=os.getenv("GROQ_KEY"))

# And open it!
with mic.recorder(samplerate=sample_rate, channels=1) as recorder:
    print("Now listening for wake-words!!! Try saying Alexa!")

    while True:
        # Get a super short chunk of audio 80ms from the mic
        data = recorder.record(numframes=chunk_size)
        samples = (data.flatten() * 32767).astype(numpy.int16)

        # See if we detect a activation in it
        prediction = model.predict(
            samples, debounce_time=3, threshold={"alexa": sensitivity}
        )

        if prediction["alexa"] > sensitivity:
            # If we got one - Print it!!!
            print("Got activation - Recording for 3 seconds!")

            # Record mic for 'record_duration' seconds
            recording = recorder.record(numframes=sample_rate * record_duration)

            # Converting stuff - Irrelevant
            wav_buf = io.BytesIO()
            wav_buf.name = "file.wav"
            soundfile.write(wav_buf, recording, samplerate=sample_rate)
            wav_buf.seek(0)

            # Sending it of to transcribe!
            print("Transcribing....")

            transcription = client.audio.transcriptions.create(
                file=("file.wav", wav_buf.read()),
                model="whisper-large-v3",
            )

            print(f"I've understood: {transcription.text}")

            chat_completion = client.chat.completions.create(
                messages=[
                    # You can modify this "system message" to tell model how to act or behave! Try adding "always respond with brainrot" - Play around with it!
                    {
                        "role": "system",
                        "content": "You are a helpful voice assistant. Please keep your responses super short - Two sentences maximum. You may be called Alexa.",
                    },
                    {
                        "role": "user",
                        "content": transcription.text,  # Our transcription from before!
                    },
                ],
                model="llama-3.3-70b-versatile",  # The model which will generate the response - Picked from here: https://console.groq.com/docs/models
            )

            print(chat_completion.choices[0].message.content)  # The response!!!

            # Let's now convert the response of the llm to speech
            print("Converting it to speech...")

            response = client.audio.speech.create(
                model="playai-tts",
                voice="Arista-PlayAI",
                input=chat_completion.choices[0].message.content,
                response_format="wav",
            )

            # And play it back through your speakers!
            playback_data, playback_sample_rate = soundfile.read(
                io.BytesIO(response.read())  # This is another conversion step!
            )

            default_speaker = soundcard.default_speaker()
            default_speaker.play(playback_data, samplerate=playback_sample_rate)

            print("Done!")
