from cartesia import Cartesia
import subprocess
import os

client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))

print("Starting ffplay to play streaming audio output...")
player = subprocess.Popen(
    ["ffplay", "-f", "f32le", "-ar", "44100", "-probesize", "32", "-analyzeduration", "0", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
    stdin=subprocess.PIPE,
    bufsize=0,
)

print("Connecting to Cartesia via websockets...")
with client.tts.websocket_connect() as connection:
    ctx = connection.context(
        model_id="sonic-3",
        voice={"mode": "id", "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02"},
        output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        },
    )

    print("Sending chunked text input...")
    for part in ["Hi there! ", "Welcome to ", "Cartesia Sonic."]:
        ctx.push(part)

    ctx.no_more_inputs()

    for response in ctx.receive():
        if response.type == "chunk" and response.audio:
            print(f"Received audio chunk ({len(response.audio)} bytes)")
            # Here we pipe audio to ffplay. In a production app you might play audio in
            # a client, or forward it to another service, eg, a telephony provider.
            player.stdin.write(response.audio)
        elif response.type == "done":
            break

player.stdin.close()
player.wait()