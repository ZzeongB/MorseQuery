import os
import subprocess
import threading
import time

from cartesia import Cartesia

client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))

SAMPLE_RATE = 44100
BYTES_PER_SAMPLE = 4  # pcm_f32le
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE

print("Starting ffplay to play streaming audio output...")
player = subprocess.Popen(
    [
        "ffplay",
        "-f",
        "f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-fflags",
        "nobuffer",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
        "-",
    ],
    stdin=subprocess.PIPE,
    bufsize=0,
)

# ffplay에 써 넣은 오디오가 언제 끝날지 "추정"하기 위한 상태
playback_cursor_time = time.monotonic()
utterance_seq = 0
utterance_done = {}  # utt_id -> generation done?
utterance_playback_end = {}  # utt_id -> estimated playback end time


def register_audio_chunk(utt_id: int, num_bytes: int):
    """이 chunk가 재생 큐에 추가되었을 때, 예상 playback end time 갱신."""
    global playback_cursor_time

    now = time.monotonic()
    duration_sec = num_bytes / BYTES_PER_SECOND

    # 이미 밀려 있으면 그 뒤에 붙고, 비어 있으면 지금부터 재생된다고 가정
    start_time = max(now, playback_cursor_time)
    end_time = start_time + duration_sec
    playback_cursor_time = end_time

    utterance_playback_end[utt_id] = end_time


def is_playback_finished(utt_id: int) -> bool:
    """해당 utterance의 생성도 끝났고, 추정 playback end 시점도 지났는지."""
    if not utterance_done.get(utt_id, False):
        return False

    end_time = utterance_playback_end.get(utt_id)
    if end_time is None:
        return False

    return time.monotonic() >= end_time


def wait_for_playback_finish(utt_id: int, poll_interval: float = 0.01):
    while not is_playback_finished(utt_id):
        time.sleep(poll_interval)
    print(f"[utt {utt_id}] playback finished")


print("Connecting to Cartesia via websockets...")
with client.tts.websocket_connect() as connection:
    ctx = connection.context(
        model_id="sonic-3",
        voice={"mode": "id", "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02"},
        output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": SAMPLE_RATE,
        },
    )

    # 예시: 하나의 utterance
    utterance_seq += 1
    utt_id = utterance_seq
    utterance_done[utt_id] = False

    print(f"Sending chunked text input for utterance {utt_id}...")
    for part in ["Hi there! ", "Welcome to ", "Cartesia Sonic."]:
        ctx.push(part)

    ctx.no_more_inputs()

    for response in ctx.receive():
        if response.type == "chunk" and response.audio:
            num_bytes = len(response.audio)
            print(f"[utt {utt_id}] Received audio chunk ({num_bytes} bytes)")
            player.stdin.write(response.audio)
            register_audio_chunk(utt_id, num_bytes)

        elif response.type == "done":
            utterance_done[utt_id] = True
            print(f"[utt {utt_id}] generation done")
            threading.Thread(
                target=wait_for_playback_finish,
                args=(utt_id,),
                daemon=True,
            ).start()
            break

print("Stream still open if you want to keep sending more text later.")
# 세션 전체 종료 시에만 아래 호출
player.stdin.close()
player.wait()
