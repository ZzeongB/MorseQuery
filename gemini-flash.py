# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss pydub
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones.

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```

To stream audio from an mp3 file:
```
python gemini-flash.py --mode mp3 --file audio.mp3
```

To stream audio and video from an mp4 file:
```
python gemini-flash.py --mode mp4 --file video.mp4
```

To use study mode (real-time transcription with playback, summaries, and term explanations):
```
python gemini-flash.py --mode study --file lecture.mp3
python gemini-flash.py --mode study --file video.mp4
```
"""

import argparse
import asyncio
import base64
import io
import os
import sys
import traceback
import wave
from datetime import datetime

import cv2
import mss
import PIL.Image
import pyaudio
from google import genai
from pydub import AudioSegment

if sys.version_info < (3, 11, 0):
    import exceptiongroup

    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
# STUDY_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

MODEL = "models/gemini-2.0-flash-exp"
STUDY_MODEL = "models/gemini-2.0-flash-exp"

DEFAULT_MODE = "none"
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(http_options={"api_version": "v1beta"}, api_key=API_KEY)

from google.genai import types

grounding_tool = types.Tool(google_search=types.GoogleSearch())

tools = [{"google_search": {}}]

CONFIG = {"response_modalities": ["AUDIO"]}

STUDY_CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": """You are a real-time learning assistant. Listen to the audio and perform the following tasks:

1. **[Captions]**: Transcribe what you hear in real time (verbatim, in the original language).
2. **[Summary]**: Continuously update the summary with TWO parts:
   - **Overall context**: a brief one-line summary of everything understood so far.
   - **Current segment**: a one-line summary of what is being discussed right now.
   Update both as new information comes in.

3. **[Terms]**: When technical terms, difficult words, or important concepts appear, briefly explain them.
   Always provide at least THREE terms when possible.

Output format:
[Captions] Transcribed content...
[Summary] Key point summary...
[Terms] Term: explanation...

Be concise and respond in real time.""",
    "tools": tools,
}

pya = pyaudio.PyAudio()

import re
from typing import Dict, List

SECTION_RE = re.compile(r"\[(Captions|Summary|Terms)\]", re.IGNORECASE)


def _norm_ws(s: str) -> str:
    """Collapse weird newlines/spaces into readable single spaces."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # keep newlines for section splitting, but normalize inside fields later
    return s


def _squash_inline(s: str) -> str:
    """Turn arbitrary newlines into spaces, collapse repeated whitespace."""
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)  # newline -> space
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _parse_terms_block(text: str) -> List[Dict[str, str]]:
    """
    Parse a terms block where entries are like:
    Term: Amygdala - A brain structure...
    but newlines can appear anywhere.
    """
    if not text:
        return []

    t = _norm_ws(text)

    # Robustly match Term entries across arbitrary newlines
    # - Term label may be 'Term:' possibly with extra spaces/newlines
    # - Separator between term and definition may be ' - ' or ':' (fallback)
    term_pat = re.compile(
        r"""
        Term\s*:\s*
        (?P<term>.+?)
        \s*(?:-\s*|:\s*)
        (?P<defn>.+?)
        (?=(?:\n?\s*Term\s*:|\Z))
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    terms: List[Dict[str, str]] = []
    for m in term_pat.finditer(t):
        term = _squash_inline(m.group("term"))
        definition = _squash_inline(m.group("defn"))
        if term and definition:
            terms.append({"term": term, "definition": definition})

    return terms


def parse_realtime_output(raw: str) -> Dict[str, object]:
    """
    Parse model output that contains:
      [Captions] ...
      [Summary] Overall context: ... Current segment: ...
      [Terms] Term: ... - ...
    while being resilient to arbitrary newlines.
    """
    raw = _norm_ws(raw)

    # Split into sections by markers, but keep any preamble text
    matches = list(SECTION_RE.finditer(raw))
    sections: Dict[str, str] = {"captions": "", "summary": "", "terms": ""}
    preamble = raw[: matches[0].start()] if matches else raw

    # Collect section contents
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()

        if name == "captions":
            sections["captions"] += (
                ("\n" + content) if sections["captions"] else content
            )
        elif name == "summary":
            sections["summary"] += ("\n" + content) if sections["summary"] else content
        elif name == "terms":
            sections["terms"] += ("\n" + content) if sections["terms"] else content

    # Parse captions (join lines robustly)
    captions = _squash_inline(sections["captions"])

    # Parse summary (handle broken labels like "Current\n segment")
    summary_text = sections["summary"]
    summary_text_inline = _squash_inline(summary_text)

    overall = ""
    current = ""

    m_overall = re.search(
        r"Overall\s*context\s*:\s*(.+?)(?=(Current\s*segment\s*:|$))",
        summary_text_inline,
        flags=re.IGNORECASE,
    )
    if m_overall:
        overall = _squash_inline(m_overall.group(1))

    m_current = re.search(
        r"Current\s*segment\s*:\s*(.+)$", summary_text_inline, flags=re.IGNORECASE
    )
    if m_current:
        current = _squash_inline(m_current.group(1))

    # Parse terms: include preamble terms + [Terms] block terms
    terms = []
    terms.extend(_parse_terms_block(preamble))
    terms.extend(_parse_terms_block(sections["terms"]))

    return {
        "captions": captions,
        "summary": {
            "overall_context": overall,
            "current_segment": current,
        },
        "terms": terms,
        "raw_sections": sections,  # 디버깅 필요 없으면 제거해도 됨
    }


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, file_path=None):
        self.video_mode = video_mode
        self.file_path = file_path

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        # Output audio file
        self.output_wav = None
        self.output_filename = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def listen_audio_from_file(self):
        """Read audio from mp3/mp4 file and stream it"""
        print(f"Loading audio from: {self.file_path}")

        # Load audio file using pydub (supports mp3, mp4, wav, etc.)
        audio = await asyncio.to_thread(AudioSegment.from_file, self.file_path)

        # Convert to 16kHz mono PCM (required format for Gemini)
        audio = audio.set_frame_rate(SEND_SAMPLE_RATE).set_channels(CHANNELS)
        raw_data = audio.raw_data

        print(f"Audio loaded: {len(audio)}ms, streaming...")

        # Stream audio in chunks
        offset = 0
        bytes_per_chunk = CHUNK_SIZE * 2  # 16-bit = 2 bytes per sample

        while offset < len(raw_data):
            chunk = raw_data[offset : offset + bytes_per_chunk]
            if len(chunk) < bytes_per_chunk:
                # Pad the last chunk with silence if needed
                chunk = chunk + b"\x00" * (bytes_per_chunk - len(chunk))

            await self.out_queue.put({"data": chunk, "mime_type": "audio/pcm"})
            offset += bytes_per_chunk

            # Simulate real-time playback speed
            await asyncio.sleep(CHUNK_SIZE / SEND_SAMPLE_RATE)

        print("Audio file streaming complete.")

    async def listen_audio_from_file_with_playback(self):
        """Read audio from file, play through speakers, and stream to Gemini simultaneously"""
        print(f"Loading audio from: {self.file_path}")

        # Load audio file using pydub
        audio = await asyncio.to_thread(AudioSegment.from_file, self.file_path)

        # Convert to 16kHz mono PCM (required format for Gemini)
        audio = audio.set_frame_rate(SEND_SAMPLE_RATE).set_channels(CHANNELS)
        raw_data = audio.raw_data

        print(f"Audio loaded: {len(audio)}ms, streaming with playback...")

        # Open playback stream
        play_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            output=True,
        )

        # Stream audio in chunks
        offset = 0
        bytes_per_chunk = CHUNK_SIZE * 2  # 16-bit = 2 bytes per sample

        try:
            while offset < len(raw_data):
                chunk = raw_data[offset : offset + bytes_per_chunk]
                if len(chunk) < bytes_per_chunk:
                    # Pad the last chunk with silence if needed
                    chunk = chunk + b"\x00" * (bytes_per_chunk - len(chunk))

                # Play through speakers
                await asyncio.to_thread(play_stream.write, chunk)
                # Send to Gemini
                await self.out_queue.put({"data": chunk, "mime_type": "audio/pcm"})
                offset += bytes_per_chunk

                # Small sleep to prevent blocking
                await asyncio.sleep(0.001)
        finally:
            play_stream.stop_stream()
            play_stream.close()

        print("Audio file streaming complete.")

    async def get_frames_from_video(self):
        """Read frames from mp4 video file"""
        print(f"Loading video from: {self.file_path}")

        cap = await asyncio.to_thread(cv2.VideoCapture, self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0  # Send 1 frame per second to match other modes

        frame_count = 0
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            frame_count += 1
            await self.out_queue.put(frame)
            await asyncio.sleep(frame_interval)

        cap.release()
        print(f"Video streaming complete. Total frames sent: {frame_count}")

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def receive_text(self):
        """Background task to receive text responses for study mode"""
        while True:
            try:
                turn = self.session.receive()
                async for response in turn:
                    try:
                        # 텍스트 출력
                        text = getattr(response, "text", None)
                        if text:
                            self._print_formatted(text)

                        # print(response.__dict__)

                        # 후보가 없는 경우 대비
                        candidates = getattr(response, "candidates", None)
                        if not candidates:
                            continue

                        print("Yes candidate")

                        candidate = candidates[0]

                        # grounding metadata 안전 접근
                        metadata = getattr(candidate, "grounding_metadata", None)
                        if metadata:
                            supports = getattr(metadata, "grounding_supports", None)
                            chunks = getattr(metadata, "grounding_chunks", None)
                            if supports and chunks:
                                print(supports, chunks)

                    except Exception as e:
                        # 개별 response 처리 중 에러
                        print(f"[receive_text] response handling error: {e}")
                        continue

            except Exception as e:
                # receive() 자체가 실패한 경우
                print(f"[receive_text] session receive error: {e}")
                await asyncio.sleep(0.5)  # 과도한 루프 방지

    def _print_formatted(self, text):
        """Print text with ANSI colors based on tag type"""
        # ANSI color codes
        WHITE = "\033[97m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        lines = text.split("\n")
        for line in lines:
            if "[Captions]" in line:
                print(f"{WHITE}{line}{RESET}")
            elif "[Summary]" in line:
                print(f"{YELLOW}{line}{RESET}")
            elif "[Terms]" in line:
                print(f"{GREEN}{line}{RESET}")
            else:
                print(line)

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        # Create output wav file
        self.output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        self.output_wav = wave.open(self.output_filename, "wb")
        self.output_wav.setnchannels(CHANNELS)
        self.output_wav.setsampwidth(2)  # 16-bit = 2 bytes
        self.output_wav.setframerate(RECEIVE_SAMPLE_RATE)
        print(f"Recording output to: {self.output_filename}")

        while True:
            bytestream = await self.audio_in_queue.get()
            # Save to file
            self.output_wav.writeframes(bytestream)
            # Play through speaker
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            # Choose model and config based on mode
            if self.video_mode == "study":
                model = STUDY_MODEL
                config = STUDY_CONFIG
            else:
                model = MODEL
                config = CONFIG

            async with (
                client.aio.live.connect(model=model, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())

                # Choose audio source based on mode
                if self.video_mode == "study":
                    tg.create_task(self.listen_audio_from_file_with_playback())
                elif self.video_mode in ("mp3", "mp4"):
                    tg.create_task(self.listen_audio_from_file())
                else:
                    tg.create_task(self.listen_audio())

                # Choose video source based on mode
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())
                elif self.video_mode == "mp4":
                    tg.create_task(self.get_frames_from_video())

                # Choose response handler based on mode
                if self.video_mode == "study":
                    tg.create_task(self.receive_text())
                    # No play_audio() for study mode - audio plays directly in listen_audio_from_file_with_playback()
                else:
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)
        finally:
            # Close output wav file
            if self.output_wav:
                self.output_wav.close()
                print(f"\nOutput audio saved to: {self.output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="source mode: camera, screen, none, mp3, mp4, or study",
        choices=["camera", "screen", "none", "mp3", "mp4", "study"],
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="path to mp3 or mp4 file (required for mp3/mp4 modes)",
    )
    args = parser.parse_args()

    # Validate file argument for mp3/mp4/study modes
    if args.mode in ("mp3", "mp4", "study") and not args.file:
        parser.error(f"--file is required when using --mode {args.mode}")

    main = AudioLoop(video_mode=args.mode, file_path=args.file)
    asyncio.run(main.run())
