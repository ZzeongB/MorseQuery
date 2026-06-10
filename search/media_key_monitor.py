"""
macOS Media Key Monitor for Sony WH-1000XM5 Swipe Gestures

Sony XM5 sends AVRCP commands via Bluetooth.
We need to register as a "Now Playing" app to receive these events.

Requires: pip install pyobjc-framework-MediaPlayer pyobjc-framework-Cocoa
"""

import signal
import sys
from Foundation import NSRunLoop, NSDate
from AppKit import NSApplication
import MediaPlayer


class MediaKeyMonitor:
    """Monitor for Sony XM5 headphone swipe gestures via MPRemoteCommandCenter."""

    def __init__(self, on_swipe_left=None, on_swipe_right=None, on_play_pause=None):
        self.on_swipe_left = on_swipe_left
        self.on_swipe_right = on_swipe_right
        self.on_play_pause = on_play_pause
        self.command_center = None

    def _setup_remote_commands(self):
        """Register for remote control events using blocks."""
        self.command_center = MediaPlayer.MPRemoteCommandCenter.sharedCommandCenter()

        # Previous track (Swipe Left)
        def handle_previous(event):
            print("[Remote] Previous Track (Swipe Left)")
            if self.on_swipe_left:
                self.on_swipe_left()
            return MediaPlayer.MPRemoteCommandHandlerStatusSuccess

        self.command_center.previousTrackCommand().setEnabled_(True)
        self.command_center.previousTrackCommand().addTargetWithHandler_(
            handle_previous
        )

        # Next track (Swipe Right)
        def handle_next(event):
            print("[Remote] Next Track (Swipe Right)")
            if self.on_swipe_right:
                self.on_swipe_right()
            return MediaPlayer.MPRemoteCommandHandlerStatusSuccess

        self.command_center.nextTrackCommand().setEnabled_(True)
        self.command_center.nextTrackCommand().addTargetWithHandler_(handle_next)

        # Play/Pause (Tap)
        def handle_toggle(event):
            print("[Remote] Play/Pause (Tap)")
            if self.on_play_pause:
                self.on_play_pause()
            return MediaPlayer.MPRemoteCommandHandlerStatusSuccess

        self.command_center.togglePlayPauseCommand().setEnabled_(True)
        self.command_center.togglePlayPauseCommand().addTargetWithHandler_(
            handle_toggle
        )

        # Play
        def handle_play(event):
            print("[Remote] Play")
            if self.on_play_pause:
                self.on_play_pause()
            return MediaPlayer.MPRemoteCommandHandlerStatusSuccess

        self.command_center.playCommand().setEnabled_(True)
        self.command_center.playCommand().addTargetWithHandler_(handle_play)

        # Pause
        def handle_pause(event):
            print("[Remote] Pause")
            if self.on_play_pause:
                self.on_play_pause()
            return MediaPlayer.MPRemoteCommandHandlerStatusSuccess

        self.command_center.pauseCommand().setEnabled_(True)
        self.command_center.pauseCommand().addTargetWithHandler_(handle_pause)

        # Set up "Now Playing" info so macOS recognizes us as active player
        now_playing = MediaPlayer.MPNowPlayingInfoCenter.defaultCenter()
        now_playing.setNowPlayingInfo_(
            {
                MediaPlayer.MPMediaItemPropertyTitle: "MorseQuery",
                MediaPlayer.MPMediaItemPropertyArtist: "Listening...",
                MediaPlayer.MPNowPlayingInfoPropertyPlaybackRate: 1.0,
            }
        )

    def start(self):
        """Start monitoring media keys."""
        print("=" * 50)
        print("Sony XM5 Swipe Gesture Monitor (AVRCP)")
        print("=" * 50)
        print("Registering as Now Playing app...")

        # Initialize NSApplication (required for event handling)
        NSApplication.sharedApplication()

        self._setup_remote_commands()

        print("Listening for swipe gestures...")
        print("  - Swipe Left  -> Previous Track")
        print("  - Swipe Right -> Next Track")
        print("  - Tap         -> Play/Pause")
        print("")
        print("Press Ctrl+C to stop")
        print("=" * 50)

        # Run the event loop
        try:
            while True:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    "kCFRunLoopDefaultMode", NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop monitoring."""
        print("\n[Stopping] Media key monitor...")
        now_playing = MediaPlayer.MPNowPlayingInfoCenter.defaultCenter()
        now_playing.setNowPlayingInfo_(None)


if __name__ == "__main__":

    def on_swipe_left():
        print("  -> SWIPE LEFT!")

    def on_swipe_right():
        print("  -> SWIPE RIGHT!")

    def on_play_pause():
        print("  -> PLAY/PAUSE!")

    def signal_handler(sig, frame):
        print("\n[Interrupted] Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    monitor = MediaKeyMonitor(
        on_swipe_left=on_swipe_left,
        on_swipe_right=on_swipe_right,
        on_play_pause=on_play_pause,
    )
    monitor.start()
