import asyncio

from mudra_sdk import Mudra, MudraDevice
from mudra_sdk.models.callbacks import MudraDelegate

# Create Mudra instance
# mudra = Mudra()

# Retrieve license from cloud (required for full functionality)
# mudra.get_license_for_email_from_cloud("parkjeong02@gmail.com")

discovered_devices = []

import sys
from pathlib import Path

from mudra_sdk.models.enums import FirmwareTarget, HandType

# Add parent directory to path so mudra_sdk can be imported
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import platform
import tkinter as tk
from threading import Thread
from tkinter import ttk

# Global reference to the event loop and thread
loop = None
loop_thread = None

# Keep the actual MudraDevice objects so we can connect/disconnect to them
devices_list = []

# Dictionary to store status indicators (initialized in main)
status_indicators = {}

# Navigation smoothing (moving average)
navigation_history = {"x": [], "y": []}
NAVIGATION_SMOOTHING_WINDOW = 5  # Number of samples to average


def update_devices_list(device):
    print(device.name)
    devices_list.append(device)
    listbox.after(0, refresh_listbox)


mudra = Mudra()


class MyMudraDelegate(MudraDelegate):
    def on_device_discovered(self, device: MudraDevice):
        print(f"Discovered: {device.name}")
        update_devices_list(device)

    def on_mudra_device_disconnected(self, device: MudraDevice):
        print(f"Device disconnected: {device.name}")

    def on_mudra_device_disconnecting(self, device: MudraDevice):
        print(f"Device disconnecting: {device.name}")

    def on_mudra_device_connected(self, device: MudraDevice):
        print(f"Device connected: {device.name}")

    def on_mudra_device_connecting(self, device: MudraDevice):
        print(f"Device connecting: {device.name}")

    def on_mudra_device_connection_failed(self, device: MudraDevice, error: str):
        print(f"Connection failed: {device.name}, Error: {error}")

    def on_bluetooth_state_changed(self, state: bool):
        print(f"Bluetooth state changed: {'On' if state else 'Off'}")


mudra.set_delegate(MyMudraDelegate())
mudra.get_license_for_email_from_cloud("parkjeong02@gmail.com")


def refresh_listbox():
    listbox.delete(0, tk.END)
    for dev in devices_list:
        listbox.insert(tk.END, dev.name)


def run_event_loop():
    """Run the event loop in a separate thread."""
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_forever()


def ensure_event_loop_running():
    """Ensure the background asyncio event loop is running."""
    global loop, loop_thread
    if loop_thread is None or not loop_thread.is_alive():
        loop_thread = Thread(target=run_event_loop, daemon=True)
        loop_thread.start()
        # Wait a bit for the loop to start
        import time

        time.sleep(0.1)


def start_scan():
    # Ensure event loop is running
    ensure_event_loop_running()

    # Schedule the scan on the event loop
    if loop is not None:
        asyncio.run_coroutine_threadsafe(mudra.scan(), loop)


def stop_scan():
    global loop
    # Stop scanning using the same event loop
    if loop is not None:
        asyncio.run_coroutine_threadsafe(mudra.stop_scan(), loop)


def get_selected_device():
    """Return the currently selected MudraDevice from the listbox, or None."""
    selection = listbox.curselection()
    if not selection:
        return None
    index = selection[0]
    if 0 <= index < len(devices_list):
        return devices_list[index]
    return None


def connect_selected_device():
    """Connect to the selected device."""
    global loop
    device = get_selected_device()
    if device is None:
        print("No device selected to connect.")
        return

    ensure_event_loop_running()

    if loop is not None:
        asyncio.run_coroutine_threadsafe(device.connect(), loop)


def disconnect_selected_device():
    """Disconnect from the selected device."""
    global loop
    device = get_selected_device()
    if device is None:
        print("No device selected to disconnect.")
        return

    ensure_event_loop_running()

    if loop is not None:
        asyncio.run_coroutine_threadsafe(device.disconnect(), loop)


def discover_selected_device():
    """Discover GATT services/characteristics on the selected device."""
    global loop
    device = get_selected_device()
    if device is None:
        print("No device selected to discover.")
        return

    ensure_event_loop_running()

    if loop is not None:
        future = asyncio.run_coroutine_threadsafe(
            mudra.ble_service.discover_services_and_characteristics(device),
            loop,
        )

        # Log completion or errors from the background task
        def _on_done(fut: asyncio.Future):
            try:
                fut.result()
                print("Discovery completed.")
            except Exception as e:
                print(f"Discovery failed with error: {e}")

        future.add_done_callback(_on_done)


def on_snc_ready(timestamp, data_list, frequency, frequency_std, rms_list):
    print(f"SNC frequency: {frequency}")

    # Expecting rms_list to contain 3 float samples in [0.0, 1.0].
    # Safely update the 3 progress bars on the Tkinter main thread.
    def _update_rms_bars():
        if not root.winfo_exists():
            return

        # Pad/trim to exactly 3 values
        values = (list(rms_list) + [0.0, 0.0, 0.0])[:3]

        for i, (bar, label, value) in enumerate(
            zip(rms_bars, rms_labels, values), start=1
        ):
            # Clamp to [0.0, 1.0]
            value = max(0.0, min(1.0, float(value)))
            bar.config(value=value * 100.0)
            label.config(text=f"RMS {i}: {value:.2f}")

    # Ensure UI update is done in the main thread
    if root.winfo_exists():
        root.after(0, _update_rms_bars)


def on_imu_acc_ready(timestamp, data_list, frequency, frequency_std, rms_list):
    print(f"IMU Acc frequency: {frequency}")

    # Update the IMU Acc label on the Tkinter main thread
    def _update_imu_acc_label():
        if not root.winfo_exists():
            return
        imu_acc_label.config(text=f"IMU Acc Frequency: {frequency:.2f} Hz")

    if root.winfo_exists():
        root.after(0, _update_imu_acc_label)


def on_imu_gyro_ready(timestamp, data_list, frequency, frequency_std, rms_list):
    print(f"IMU Gyro frequency: {frequency}")

    # Update the IMU Gyro label on the Tkinter main thread
    def _update_imu_gyro_label():
        if not root.winfo_exists():
            return
        imu_gyro_label.config(text=f"IMU Gyro Frequency: {frequency:.2f} Hz")

    if root.winfo_exists():
        root.after(0, _update_imu_gyro_label)


def on_navigation_ready(delta_x, delta_y):
    print(f"Navigation delta: {delta_x}, {delta_y}")

    # Apply smoothing using moving average
    global navigation_history
    navigation_history["x"].append(float(delta_x))
    navigation_history["y"].append(float(delta_y))

    # Keep only the last N samples
    if len(navigation_history["x"]) > NAVIGATION_SMOOTHING_WINDOW:
        navigation_history["x"].pop(0)
        navigation_history["y"].pop(0)

    # Calculate smoothed values
    smoothed_x = sum(navigation_history["x"]) / len(navigation_history["x"])
    smoothed_y = sum(navigation_history["y"]) / len(navigation_history["y"])

    # Update the Navigation label on the Tkinter main thread
    def _update_navigation_label():
        if not root.winfo_exists():
            return
        navigation_label.config(
            text=f"Navigation: ΔX={smoothed_x:.3f}, ΔY={smoothed_y:.3f}"
        )

    if root.winfo_exists():
        root.after(0, _update_navigation_label)


def on_pressure_ready(pressure_data):
    # Update the pressure_bar on the Tkinter main thread
    def _update_pressure_bar():
        if not root.winfo_exists():
            return

        # Incoming pressure is in [0, 100]; clamp to that range
        value = max(0.0, min(100.0, float(pressure_data)))
        pressure_bar.config(value=value)
        pressure_label.config(text=f"Pressure: {value:.2f}")

    if root.winfo_exists():
        root.after(0, _update_pressure_bar)


def on_gesture_ready(gesture_type):
    print(f"Gesture received: {gesture_type}")

    # Update the Gesture label on the Tkinter main thread
    def _update_gesture_label():
        if not root.winfo_exists():
            return
        gesture_label.config(text=f"Gesture: {gesture_type}")

    if root.winfo_exists():
        root.after(0, _update_gesture_label)


def on_airmouse_button_changed_ready(air_touch_button):
    print(f"Air Touch Button changed: {air_touch_button}")

    # Update the Air Touch Button label on the Tkinter main thread
    def _update_air_touch_button_label():
        if not root.winfo_exists():
            return
        air_touch_button_label.config(text=f"Air Touch Button: {air_touch_button}")

    if root.winfo_exists():
        root.after(0, _update_air_touch_button_label)


def enable_snc_feature():
    """Placeholder to enable a feature on the selected device."""
    device = get_selected_device()
    if device is None:
        print("No device selected to enable feature.")
        return
    print(f"Enable feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_snc_ready(on_snc_ready), loop)


def disable_snc_feature():
    """Placeholder to disable a feature on the selected device."""
    device = get_selected_device()
    if device is None:
        print("No device selected to disable feature.")
        return
    print(f"Disable feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_snc_ready(None), loop)


def enable_imu_acc_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable IMU Acc feature.")
        return
    print(f"Enable IMU Acc feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_on_imu_acc_ready(on_imu_acc_ready), loop
    )


def disable_imu_acc_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable IMU Acc feature.")
        return
    print(f"Disable IMU Acc feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_imu_acc_ready(None), loop)


def enable_imu_gyro_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable IMU Gyro feature.")
        return
    print(f"Enable IMU Gyro feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_on_imu_gyro_ready(on_imu_gyro_ready), loop
    )


def disable_imu_gyro_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable IMU Gyro feature.")
        return
    print(f"Disable IMU Gyro feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_imu_gyro_ready(None), loop)


def enable_pressure_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Pressure feature.")
        return
    print(f"Enable Pressure feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_on_pressure_ready(on_pressure_ready), loop
    )


def disable_pressure_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Pressure feature.")
        return
    print(f"Disable Pressure feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_pressure_ready(None), loop)


def enable_navigation_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Navigation feature.")
        return
    print(f"Enable Navigation feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_on_navigation_ready(on_navigation_ready), loop
    )


def disable_navigation_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Navigation feature.")
        return
    print(f"Disable Navigation feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_navigation_ready(None), loop)


def enable_gesture_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Gesture feature.")
        return
    print(f"Enable Gesture feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_on_gesture_ready(on_gesture_ready), loop
    )


def disable_gesture_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Gesture feature.")
        return
    print(f"Disable Gesture feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_gesture_ready(None), loop)


def enable_air_mouse_button_changed_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Air Touch Button Changed feature.")
        return
    print(f"Enable Air Touch Button Changed feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_on_button_changed(on_airmouse_button_changed_ready), loop
    )


def disable_air_mouse_button_changed_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Air Touch Button Changed feature.")
        return
    print(f"Disable Air Touch Button Changed feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_on_button_changed(None), loop)


def enable_embedded_airtouch_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Embedded AirTouch feature.")
        return
    print(f"Enable Embedded AirTouch feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_air_touch_active(True), loop)


def disable_embedded_airtouch_feature():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Embedded AirTouch feature.")
        return
    print(f"Disable Embedded AirTouch feature called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_air_touch_active(False), loop)


def enable_navigation_to_app():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Navigation To App.")
        return
    print(f"Enable Navigation To App called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_firmware_target(FirmwareTarget.navigation_to_app, True), loop
    )


def disable_navigation_to_app():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Navigation To App.")
        return
    print(f"Disable Navigation To App called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_firmware_target(FirmwareTarget.navigation_to_app, False), loop
    )


def enable_gesture_to_hid():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Gesture To HID.")
        return
    print(f"Enable Gesture To HID called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_firmware_target(FirmwareTarget.gesture_to_hid, True), loop
    )


def disable_gesture_to_hid():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Gesture To HID.")
        return
    print(f"Disable Gesture To HID called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_firmware_target(FirmwareTarget.gesture_to_hid, False), loop
    )


def enable_navigation_to_hid():
    device = get_selected_device()
    if device is None:
        print("No device selected to enable Navigation To HID.")
        return
    print(f"Enable Navigation To HID called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_firmware_target(FirmwareTarget.navigation_to_hid, True), loop
    )


def disable_navigation_to_hid():
    device = get_selected_device()
    if device is None:
        print("No device selected to disable Navigation To HID.")
        return
    print(f"Disable Navigation To HID called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(
        device.set_firmware_target(FirmwareTarget.navigation_to_hid, False), loop
    )


def set_hand_left():
    device = get_selected_device()
    if device is None:
        print("No device selected to set hand to left.")
        return
    print(f"Set hand to left called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_hand(HandType.left), loop)


def set_hand_right():
    device = get_selected_device()
    if device is None:
        print("No device selected to set hand to right.")
        return
    print(f"Set hand to right called for device: {device.name}")
    asyncio.run_coroutine_threadsafe(device.set_hand(HandType.right), loop)


def _update_indicator(indicator_or_list, color):
    """Helper function to update indicator(s) - handles both single indicator and list of indicators."""
    if not root.winfo_exists():
        return
    if isinstance(indicator_or_list, list):
        for indicator in indicator_or_list:
            if indicator:
                indicator.config(bg="white", fg=color)
    else:
        if indicator_or_list:
            indicator_or_list.config(bg="white", fg=color)


def update_status_indicators():
    """Update all status indicators based on the selected device's firmware_status."""
    global status_indicators
    device = get_selected_device()
    if device is None or not hasattr(device, "firmware_status"):
        # Set all indicators to gray (unknown)
        for indicator in status_indicators.values():
            _update_indicator(indicator, "gray")
        if root.winfo_exists():
            root.after(500, update_status_indicators)
        return

    fs = device.firmware_status

    # Update Data Features indicators
    if "snc" in status_indicators:
        color = "green" if fs.is_snc_enabled else "red"
        _update_indicator(status_indicators["snc"], color)

    if "imu_acc" in status_indicators:
        color = "green" if fs.is_acc_enabled else "red"
        _update_indicator(status_indicators["imu_acc"], color)

    if "imu_gyro" in status_indicators:
        color = "green" if fs.is_gyro_enabled else "red"
        _update_indicator(status_indicators["imu_gyro"], color)

    if "pressure" in status_indicators:
        color = "green" if fs.is_pressure_enabled else "red"
        _update_indicator(status_indicators["pressure"], color)

    if "navigation" in status_indicators:
        color = "green" if fs.is_navigation_enabled else "red"
        _update_indicator(status_indicators["navigation"], color)

    if "gesture" in status_indicators:
        color = "green" if fs.is_gesture_enabled else "red"
        _update_indicator(status_indicators["gesture"], color)

    if "air_touch" in status_indicators:
        color = "green" if fs.is_air_touch_enabled else "red"
        _update_indicator(status_indicators["air_touch"], color)

    # Update Firmware Targets indicators
    if "nav_to_app" in status_indicators:
        color = "green" if fs.is_sends_navigation_to_app_enabled else "red"
        _update_indicator(status_indicators["nav_to_app"], color)

    if "gesture_to_hid" in status_indicators:
        color = "green" if fs.is_sends_gesture_to_hid_enabled else "red"
        _update_indicator(status_indicators["gesture_to_hid"], color)

    if "nav_to_hid" in status_indicators:
        color = "green" if fs.is_sends_navigation_to_hid_enabled else "red"
        _update_indicator(status_indicators["nav_to_hid"], color)

    # Update Embedded Features indicators
    if "embedded_airtouch" in status_indicators:
        color = "green" if fs.is_air_touch_enabled else "red"
        _update_indicator(status_indicators["embedded_airtouch"], color)

    # Schedule next update
    if root.winfo_exists():
        root.after(500, update_status_indicators)  # Update every 500ms


def main():
    """Main function to set up and run the Tkinter GUI application."""
    global \
        root, \
        listbox, \
        pressure_label, \
        pressure_bar, \
        rms_bars, \
        rms_labels, \
        status_indicators
    global \
        imu_acc_label, \
        imu_gyro_label, \
        navigation_label, \
        gesture_label, \
        air_touch_button_label

    # Clear status indicators for fresh start
    status_indicators.clear()

    # Set up basic Tkinter window
    root = tk.Tk()
    root.title("Mudra BLE Device Manager")
    root.geometry("900x750")

    # Configure style
    style = ttk.Style()
    # Use platform-appropriate theme
    if platform.system() == "Darwin":  # macOS
        style.theme_use("aqua")
    else:
        style.theme_use("clam")

    # Platform-specific font
    if platform.system() == "Darwin":  # macOS
        default_font = ("Helvetica", 10)
        small_font = ("Helvetica", 9)
    else:  # Windows/Linux
        default_font = ("Segoe UI", 10)
        small_font = ("Segoe UI", 9)

    # Main container
    main_container = tk.Frame(root, bg="#f0f0f0")
    main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Left panel - Device List and Sensor Data
    left_panel = tk.Frame(main_container, bg="#f0f0f0")
    left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

    # Right panel - Controls
    right_panel = tk.Frame(main_container, bg="#f0f0f0")
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

    # ========== LEFT PANEL ==========

    # Device List Section
    device_frame = ttk.LabelFrame(left_panel, text="Discovered Devices", padding=10)
    device_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    listbox = tk.Listbox(
        device_frame, width=35, height=12, font=default_font, bg="white", fg="black"
    )
    listbox.pack(fill=tk.BOTH, expand=True)

    # Scrollbar for listbox
    scrollbar = ttk.Scrollbar(device_frame, orient=tk.VERTICAL, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    # Sensor Data Section
    sensor_frame = ttk.LabelFrame(left_panel, text="Sensor Data", padding=10)
    sensor_frame.pack(fill=tk.BOTH, expand=True)

    # Pressure UI
    pressure_container = tk.Frame(sensor_frame, bg="white")
    pressure_container.pack(fill=tk.X, pady=(0, 10))

    pressure_label = tk.Label(
        pressure_container,
        text="Pressure: 0.00",
        font=small_font,
        bg="white",
        fg="black",
        width=15,
        anchor="w",
    )
    pressure_label.pack(side=tk.LEFT, padx=(0, 10))

    pressure_bar = ttk.Progressbar(
        pressure_container,
        orient=tk.HORIZONTAL,
        length=200,
        mode="determinate",
        maximum=100,
    )
    pressure_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # RMS UI: 3 progress bars
    rms_bars = []
    rms_labels = []
    for i in range(3):
        rms_container = tk.Frame(sensor_frame, bg="white")
        rms_container.pack(fill=tk.X, pady=(0, 5) if i < 2 else 0)

        lbl = tk.Label(
            rms_container,
            text=f"RMS {i+1}: 0.00",
            font=small_font,
            bg="white",
            fg="black",
            width=15,
            anchor="w",
        )
        lbl.pack(side=tk.LEFT, padx=(0, 10))

        bar = ttk.Progressbar(
            rms_container,
            orient=tk.HORIZONTAL,
            length=200,
            mode="determinate",
            maximum=100,
            value=0,
        )
        bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        rms_labels.append(lbl)
        rms_bars.append(bar)

    # IMU Accelerometer UI
    imu_acc_container = tk.Frame(sensor_frame, bg="white")
    imu_acc_container.pack(fill=tk.X, pady=(10, 5))

    imu_acc_label = tk.Label(
        imu_acc_container,
        text="IMU Acc Frequency: -- Hz",
        font=small_font,
        bg="white",
        fg="black",
        anchor="w",
    )
    imu_acc_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    # IMU Gyroscope UI
    imu_gyro_container = tk.Frame(sensor_frame, bg="white")
    imu_gyro_container.pack(fill=tk.X, pady=(0, 5))

    imu_gyro_label = tk.Label(
        imu_gyro_container,
        text="IMU Gyro Frequency: -- Hz",
        font=small_font,
        bg="white",
        fg="black",
        anchor="w",
    )
    imu_gyro_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    # Navigation UI
    navigation_container = tk.Frame(sensor_frame, bg="white")
    navigation_container.pack(fill=tk.X, pady=(0, 5))

    navigation_label = tk.Label(
        navigation_container,
        text="Navigation: ΔX=0.000, ΔY=0.000",
        font=small_font,
        bg="white",
        fg="black",
        anchor="w",
    )
    navigation_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    # Gesture UI
    gesture_container = tk.Frame(sensor_frame, bg="white")
    gesture_container.pack(fill=tk.X, pady=(0, 5))

    gesture_label = tk.Label(
        gesture_container,
        text="Gesture: --",
        font=small_font,
        bg="white",
        fg="black",
        anchor="w",
    )
    gesture_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    # Air Touch Button UI
    air_touch_button_container = tk.Frame(sensor_frame, bg="white")
    air_touch_button_container.pack(fill=tk.X, pady=(0, 0))

    air_touch_button_label = tk.Label(
        air_touch_button_container,
        text="Air Touch Button: --",
        font=small_font,
        bg="white",
        fg="black",
        anchor="w",
    )
    air_touch_button_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    # ========== RIGHT PANEL ==========

    # Device Control Section
    device_control_frame = ttk.LabelFrame(
        right_panel, text="Device Control", padding=10
    )
    device_control_frame.pack(fill=tk.X, pady=(0, 10))

    scan_frame = tk.Frame(device_control_frame, bg="white")
    scan_frame.pack(fill=tk.X, pady=(0, 5))

    start_btn = ttk.Button(scan_frame, text="Start Scan", command=start_scan, width=15)
    start_btn.pack(side=tk.LEFT, padx=(0, 5))

    stop_btn = ttk.Button(scan_frame, text="Stop Scan", command=stop_scan, width=15)
    stop_btn.pack(side=tk.LEFT)

    connection_frame = tk.Frame(device_control_frame, bg="white")
    connection_frame.pack(fill=tk.X)

    connect_btn = ttk.Button(
        connection_frame, text="Connect", command=connect_selected_device, width=15
    )
    connect_btn.pack(side=tk.LEFT, padx=(0, 5))

    disconnect_btn = ttk.Button(
        connection_frame,
        text="Disconnect",
        command=disconnect_selected_device,
        width=15,
    )
    disconnect_btn.pack(side=tk.LEFT, padx=(0, 5))

    discover_btn = ttk.Button(
        connection_frame, text="Discover", command=discover_selected_device, width=15
    )
    discover_btn.pack(side=tk.LEFT)

    # Hand Selection Frame
    hand_frame = tk.Frame(device_control_frame, bg="white")
    hand_frame.pack(fill=tk.X, pady=(5, 0))

    hand_label = tk.Label(
        hand_frame,
        text="Hand:",
        font=small_font,
        bg="white",
        fg="black",
        width=8,
        anchor="w",
    )
    hand_label.pack(side=tk.LEFT, padx=(0, 5))

    set_hand_left_btn = ttk.Button(
        hand_frame, text="Left", command=set_hand_left, width=12
    )
    set_hand_left_btn.pack(side=tk.LEFT, padx=(0, 3))

    set_hand_right_btn = ttk.Button(
        hand_frame, text="Right", command=set_hand_right, width=12
    )
    set_hand_right_btn.pack(side=tk.LEFT)

    # Data Features Section
    features_frame = ttk.LabelFrame(right_panel, text="Data Features", padding=10)
    features_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    # Create a scrollable frame for features
    canvas = tk.Canvas(features_frame, highlightthickness=0)
    scrollbar_features = ttk.Scrollbar(
        features_frame, orient="vertical", command=canvas.yview
    )
    scrollable_frame = tk.Frame(canvas, bg="white")

    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_features.set)

    def configure_scroll_region(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Update canvas window width to match canvas width
        canvas_width = canvas.winfo_width()
        if canvas_width > 1:
            canvas.itemconfig(canvas_window, width=canvas_width)

    scrollable_frame.bind("<Configure>", configure_scroll_region)

    # Bind canvas resize to update scrollable frame width
    def on_canvas_configure(event):
        canvas_width = event.width
        canvas.itemconfig(canvas_window, width=canvas_width)

    canvas.bind("<Configure>", on_canvas_configure)

    # Add mouse wheel support for macOS
    def on_mousewheel(event):
        # Only scroll if mouse is over the canvas
        if (
            canvas.winfo_containing(event.x_root, event.y_root) == canvas
            or canvas.winfo_containing(event.x_root, event.y_root) in canvas.find_all()
        ):
            if platform.system() == "Darwin":
                canvas.yview_scroll(int(-1 * (event.delta)), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Bind mouse wheel to canvas and scrollable frame
    def bind_mousewheel(widget):
        widget.bind("<MouseWheel>", on_mousewheel)
        if platform.system() == "Darwin":
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    bind_mousewheel(canvas)
    bind_mousewheel(scrollable_frame)

    # Feature buttons in pairs (Enable/Disable)
    features = [
        ("SNC", "snc", enable_snc_feature, disable_snc_feature),
        ("IMU Acc", "imu_acc", enable_imu_acc_feature, disable_imu_acc_feature),
        ("IMU Gyro", "imu_gyro", enable_imu_gyro_feature, disable_imu_gyro_feature),
        ("Pressure", "pressure", enable_pressure_feature, disable_pressure_feature),
        (
            "Navigation",
            "navigation",
            enable_navigation_feature,
            disable_navigation_feature,
        ),
        ("Gesture", "gesture", enable_gesture_feature, disable_gesture_feature),
        (
            "Button Changed",
            "gesture",
            enable_air_mouse_button_changed_feature,
            disable_air_mouse_button_changed_feature,
        ),
    ]

    for feature_name, feature_key, enable_cmd, disable_cmd in features:
        feature_row = tk.Frame(scrollable_frame, bg="white")
        feature_row.pack(fill=tk.X, pady=3)

        # Status indicator dot - store as list if key already exists to support multiple indicators
        indicator = tk.Label(
            feature_row,
            text="●",
            font=(default_font[0], 12),
            bg="white",
            fg="gray",
            width=2,
        )
        indicator.pack(side=tk.LEFT, padx=(0, 3))
        if feature_key in status_indicators:
            # If key already exists, convert to list and append
            existing = status_indicators[feature_key]
            if isinstance(existing, list):
                existing.append(indicator)
            else:
                status_indicators[feature_key] = [existing, indicator]
        else:
            status_indicators[feature_key] = indicator

        label = tk.Label(
            feature_row,
            text=feature_name,
            font=small_font,
            bg="white",
            fg="black",
            width=12,
            anchor="w",
        )
        label.pack(side=tk.LEFT, padx=(0, 5))

        enable_btn = ttk.Button(feature_row, text="On", command=enable_cmd, width=8)
        enable_btn.pack(side=tk.LEFT, padx=(0, 3))

        disable_btn = ttk.Button(feature_row, text="Off", command=disable_cmd, width=8)
        disable_btn.pack(side=tk.LEFT)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar_features.pack(side="right", fill="y")

    # Firmware Targets Section
    firmware_frame = ttk.LabelFrame(right_panel, text="Firmware Targets", padding=10)
    firmware_frame.pack(fill=tk.X, pady=(0, 10))

    firmware_targets = [
        (
            "Nav To App",
            "nav_to_app",
            enable_navigation_to_app,
            disable_navigation_to_app,
        ),
        (
            "Gesture To HID",
            "gesture_to_hid",
            enable_gesture_to_hid,
            disable_gesture_to_hid,
        ),
        (
            "Nav To HID",
            "nav_to_hid",
            enable_navigation_to_hid,
            disable_navigation_to_hid,
        ),
    ]

    for target_name, target_key, enable_cmd, disable_cmd in firmware_targets:
        target_row = tk.Frame(firmware_frame, bg="white")
        target_row.pack(fill=tk.X, pady=3)

        # Status indicator dot
        indicator = tk.Label(
            target_row,
            text="●",
            font=(default_font[0], 12),
            bg="white",
            fg="gray",
            width=2,
        )
        indicator.pack(side=tk.LEFT, padx=(0, 3))
        status_indicators[target_key] = indicator

        label = tk.Label(
            target_row,
            text=target_name,
            font=small_font,
            bg="white",
            fg="black",
            width=15,
            anchor="w",
        )
        label.pack(side=tk.LEFT, padx=(0, 5))

        enable_btn = ttk.Button(target_row, text="On", command=enable_cmd, width=8)
        enable_btn.pack(side=tk.LEFT, padx=(0, 3))

        disable_btn = ttk.Button(target_row, text="Off", command=disable_cmd, width=8)
        disable_btn.pack(side=tk.LEFT)

    # Embedded Features Section
    embedded_frame = ttk.LabelFrame(right_panel, text="Embedded Features", padding=10)
    embedded_frame.pack(fill=tk.X)

    embedded_row = tk.Frame(embedded_frame, bg="white")
    embedded_row.pack(fill=tk.X)

    # Status indicator dot for embedded airtouch
    embedded_indicator = tk.Label(
        embedded_row,
        text="●",
        font=(default_font[0], 12),
        bg="white",
        fg="gray",
        width=2,
    )
    embedded_indicator.pack(side=tk.LEFT, padx=(0, 3))
    status_indicators["embedded_airtouch"] = embedded_indicator

    embedded_label = tk.Label(
        embedded_row,
        text="AirTouch",
        font=small_font,
        bg="white",
        fg="black",
        width=15,
        anchor="w",
    )
    embedded_label.pack(side=tk.LEFT, padx=(0, 5))

    enable_embedded_btn = ttk.Button(
        embedded_row, text="On", command=enable_embedded_airtouch_feature, width=8
    )
    enable_embedded_btn.pack(side=tk.LEFT, padx=(0, 3))

    disable_embedded_btn = ttk.Button(
        embedded_row, text="Off", command=disable_embedded_airtouch_feature, width=8
    )
    disable_embedded_btn.pack(side=tk.LEFT)

    # Start status indicator updates
    root.after(500, update_status_indicators)

    root.mainloop()


if __name__ == "__main__":
    main()
