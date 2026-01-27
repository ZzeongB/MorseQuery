import asyncio

from mudra_sdk import Mudra, MudraDevice
from mudra_sdk.models.callbacks import MudraDelegate

# Create Mudra instance
mudra = Mudra()

# Retrieve license from cloud (required for full functionality)
mudra.get_license_for_email_from_cloud("parkjeong02@gmail.com")

discovered_devices = []


class MyMudraDelegate(MudraDelegate):
    def on_device_discovered(self, device: MudraDevice):
        discovered_devices.append(device)
        print(f"Discovered: {device.name}")

    def on_mudra_device_connected(self, device: MudraDevice):
        print(f"Device connected: {device.name}")

    def on_mudra_device_disconnected(self, device: MudraDevice):
        print(f"Device disconnected: {device.name}")

    def on_mudra_device_connecting(self, device: MudraDevice):
        print(f"Device connecting: {device.name}...")

    def on_mudra_device_disconnecting(self, device: MudraDevice):
        print(f"Device disconnecting: {device.name}...")

    def on_mudra_device_connection_failed(self, device: MudraDevice, error: str):
        print(f"Connection failed: {device.name}, Error: {error}")

    def on_bluetooth_state_changed(self, state: bool):
        print(f"Bluetooth state changed: {'On' if state else 'Off'}")


async def main():
    mudra = Mudra()
    mudra.set_delegate(MyMudraDelegate())

    # Start scanning
    await mudra.scan()
    await asyncio.sleep(5)  # Wait for discovery

    # Connect to the first discovered device
    if discovered_devices:
        device = discovered_devices[0]
        await device.connect()
        print(f"Connected to {device.name}")

        # ... use the device ...

        # Disconnect when done
        await device.disconnect()


asyncio.run(main())
