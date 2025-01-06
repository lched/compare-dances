from pythonosc.udp_client import SimpleUDPClient
import time


ip = "127.0.0.1"
port = 8000


client = SimpleUDPClient(ip, port)  # Create client
while True:
    client.send_message("/ping", "Hello!")  # Send float message
    print(f"Sent a ping to {ip}:{port}")
    time.sleep(1)
