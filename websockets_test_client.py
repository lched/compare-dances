import asyncio
import websockets


async def pong_handler():
    async with websockets.connect("ws://localhost:8765") as websocket:
        try:
            while True:
                # Wait for a message from the server
                message = await websocket.recv()
                print(f"Client: Received '{message}'")

                if message == "ping":
                    # Respond with "pong"
                    await websocket.send("pong")
                    print("Client: Sent 'pong'")
        except websockets.ConnectionClosed:
            print("Client: Server disconnected")


# Run the client
print("Client is connecting to ws://localhost:8765")
asyncio.get_event_loop().run_until_complete(pong_handler())
