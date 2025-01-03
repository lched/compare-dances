import asyncio
import websockets


async def ping_pong_handler(websocket, path):
    try:
        while True:
            # Send a "ping" message to the client
            await websocket.send("ping")
            print("Server: Sent 'ping'")

            # Wait for the client's response
            response = await websocket.recv()
            print(f"Server: Received '{response}'")

            # Wait for 1 second before sending the next ping
            await asyncio.sleep(1)
    except websockets.ConnectionClosed:
        print("Server: Client disconnected")


# Start the server on localhost, port 8765
start_server = websockets.serve(ping_pong_handler, "localhost", 8765)

print("Server is running on ws://localhost:8765")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
