import json

import asyncio
import websockets
import pyzed.sl as sl

import cv2

WEBSOCKET_PORT = 8000


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True  # Track people across images flow
    body_param.enable_body_fitting = False  # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
    body_param.body_format = (
        sl.BODY_FORMAT.BODY_38
    )  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(
        min(camera_info.camera_configuration.resolution.width, 1280),
        min(camera_info.camera_configuration.resolution.height, 720),
    )

    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()

    async def send_bodies():
        async with websockets.connect(f"ws://localhost:{WEBSOCKET_PORT}") as websocket:
            try:
                while True:
                    # Grab an image
                    if zed.grab() == sl.ERROR_CODE.SUCCESS:
                        # Retrieve left image
                        zed.retrieve_image(
                            image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution
                        )
                        # Retrieve bodies
                        zed.retrieve_bodies(bodies, body_runtime_param)

                        motion_data = bodies.body_list[0].keypoint
                        data = {
                            f"bone{idx}": motion_data[idx].tolist()
                            for idx in range(motion_data.shape[0])
                        }
                        await websocket.send(json.dumps(data))
                        print("Client: Sent message")
            except websockets.ConnectionClosed:
                print("Client: Server disconnected")

    # Run the client
    print(f"Client is connecting to ws://localhost:{WEBSOCKET_PORT}")
    asyncio.get_event_loop().run_until_complete(send_bodies())

    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
