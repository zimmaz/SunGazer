from picamera import PiCamera
import time


class Camera:
    def __init__(self, integrated=True):
        self.integrated = integrated
        if self.integrated:
            self.cam = PiCamera()

    def start_preview(self):
        self.cam.start_preview()

    def stop_preview(self):
        self.cam.stop_preview()

    # todo check
    def cap_pic(self, image):
        self.cam.capture(image)

    def cap_video(self):
        pass


class IrrSensor:
    def __init__(self):
        pass