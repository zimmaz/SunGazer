#!/usr/bin/python3
import os
import datetime as dt
import threading
from queue import LifoQueue
import time

from astral import Astral, Location

from SkyImageAgg.Controller import Controller
from SkyImageAgg.GSM import Messenger, GPRS, retry_on_failure
from SkyImageAgg.Collector import IrrSensor
from SkyImageAgg.Configuration import Configuration


class SkyScanner(Controller, Configuration):
    def __init__(self):
        self.config = Configuration()
        super().__init__(
            server=self.config.server,
            camera_id=self.config.id,
            image_quality=self.config.image_quality,
            auth_key=self.config.key,
            storage_path=self.config.storage_path,
            ext_storage_path=self.config.ext_storage_path,
            time_format=self.config.time_format,
            autonomous_mode=self.config.autonomous_mode,
            cam_address=self.config.cam_address,
            username=self.config.cam_username,
            pwd=self.config.cam_pwd,
            rpi_cam=self.config.integrated_cam
        )
        if self.config.light_sensor:
            self.sensor = IrrSensor(
                port=self.config.MODBUS_port,
                address=self.config.MODBUS_sensor_address,
                baudrate=self.config.MODBUS_baudrate,
                bytesize=self.config.MODBUS_bytesize,
                parity=self.config.MODBUS_parity,
                stopbits=self.config.MODBUS_stopbits
            )
        self.Messenger = Messenger()
        self.GPRS = GPRS(ppp_config_file=self.config.GSM_ppp_config_file)
        self.mask = self.get_binary_image(self.config.mask_path)
        self.main_stack = LifoQueue()
        self.aux_stack = LifoQueue()

    # TODO
    def set_requirements(self):
        if not self.GPRS.hasInternetConnection():
            self.GPRS.enable_GPRS()
        self.Messenger.send_sms(
            self.GSM_phone_no,
            'SOME MESSAGE AND INFO'
        )

    @staticmethod
    def sync_time():
        if os.system('sudo ntpdate -u tik.cesnet.cz') == 0:
            return True

    @staticmethod
    def get_sunrise_and_sunset_time(cam_latitude, cam_longitude, cam_altitude, date=None):
        if not date:
            date = dt.datetime.now(dt.timezone.utc).date()

        astral = Astral()
        astral.solar_depression = 'civil'
        location = Location(('custom', 'region', cam_latitude, cam_longitude, 'UTC', cam_altitude))
        sun = location.sun(date=date)

        return sun['sunrise'], sun['sunset']

    def _stamp_curr_time(self):
        return dt.datetime.utcnow().strftime(self.time_format)

    def scan(self):
        # store the current time according to the time format
        cap_time = self._stamp_curr_time()
        # set the path to save the image
        output_path = os.path.join(self.storage_path, cap_time)
        return cap_time, output_path, self.cam.cap_pic(output=output_path, return_arr=True)

    def preprocess(self, image_arr):
        # Crop
        image_arr = self.crop(image_arr, self.config.crop)
        # Apply mask
        image_arr = self.apply_binary_mask(self.mask, image_arr)
        return image_arr

    def execute(self):
        # capture the image and set the proper name and path
        cap_time, img_path, img_arr = self.scan()
        # preprocess the image
        preproc_img = self.preprocess(img_arr)
        # try to upload the image to the server, if failed, save it to storage
        try:
            self.upload_as_json(preproc_img, time_stamp=cap_time)
            print('Uploading {} was successful!'.format(img_path))
        except Exception:
            print('Couldn\'t upload {}! Queueing for retry!'.format(img_path))
            self.main_stack.put((cap_time, img_path, img_arr))

    @retry_on_failure(attempts=2)
    def retry_upload(self, image, time_stamp, convert_to_arr=False):
        self.upload_as_json(image, time_stamp, convert_to_arr)

    def execute_periodically(self, period=10):
        while True:
            kick_off = time.time()
            self.execute()
            try:
                wait = period - (time.time() - kick_off)
                print('Waiting {} seconds to capture the next image...'.format(round(wait, 1)))
                time.sleep(wait)
            except ValueError:
                pass

    def check_main_stack(self):
        while True:
            if not self.main_stack.empty():
                item = self.main_stack.get()
                try:
                    self.retry_upload(image=item[2], time_stamp=item[0])
                    print('Retrying to upload {} was successful!'.format(item[1]))
                except Exception:
                    print('Retrying to upload {} failed! Queueing for saving on disk'.format(item[1]))
                    self.aux_stack.put(item)
            else:
                print('Main stack is empty!')
                time.sleep(5)

    def check_aux_stack(self):
        while True:
            if not self.aux_stack.empty():
                item = self.aux_stack.get()
                try:
                    self.save_as_pic(image_arr=item[2], output_name=item[1])
                    print('{} saved to storage.'.format(item[1]))
                except Exception:
                    time.sleep(10)
            else:
                print('Auxiliary stack is empty!')
                time.sleep(5)

    def run(self):
        jobs = []
        print('Initiating the uploader!')
        uploader = threading.Thread(target=self.execute_periodically)
        jobs.append(uploader)
        print('Initiating the retriever!')
        retriever = threading.Thread(target=self.check_main_stack)
        jobs.append(retriever)
        print('Initiating the writer!')
        writer = threading.Thread(target=self.check_aux_stack)
        jobs.append(writer)

        for job in jobs:
            job.start()


if __name__ == '__main__':
    s = SkyScanner()
    s.run()
