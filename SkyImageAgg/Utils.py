import os
import hmac
import math
import time
import glob
import datetime
import hashlib
from datetime import datetime, timezone

import requests
import numpy as np
from astral import Astral, Location


class Utils:
    @staticmethod
    def encrypt_data(key, message):
        return hmac.new(key, bytes(message, 'ascii'), digestmod=hashlib.sha256).hexdigest()

    @staticmethod
    def send_post_request(url, data):
        post_data = {
            'data': data
        }
        return requests.post(url, data=post_data)

    @staticmethod
    def make_array_from_image(file):
        return np.fromfile(file, dtype=np.uint8)

    @staticmethod
    def get_file_timestamp(file):
        return datetime.fromtimestamp(os.path.getmtime(file))

    def get_file_datetime_as_string(self, file, datetime_format):
        return self.get_file_timestamp(file).strftime(datetime_format)

    @staticmethod
    def list_files(path):
        return glob.iglob(os.path.join(path, '*'))

    @staticmethod
    def get_sunrise_and_sunset_time(cam_latitude, cam_longitude, cam_altitude, date=None):
        if not date:
            date = datetime.now(timezone.utc).date()

        astral = Astral()
        astral.solar_depression = 'civil'
        location = Location(('custom', 'region', cam_latitude, cam_longitude, 'UTC', cam_altitude))

        try:
            sun = location.sun(date=date)
        except Exception:
            return datetime.combine(date, datetime.time(3, 0, 0, 0, timezone.utc)), \
                   datetime.combine(date, datetime.time(21, 0, 0, 0, timezone.utc))

        return sun['sunrise'], sun['sunset']

    @staticmethod
    def retry_on_failure(attempts, delay=3, back_off=1):
        # Taken from https://wiki.python.org/moin/PythonDecoratorLibrary#Retry
        if back_off < 1:
            raise ValueError("back_off must be greater than or equal to 1")

        attempts = math.floor(attempts)
        if attempts < 0:
            raise ValueError("tries must be 0 or greater")

        if delay <= 0:
            raise ValueError("delay must be greater than 0")

        def deco_retry(f):
            def f_retry(*args, **kwargs):
                m_tries, m_delay = attempts, delay  # make mutable

                rv = f(*args, **kwargs)  # first attempt
                while m_tries > 0:
                    if rv is True:  # Done on success
                        return True

                    m_tries -= 1  # consume an attempt
                    time.sleep(m_delay)  # wait...
                    m_delay *= back_off  # make future wait longer

                    rv = f(*args, **kwargs)  # Try again

                return False  # Ran out of tries :-(

            return f_retry  # true decorator -> decorated function

        return deco_retry  # @retry(arg[, ...]) -> true decorator