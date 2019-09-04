import os
import time
from datetime import datetime
import base64
import hashlib
import hmac
import json
import glob

import requests
import numpy as np

from SkyImageAgg.GSM import Messenger, GPRS


class Controller(Messenger, GPRS):
    def __init__(self, server, camera_id, auth_key, storage_path, time_format):
        super().__init__()
        self.cam_id = camera_id
        self.key = auth_key
        self.server = server
        self.storage_path = storage_path
        self.time_format = time_format

    def encrypt_data(self, message):
        return hmac.new(self.key, bytes(message, 'ascii'), digestmod=hashlib.sha256).hexdigest()

    def sync_time(self):
        if not self.enable_GPRS():
            return False
        counter = 0
        while True:
            self.logger.debug('Sync time')
            if os.system('sudo ntpdate -u tik.cesnet.cz') == 0:
                self.logger.info('Sync time OK')
                return True
            else:
                counter += 1
                time.sleep(1)
            if counter > 10:
                self.logger.error('Sync time error')
                return False

    @staticmethod
    def send_post_request(url, data):
        post_data = {
            'data': data
        }
        return requests.post(url, data=post_data)

    @staticmethod
    def make_array_from_file(file):
        return np.fromfile(file, dtype=np.uint8)

    @staticmethod
    def get_file_timestamp(file):
        return datetime.fromtimestamp(os.path.getmtime(file))

    def get_file_datetime_as_string(self, file):
        return self.get_file_timestamp(file).strftime(self.time_format)

    def upload_file_as_json(self, file, convert_to_array=True):
        if convert_to_array:
            file = self.make_array_from_file(file)

        data = {
            'status': 'ok',
            'id': self.cam_id,
            'time': self.get_file_datetime_as_string(file),
            'coding': 'Base64',
            'data': base64.b64encode(file).decode('ascii')
        }

        json_data = json.dumps(data)
        signature = self.encrypt_data(json_data)
        url = '{}{}'.format(self.server, signature)
        response = self.send_post_request(url, json_data)

        try:
            json_response = json.loads(response.text)
        except Exception as e:
            raise Exception(e)

        if json_response['status'] != 'ok':
            raise Exception(json_response['message'])

        return json_response

    def upload_file_as_bson(self, file):
        data = {
            "status": "ok",
            "id": self.cam_id,
            "time": self.get_file_datetime_as_string(file),
            "coding": "none"
        }

        json_data = json.dumps(data)
        signature = self.encrypt_data(json_data)
        url = '{}{}'.format(self.server, signature)

        if isinstance(file, str) or isinstance(file, bytes):
            files = [('image', file), ('json', json_data)]
        else:
            files = [('image', str(file)), ('json', json_data)]

        response = requests.post(url=url, files=files)

        try:
            json_response = json.loads(response.text)
        except Exception as e:
            raise Exception(e)

        if json_response['status'] != 'ok':
            raise Exception(json_response['message'])

        return json_response

    def send_thumbnail_file(self, file):
        self.logger.debug('Uploading log to the server')
        counter = 0
        while True:
            counter += 1
            self.enable_GPRS()
            try:
                self.upload_file_as_bson(file)
                self.logger.info('Upload thumbnail to server OK')
                self.disable_ppp()
                return
            except Exception as e:
                self.logger.error('Upload thumbnail to server error: {}'.format(e))
            if counter > 5:
                self.logger.error('Upload thumbnail to server error: too many attempts')
                break
        self.logger.debug('Upload thumbnail to server end')
        self.disable_ppp()

    def upload_logfile(self, log_file):
        self.logger.debug('Start upload log to server')
        counter = 0
        while True:
            counter += 1
            self.enable_GPRS()
            try:
                self.upload_file_as_bson(log_file)
                self.logger.info('upload log to server OK')

                return
            except Exception as e:
                self.logger.error('upload log to server error : ' + str(e))

            if counter > 5:
                self.logger.error('error upload log to server')
                break

        self.logger.debug('end upload log to server')

    def list_files_in_storage(self):
        return glob.iglob(os.path.join(self.storage_path, '*'))

    ########### THIS SHOULD BE DONE IN THE MAIN FILE ################
    def run_storage_controller(self):
        # Check if there are any images in the storage
        if self.list_files_in_storage():
            self.logger.info('Storage is not empty!')
            # iterate over the images in the storage path

            for image in self.list_files_in_storage():
                try:
                    self.upload_file_as_json(image)
                    self.logger.info('{} was successfully uploaded to server'.format(image))

                    try:
                        os.remove(os.path.join(image))

                    except Exception as e:
                        self.logger.error('{} could not be deleted due to the following error:\n{}'.format(image, e))

                except Exception as e:
                    self.logger.error('{} could not be uploaded to server due to the following error:\n{}'.format(image, e))

        else:
            self.logger.info('Storage is empty!')
