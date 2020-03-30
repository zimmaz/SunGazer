from os.path import dirname, join
import datetime as dt
import pickle
from pathlib import Path
import glob

import cv2
import numpy as np
import pandas as pd
from pvlib.location import Location
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression

_base_dir = dirname(dirname(__file__))
_azimuth_reg = join(_base_dir, 'azimuth_regressor.pkl')
_zenith_reg = join(_base_dir, 'zenith_regressor.pkl')


def click_on_sun(img_path):
    """

    Parameters
    ----------
    img_path

    Returns
    -------

    """
    img = cv2.imread(img_path)

    def draw_circle(event, x, y, *args, **kwargs):
        global mouse_x, mouse_y
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
            mouse_x, mouse_y = x, y

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.setMouseCallback('image', draw_circle)

    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 32:
            return [Path(img_path).stem, mouse_x, mouse_y, img.shape[0], img.shape[1]]
        elif k == ord('x'):
            raise KeyboardInterrupt


class SunLocator(Location):
    """

    """

    def __init__(self, latitude, longitude, altitude, time_format='%Y-%m-%d_%H-%M-%S'):
        super().__init__(latitude=latitude, longitude=longitude, altitude=altitude)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.time_format = time_format
        self.time_range = None
        self.time_table = None
        self.azim_lr = None
        self.znth_lr = None
        self.resolution = None

    def set_time_format(self, time_format):
        """

        Parameters
        ----------
        time_format
        """
        self.time_format = time_format

    def set_resolution(self, width, height):
        """

        Parameters
        ----------
        width
        height
        """
        self.resolution = (width, height)

    def set_time_range(self, start='today', end='next_day', freq='10S'):
        """

        Parameters
        ----------
        start
        end
        freq
        """
        if start == 'today':
            start = dt.datetime.combine(dt.datetime.today(), dt.time.min)
        else:
            assert isinstance(start, dt.date)
            start = dt.datetime.combine(start, dt.time.min)

        if end == 'next_day':
            end = start + dt.timedelta(days=1)
        else:
            assert isinstance(end, dt.date)
            end = dt.datetime.combine(end, dt.time.min)
        self.time_range = pd.date_range(start, end, freq=freq)

    def load_regressors(self):
        """

        """
        try:
            with open(_azimuth_reg, 'rb') as f1, open(_zenith_reg, 'rb') as f2:
                self.azim_lr = pickle.load(f1)
                self.znth_lr = pickle.load(f2)
        except FileNotFoundError:
            raise UserWarning('The model is not calibrated! You may consider calibrating the model first!')

    def set_up(self, start='today', end='next_day', freq='10S', resolution=(1926, 1926)):
        """

        Parameters
        ----------
        start
        end
        freq
        resolution
        """
        self.set_time_range(start, end, freq)
        self.time_table = self.get_solarposition(self.time_range)
        self.load_regressors()
        self.resolution = resolution
        self.get_solar_position_in_images()

    def reset(self):
        """

        """
        self.set_up()

    def _mark_sun_in_images(self, images_dir):
        df = pd.DataFrame(columns=['Time', 'Sx', 'Sy', 'img_height', 'img_width'])

        for img in glob.iglob(join(images_dir, '*.jpg')):
            try:
                try:
                    df.loc[len(df)] = click_on_sun(img)
                except NameError:
                    pass
            except KeyboardInterrupt:
                df = df.drop_duplicates(subset=['Sx', 'Sy'], keep='first').reset_index(drop=True)
                df_ = self.get_solarposition(pd.to_datetime(df['Time'], format=self.time_format))
                return df.join(df_.reset_index(drop=True))

        df = df.drop_duplicates(subset=['Sx', 'Sy'], keep='first').reset_index(drop=True)
        df_ = self.get_solarposition(pd.to_datetime(df['Time'], format=self.time_format))
        return df.join(df_.reset_index(drop=True))

    def calibrate(self, images_dir):
        """

        Parameters
        ----------
        images_dir

        Returns
        -------

        """
        df = self._mark_sun_in_images(images_dir)
        df['cartesian_az'] = df.apply(
            lambda r: np.degrees(np.arctan((r['Sx'] - r['img_width'] / 2) / (r['Sy'] - r['img_height'] / 2))),
            axis=1
        )
        df['cartesian_zn'] = df.apply(
            lambda r: distance.euclidean((r['Sx'], r['Sy']), (r['img_width'] / 2, r['img_height'] / 2)),
            axis=1
        )
        lr_az = LinearRegression()
        lr_zn = LinearRegression()
        lr_az.fit(df[['azimuth']], df[['cartesian_az']])
        lr_zn.fit(df[['zenith']], df[['cartesian_zn']])

        with open(_azimuth_reg, 'wb') as f1, open(_zenith_reg, 'wb') as f2:
            pickle.dump(lr_az, f1)
            pickle.dump(lr_zn, f2)

        return df

    def evaluate(self, images_dir):
        """

        Parameters
        ----------
        images_dir
        """
        # create a dataframe for the images in the folder
        df = pd.DataFrame()
        df['Time'] = [Path(img).stem for img in glob.iglob(join(images_dir, '*.jpg'))]

        # calculate the azimuth and zenith for each picture
        df = self.get_solarposition(
            pd.to_datetime(df['Time'], format=self.time_format)
        ).reset_index(drop=True)

        # get the path to each pic
        df['Path'] = [img for img in glob.iglob(join(images_dir, '*.jpg'))]

        # load the regression models
        self.load_regressors()

        # calculate the cartesian-like solar positions in the image
        df['cartesian_az'] = self.azim_lr.predict(df[['azimuth']])
        df['cartesian_zn'] = self.znth_lr.predict(df[['zenith']])

        # get the size of images
        sample = cv2.imread(df['Path'].loc[len(df) - 1])
        height, width = sample.shape[:2]

        # get coordinates of sun for each pic
        df['x'] = df.apply(
            lambda r: int(-r['cartesian_zn'] * np.sin(np.deg2rad(r['cartesian_az'])) + width / 2),
            axis=1
        )
        df['y'] = df.apply(
            lambda r: int(-r['cartesian_zn'] * np.cos(np.deg2rad(r['cartesian_az'])) + height / 2),
            axis=1
        )

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600, 600)

        for _, r in df.iterrows():
            img = cv2.imread(r['Path'])
            img = cv2.circle(img, (r['x'], r['y']), 100, (0, 240, 0), 5)
            while 1:
                cv2.imshow('image', img)
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    break
                elif k == ord('x'):
                    raise KeyboardInterrupt

    def get_solar_position_in_images(self):
        """

        """
        self.time_table['cartesian_az'] = self.azim_lr.predict(self.time_table[['azimuth']])
        self.time_table['cartesian_zn'] = self.znth_lr.predict(self.time_table[['zenith']])
        self.time_table['x'] = self.time_table.apply(
            lambda r: int(-r['cartesian_zn'] * np.sin(np.deg2rad(r['cartesian_az'])) + self.resolution[1] / 2),
            axis=1
        )
        self.time_table['y'] = self.time_table.apply(
            lambda r: int(-r['cartesian_zn'] * np.cos(np.deg2rad(r['cartesian_az'])) + self.resolution[0] / 2),
            axis=1
        )

    def find_sun_coordinates(self, timestamp):
        """

        Parameters
        ----------
        timestamp

        Returns
        -------

        """
        try:
            return tuple(self.time_table[['x', 'y']].loc[timestamp])
        except KeyError:
            return tuple(self.time_table[['x', 'y']].loc[dt.datetime.strptime(timestamp, self.time_format)])
