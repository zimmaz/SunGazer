import os
import time

import minimalmodbus


class IrrSensor(minimalmodbus.Instrument):
    """

    """

    def __init__(self, serial_port, slave_address, baudrate, bytesize, parity, stopbits):
        super().__init__(port=serial_port, slaveaddress=slave_address)
        self.serial.baudrate = baudrate
        self.serial.bytesize = bytesize
        self.serial.parity = parity
        self.serial.stopbits = stopbits
        self.serial.rtscts = False
        self.serial.dsrdtr = True
        self.serial.timeout = 0.1

    def open_serial(self):
        """
        checks if the serial port is already open, if not open it.
        """
        if not self.serial.isOpen():
            self.serial.open()

    def get_data(self):
        """

        Returns
        -------

        """
        self.open_serial()
        try:
            irr = self.read_register(0, 1, 4, False)
            ext_temp = self.read_register(8, 1, 4, True)
            cell_temp = self.read_register(7, 1, 4, True)
        except Exception as e:
            self.serial.close()
            raise Exception(e)
        self.serial.close()
        return irr, ext_temp, cell_temp

    @staticmethod
    def restart_USB2Serial():
        """

        """
        time.sleep(0.5)
        os.system('sudo modprobe -r pl2303')
        time.sleep(0.2)
        os.system('sudo modprobe -r usbserial')
        time.sleep(0.2)
        os.system('sudo modprobe pl2303')
        time.sleep(0.5)
