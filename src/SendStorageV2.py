
import LibrforPiV2 as lfp

import os
import configparser
import logging 
import numpy as np
import datetime as dt

def main():
    #inicialize logging
    logger=lfp.set_logger(logging.DEBUG)
    path_of_script = os.path.realpath(__file__)
    path_config = os.path.dirname(os.path.realpath(__file__))+'/config.ini' 

    # read config file
    conf = lfp.config_obj(path_config,logger)

    #inicialize log to file
    lfp.set_log_to_file(conf.log_path,conf.log_to_console,logger)

    if os.listdir(conf.path_storage):
        logger.info('Storage is not empty')
        
        files = os.listdir(conf.path_storage)
        for file in files:
            image_time = dt.datetime.strptime(file, conf.filetime_format)

            image= np.fromfile(conf.path_storage+'/'+file,dtype=np.uint8)
            success = True
            try:
                response=lfp.upload_json(image,image_time,conf.server)
            except Exception as e:
                logger.error(file+ ' upload to server error : '+str(e))
                success=False
                break
                
            if success==True:
                logger.info(file+' upload to server OK' ) 
                try:
                    os.remove(conf.path_storage+'/'+file)
                except Exception as e:
                    logger.error(file+ ' delete errror : '+str(e))
                    break
    else:
       logger.info('Storage is empty' ) 




if __name__ == '__main__':
    print("Running program...")
    main()

