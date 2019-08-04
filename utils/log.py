#encoding=utf-8
import datetime
import logging
import os

import sys

def get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S')  

def init_logger(log_file = None, log_path = None, log_level = logging.INFO, mode = 'w', stdout = True):
    """
    log_path: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    log_file = os.path.join(log_path, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)
    logging.basicConfig(level = log_level,
                format=fmt,
                filename=log_file,
                filemode=mode)
    
    if stdout:
        console = logging.StreamHandler(stream = sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def info(msg):
    logging.info(msg)