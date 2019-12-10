#coding=utf-8
import os,sys
import time
import logging
import  fs_utils.FSPY_foo as FSLog


def init_logging():
    """
    Initialize logging parameters
    """
    curr_path = os.getcwd()
    log_path = curr_path + '/log'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_format = '[%(asctime)s] %(filename)s, line: %(lineno)d, [%(levelname)s]: %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, log_datefmt)
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=log_datefmt)
    localtime = time.strftime('%Y-%m-%d', time.localtime())
    log_filename = log_path + '/train_' + localtime + '.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        fslog( str(message) )
    def flush(self):
        pass

def fslog(arg):
    FSLog.fsdeepai_log("[FS-CPY]"+arg)

def setProgessUI(a, b, min_v, max_v,  x):
    FSLog.setProgessUI(a,b,max_v,max_v,x)

def setProgessUI2(x):
    FSLog.setProgessUI2(x)

def fsEncryption(arg):
   return  FSLog.fsEncryptionAES(arg)

def fsDecryption(arg):
   return  FSLog.fsDecryptionAES(arg)

def fsDecryption2(arg):
   return  FSLog.fsDecryptionAES2(arg)

