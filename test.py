import time
import numpy as np
import tensorflow as tf
from helper import *
from csv_manager import *

start = time.time()
print('start')

data_manager = CsvManager()

elapsed_time = time.time() - start
print("end -- new -- elapsed_time:{0}".format(elapsed_time) + "[sec]")

start = time.time()
print('start')

data_manager.pre_process()

elapsed_time = time.time() - start
print("end -- pre_process -- elapsed_time:{0}".format(elapsed_time) + "[sec]")

print((data_manager.data[10], data_manager.events[10]))

start = time.time()
print('start')

data_manager.shuffle_data_and_events()

elapsed_time = time.time() - start
print("end -- shuffle -- elapsed_time:{0}".format(elapsed_time) + "[sec]")

print(data_manager.get_data_and_events(10, 2))