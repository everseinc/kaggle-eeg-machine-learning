import numpy
import random
import csv
import re
import tensorflow as tf

class CsvManager:
	csv_path = '../EEG_grasp_and_left_data/train/'
	subj_num = 1
	series_par_subj = 2

	def __init__(self, target_subj, target_series):
		self.data = []
		self.events = []
		self.original_data = []
		self.original_events = []
		self.pop_index = []
		for i in target_subj:
			for j in target_series:
				will_skip = False

				events_path = self.csv_path + 'subj' + str(i) + '_series' + str(j) + '_events.csv'
				events_file = open(events_path, 'r')
				events_lines = events_file.readlines()
				events_file.close()
				events_lines.pop(0)
				self.original_events.extend(events_lines)

				data_path = self.csv_path + 'subj' + str(i) + '_series' + str(j) + '_data.csv'
				data_file = open(data_path, 'r')
				data_lines = data_file.readlines()
				data_file.close()
				data_lines.pop(0)
				self.original_data.extend(data_lines)

				print(events_path)

		if len(self.original_data) != len(self.original_events):
			raise Exception('Length of data list and events list is not equal')

		self.original_length = len(self.original_data)

	def pre_process(self, will_remove_channels, channels = []):
		for index, events_line in enumerate(self.original_events):
			self.original_events[index] = events_line[-12:].replace('\n', '').split(",")
			data_list = self.original_data[index].split(",")
			data_list.pop(0)
			if will_remove_channels:
				data_list = self.remove_noisy_channels(data_list, channels)
			data_list_float = [float(str) for str in data_list]
			normalized_data_list = [(value - min(data_list_float)) / (max(data_list_float) - min(data_list_float)) for value in data_list_float]
			self.original_data[index] = normalized_data_list

			if events_line[-12:].replace('\n', '') == '0,0,0,0,0,0':
				continue
			if sum(map(int, events_line[-12:].replace('\n', '').split(","))) > 1:
				continue

			self.events.append(self.original_events[index])
			self.data.append(self.original_data[index])
			self.pop_index.append(index)

		self.length = len(self.data)

				 
	def shuffle_data_and_events(self, real_data_height):
		self.index = list(range(real_data_height, self.length))
		random.shuffle(self.index)

	def remove_noisy_channels(self, data_list, channels):
		return [data_list[i] for i in channels]


	def get_data_and_events(self, i, data_height, real_data_height):
		reminder = real_data_height % data_height
		epoch = int((real_data_height - reminder) / data_height)
		position = self.index[i]
		events = self.events[position]
		data = self.original_data[self.pop_index[position] - real_data_height:self.pop_index[position]]
		data = data[::epoch]

		return ([data], [events])

