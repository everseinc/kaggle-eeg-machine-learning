import numpy
import random
import csv
import re
import tensorflow as tf

class CsvManager:
	csv_path = '../EEG_grasp_and_left_data/train/'
	subj_num = 1
	series_par_subj = 3

	def __init__(self):
		self.data = []
		self.events = []
		self.original_data = []
		self.original_events = []
		self.pop_index = []
		for i in range(self.subj_num):
			for j in range(self.series_par_subj):
				will_skip = False

				events_path = self.csv_path + 'subj' + str(i + 1) + '_series' + str(j + 1) + '_events.csv'
				events_file = open(events_path, 'r')
				events_lines = events_file.readlines()
				events_file.close()
				events_lines.pop(0)
				self.original_events.extend(events_lines)

				data_path = self.csv_path + 'subj' + str(i + 1) + '_series' + str(j + 1) + '_data.csv'
				data_file = open(data_path, 'r')
				data_lines = data_file.readlines()
				data_file.close()
				data_lines.pop(0)
				self.original_data.extend(data_lines)

				print(events_path)

		if len(self.original_data) != len(self.original_events):
			raise Exception('Length of data list and events list is not equal')

		self.original_length = len(self.original_data)

	def pre_process(self):
		for index, events_line in enumerate(self.original_events):
			self.original_events[index] = events_line[-12:].replace('\n', '').split(",")
			data_list = self.original_data[index].split(",")
			data_list.pop(0)
			data_list_float = [float(str) for str in data_list]
			normalized_data_list = [(value - min(data_list_float)) / (max(data_list_float) - min(data_list_float)) for value in data_list_float]
			self.original_data[index] = normalized_data_list

			if events_line[-12:].replace('\n', '') == '0,0,0,0,0,0':
				continue

			self.events.append(self.original_events[index])
			self.data.append(self.original_data[index])
			self.pop_index.append(index)

		self.length = len(self.data)

				 
	def shuffle_data_and_events(self):
		self.index = list(range(self.length))
		random.shuffle(self.index)

	def get_data_and_events(self, i, data_height):
		position = self.index[i]
		events = self.events[position]
		data = self.original_data[self.pop_index[position] - data_height:self.pop_index[position]]

		return ([data], [events])

