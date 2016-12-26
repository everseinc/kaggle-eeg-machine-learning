import csv
import glob

def copy_csv():
	all_file = open('../EEG_grasp_and_left_data/train/all_subj_series_data.csv', 'w')
	files = glob.glob('../EEG_grasp_and_left_data/train/*_series*_data.csv')
	count = 0
	for filename in files:
		count += 1
		if count > 10:
			continue
		data_file = open(filename, 'r')

		lines = data_file.readlines()
		data_file.close()

		i = 0
		for line in lines:
			i += 1
			if i == 1:
				continue
			all_file.writelines(line)

		print(filename)

	all_file.close()


def check_label():
	files = glob.glob('../EEG_grasp_and_left_data/train/*_series*_events.csv')
	total = 0
	type = {}
	for filename in files:
		print(filename)
		label_file = open(filename, 'r')

		lines = label_file.readlines()
		label_file.close()

		i = 0
		for line in lines:
			i += 1
			if i == 1:
				continue

			array = line.split(",")
			id =  array.pop(0)
			values = ",".join(array)

			if values in type:
				type[values] += 1
			else:
				type[values] = 0

			total += 1

	sorted_type = sorted(type.items())
	print(sorted_type)
	print(total)


copy_csv()
# copy_csv('../EEG_grasp_and_left_data/train/subj10_series1_data.csv')

