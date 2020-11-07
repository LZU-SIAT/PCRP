import pickle
import numpy as np
import os
from numpy.lib.format import open_memmap
import sys

root_path = '/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/Predict-Cluster/data_for_pytorch'

dataset = ['NTU60', 'NTU120']
protocol = ['cross_subject_data', 'cross_view_data']
paradigm = ['train', 'test']

toolbar_width = 30

def end_toolbar():
    sys.stdout.write("\n")

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def main():
    
    for dataset_i in dataset:
        for protocol_i in protocol:
            for paradigm_i in paradigm:

                max_frame = 300
                feature_len = 75
                sample_len = []
                sample_label = []

                path_temp = os.path.join(root_path, dataset_i, protocol_i)
                data_file = os.path.join(path_temp, 'trans_{}_data.pkl'.format(paradigm_i)) # (n, (), T, 75)

                with open(data_file, "rb") as fin:
                    data = pickle.load(fin)

                    fp = open_memmap(
                        '{}/trans_{}_data.npy'.format(path_temp, paradigm_i),
                        dtype='float32',
                        mode='w+',
                        shape=(len(data), max_frame, feature_len))


                    for i, s in enumerate(data):
                        print_toolbar(i * 1.0 / len(data),
                                    '({:>5}/{:<5}) Processing {:>5}-{:<5}-{:<5} data: '.format(
                                        i + 1, len(data), dataset_i, protocol_i, paradigm_i))

                        sample_label.append(s['label'])         
                        sample_len.append(len(s['input']))       
                        fp[i, 0:len(s['input']), :] = s['input']

                with open('{}/{}_label.pkl'.format(path_temp, paradigm_i), 'wb') as f:
                    pickle.dump(sample_label, f)
                with open('{}/{}_sample_len.pkl'.format(path_temp, paradigm_i), 'wb') as f:
                    pickle.dump(sample_len, f)
                end_toolbar()


if __name__ == "__main__":
    main()