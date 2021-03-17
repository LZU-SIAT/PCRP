import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# training_subjects = [
#     1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
# ]
training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
                     45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
                     83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]

# training_cameras = [2, 3]
training_cameras = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32] # For ntu 120 cross-setup

max_body = 2
num_joint = 25
max_frame = 100
toolbar_width = 30

def ntu_tranform_skeleton(test):
    """
    :param test: frames of skeleton within a video sample
    """
    remove_frame = False
    test = np.asarray(test)
    transform_test = []
    
    
    
    # tiny = 0.00000001

    frame = 0
    for time in range(test.shape[0]):
        # d = test[frame,0:3]
        v1 = test[time,1*3:1*3+3]-test[time,0*3:0*3+3]

        v2_ = test[time,12*3:12*3+3]-test[time,16*3:16*3+3]
        # if v1.all() != 0 and v2_.all() != 0:
        if np.all(v1 == 0) == False:
            if np.all(v2_ == 0) == False:
                frame = time
                break

    d = test[frame,0:3]
    v1 = test[frame,1*3:1*3+3]-test[frame,0*3:0*3+3]
    
    v1 = v1/np.linalg.norm(v1)
    
    v2_ = test[frame,12*3:12*3+3]-test[frame,16*3:16*3+3]
    proj_v2_v1 = np.dot(v1.T,v2_)*v1/np.linalg.norm(v1) 
    v2 = v2_-np.squeeze(proj_v2_v1)
    v2 = v2/np.linalg.norm(v2) 
    
    v3 = np.cross(v2,v1)/np.linalg.norm(np.cross(v2,v1)) 
    
    v1 = np.reshape(v1,(3,1))
    v2 = np.reshape(v2,(3,1))
    v3 = np.reshape(v3,(3,1))
    
    R = np.hstack([v2,v3,v1])
    
    for i in range(test.shape[0]):
        xyzs = []
        for j in range(25*2):
            if j < 25:
                if test[i][j*3:j*3+3].all()==0:
                    remove_frame = True
                    break
            xyz = np.squeeze(np.matmul(np.linalg.inv(R),np.reshape(test[i][j*3:j*3+3]-d,(3,1))))
            xyzs.append(xyz)
        if not remove_frame:
            xyzs = np.reshape(np.asarray(xyzs),(-1,75 * 2))
            transform_test.append(xyzs)
        else:
            remove_frame = False
    
    transform_test = np.squeeze(np.asarray(transform_test))
    return transform_test


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


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='cross_view_data',
            part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    data_lens = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        # camera_id = int(
        #     filename[filename.find('C') + 1:filename.find('C') + 4])
        camera_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 4])

        if benchmark == 'cross_view_data':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'cross_subject_data':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'test':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            # print("in here")
            # if action_class == 120 :
            #     print("120")
            # if action_class == 1:
            #     print("0")
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        # pickle.dump((sample_name, list(sample_label)), f)
        pickle.dump( list(sample_label ), f)

    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/trans_{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), max_frame, 3 * num_joint * max_body))


    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint) # C, T, V, M
        
        C, T, V, M = data.shape

        data = data.transpose([1, -1, -2, 0]).reshape(T, -1) # T, M * V * C

        data = ntu_tranform_skeleton(data)

        # attention !!
        if data.shape[0] > max_frame:
            fp[i] = data[0:max_frame]
        else:
            # print(data.shape)
            fp[i, 0:data.shape[0]] = data

        data_lens.append(data.shape[0])

    with open('{}/{}_sample_len.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump( data_lens, f)
    
    
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/data5/xushihao/data/ntu_raw_data/nturgb+d_skeletons/')
    parser.add_argument(
        '--ignored_sample_path',
        default='/data5/xushihao/data/NTU_RGBD120_samples_with_missing_skeletons_new.txt')
    parser.add_argument('--out_folder', default='/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/Predict-Cluster/pc_pytorch/data_for_pytorch/ntu120_2person')

    benchmark = ['cross_subject_data', 'cross_view_data']
    part = ['train', 'test']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
