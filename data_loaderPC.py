from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler

from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

import random
import torch

import h5py
import numpy as np
import pickle
import os
import  glob
import pandas as pd

def get_data_list(data_path):
    f = h5py.File(data_path, 'r')
    data_list = []
    label_list = []
    for i in range(len(f['label'])):

        if np.shape(f[str(i)][:])[0] > 10:
            x = f[str(i)][:]
            # original matrix with probability
            y = f['label'][i]

            x = torch.tensor(x, dtype=torch.float)

            data_list.append(x)
            label_list.append(y)

    return data_list, label_list


# def get_data_list_ntu(data_path):
#     data_list = []
#     label_list = []
#     with open(data_path, "rb") as fin:
#         data_src = pickle.load(fin)
#         for data in data_src:
#             data_list.append(torch.tensor(data['input'], dtype=torch.float)) # T, 75
#             label_list.append(data['label'])

#     return data_list, label_list

def concate_data(data_path, seq_len = 10):
    data_list, label_list = get_data_list(data_path)

    feature_len = data_list[0].size()[-1]
    data = torch.tensor(())
    for i in range(len(label_list)):
        if data_list[i].size()[0] == seq_len:
            tmp = troch.flatten(data_list[i])
            data = torch.cat((data, tmp)).unsqueeze(0) 

        if data_list[i].size()[0] < seq_len:
          dif = seq_len - data_list.size()[0]
          tmp = torch.cat((data_list[i], torch.zeros((dif, feature_len))))
          tmp = torch.flatten(tmp)
          data = torch.cat((data, tmp)).unsqueeze(0) 
        
        if data_list[i].size()[0] > seq_len:
          tmp = data_list[i][:seq_len,:]
          tmp = torch.flatten(tmp).unsqueeze(0) 
          data = torch.cat((data, tmp))
    label_list = np.asarray(label_list)
    return data.numpy(), label_lists


def pad_collate(batch):
    lens = [len(x[0]) for x in batch]

    data = [x[0] for x in batch]
    label = [x[1]-1 for x in batch]
    label = np.asarray(label)

    index = [x[2] for x in batch]
    index = np.asarray(index)

    # print(type(data))
    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
    return xx_pad, lens, label, index

def pad_collate_ntu(batch):
    data = [x[0] for x in batch]
    lens = [x[1] for x in batch]
    label = [x[2] for x in batch]
    label = np.asarray(label)

    index = [x[3] for x in batch]
    index = np.asarray(index)


    xx_pad = pad_sequence(data, batch_first=True, padding_value=0)
    return xx_pad, lens, label, index



class MyAutoDataset(Dataset):
    def __init__(self, data, label):
      
        self.data = data
        self.label = label
        #self.xy = zip(self.data, self.label)


    def __getitem__(self, index):
        sequence = self.data[index, :]
        label = self.label[index]
        # Transform it to Tensor
        #x = torchvision.transforms.functional.to_tensor(sequence)
        #x = torch.tensor(sequence, dtype=torch.float)
        #y = torch.tensor([self.label[index]], dtype=torch.int)
        
        return sequence, label

    def __len__(self):
        return len(self.label)

class MyDataset(Dataset):
    def __init__(self, data_path):

        self.data, self.label = get_data_list(data_path)

        # label = np.asarray(self.label)
        # train_index = np.zeros(len(self.label))

    def __getitem__(self, index):
        sequence = self.data[index]
        label = self.label[index]

        return sequence, label, index

    def __len__(self):
        return len(self.label)


def downsample(data,  target_frame=50):
    """
    Downsample input data into number of target frames
    :param data:
    :param target_frame:
    :return:
    """
    if len(data) > target_frame:
        return data[:target_frame], target_frame
    else:
        return data, len(data)
    




class MyDataset_ntu(Dataset):
    def __init__(self, data_path, flag):
        data_npy = os.path.join(data_path, 'trans_{}_data.npy'.format( flag))
        sample_label = os.path.join(data_path, '{}_label.pkl'.format(flag))
        sample_len = os.path.join(data_path, '{}_sample_len.pkl'.format(flag))

        with open(sample_label, "rb") as fin:
            label_list = pickle.load(fin)

        with open(sample_len, "rb") as fin:
            lens_list = pickle.load(fin)

        
        
        if 'ntu120' in data_path:
            self.data = np.load(data_npy, mmap_mode='r')[:, :, 0:75]
        else:
            self.data = np.load(data_npy, mmap_mode='r')
        
        self.label_list = label_list
        self.lens_list =  lens_list


    def __getitem__(self, index):
        lens = self.lens_list[index]
        sequence =  torch.tensor(self.data[index][:lens], dtype=torch.float)
        sequence, lens = downsample(sequence)

        label = self.label_list[index]
        

        return sequence,  lens, label, index

    def __len__(self):
        return len(self.label_list)



class MyDataset_uwa3d(Dataset):
    def __init__(self, data_path, flag):
        data_npy = os.path.join(data_path, '{}_data.npy'.format(flag))
        sample_label = os.path.join(data_path, '{}_label.pkl'.format(flag))
        sample_len = os.path.join(data_path, '{}_lens.pkl'.format(flag))

        with open(sample_label, "rb") as fin:
            label_list = pickle.load(fin)

        with open(sample_len, "rb") as fin:
            lens_list = pickle.load(fin)

        
        self.data = np.load(data_npy)
        self.label_list = label_list
        self.lens_list =  lens_list


    def __getitem__(self, index):
        lens = self.lens_list[index]
        sequence =  torch.tensor(self.data[index][:lens], dtype=torch.float)
        sequence, lens = downsample(sequence)
        label = self.label_list[index]
        

        return sequence,  lens, label, index

    def __len__(self):
        return len(self.label_list)


# ==================== sbu

SETS = ['s01s02','s01s03','s01s07','s02s01','s02s03','s02s06','s02s07','s03s02',
        's03s04','s03s05','s03s06','s04s02','s04s03','s04s06','s05s02','s05s03',
        's06s02','s06s03','s06s04','s07s01','s07s03']

FOLDS = [
    [ 1,  9, 15, 19],
    [ 5,  7, 10, 16],
    [ 2,  3, 20, 21],
    [ 4,  6,  8, 11],
    [12, 13, 14, 17, 18]]

ACTIONS = ['Approaching','Departing','Kicking','Punching','Pushing','Hugging',
        'ShakingHands','Exchanging']

def denormalize(norm_coords):
    """ SBU denormalization
        original_X = 1280 - (normalized_X .* 2560);
        original_Y = 960 - (normalized_Y .* 1920);
        original_Z = normalized_Z .* 10000 ./ 7.8125;
    """
    denorm_coords = np.empty(norm_coords.shape)
    denorm_coords[:, 0] = 1280 - (norm_coords[:, 0] * 2560)
    denorm_coords[:, 1] = 960 - (norm_coords[:, 1] * 1920)
    denorm_coords[:, 2] = norm_coords[:, 1] * 10000 / 7.8125

    return denorm_coords

def parse_sbu_txt(pose_filepath, normalized=False):
    video_poses_mat = np.loadtxt(pose_filepath, delimiter=',', usecols=range(1, 91))

    video_poses = []
    for frame_pose in video_poses_mat:
        people = []
        # 2 persons * 15 joints * 3 dimensions
        people_poses = frame_pose.reshape(2, 45)
        for person in people_poses:
            if normalized:
                per = person.reshape(15, 3)
            else:
                per = denormalize(person.reshape(15, 3))
            # per['confs'] = 15 * [1]
            people.append(per)
        video_poses.append(people)

    return np.array(video_poses)

def get_ground_truth(data_dir ):
    max_frams = 0
    setname_lst, fold_lst, seq_lst, action_lst, path_lst, frames_lst = [], [], [], [], [] ,[]
    for set_id, set_name in enumerate(SETS):
        for action_id in range(len(ACTIONS)):
            search_exp = '{}/{}/{:02}/*'.format(data_dir, set_name, action_id + 1)
            paths = glob.glob(search_exp)
            paths.sort()
            for path in paths:
                seq = path.split('/')[-1]
                fold = np.argwhere([set_id + 1 in lst for lst in FOLDS])[0, 0]
                frames = len(parse_sbu_txt(path + '/skeleton_pos.txt'))
                max_frams = max(max_frams, frames)

                setname_lst.append(set_name)
                fold_lst.append(fold)
                seq_lst.append(seq)
                action_lst.append(action_id)
                path_lst.append(path + '/skeleton_pos.txt')
                frames_lst.append(frames)

    dataframe_dict = {'set_name': setname_lst,
                    'fold': fold_lst,
                    'seq': seq_lst,
                    'path': path_lst,
                    'action': action_lst,
                    'frames': frames_lst
                    }
    ground_truth = pd.DataFrame(dataframe_dict)
    return ground_truth, max_frams


def get_train_gt(fold_num, ground_truth):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: " + str(fold_num))

    # ground_truth, _ = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold != fold_num]

    return gt_split


def get_val_gt(fold_num, ground_truth):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: " + str(fold_num))

    # ground_truth, _ = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold == fold_num]

    return gt_split

def subtract(data_numpy, target_joint):
	T, C, V = data_numpy.shape
	x_new = np.zeros((T, C, V ))
	for i in range(V):
		x_new[:, :, i] = data_numpy[:, :, i] - data_numpy[:, :, target_joint]
	return x_new

def subtract_torch(data, target_joint):
	T, CV = data.size()
	x_new = torch.zeros((T, CV ))
	for i in range(25):
		x_new[:, i : i+3] = data[:, i : i+3] - data[:, target_joint : target_joint + 3]
	return x_new


def ntu_tranform_skeleton(test):
    """
    :param test: frames of skeleton within a video sample
    """
    remove_frame = False
    test = np.asarray(test)
    transform_test = []
    
    d = test[0,2*3:2*3+3]
    
    v1 = test[0,1*3:1*3+3]-test[0,2*3:2*3+3]
    v1 = v1/np.linalg.norm(v1)
    
    v2_ = test[0,9*3:9*3+3]-test[0,12*3:12*3+3]
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
        for j in range(15):
            if test[i][j*3:j*3+3].all()==0:
                remove_frame = True
                break
            xyz = np.squeeze(np.matmul(np.linalg.inv(R),np.reshape(test[i][j*3:j*3+3]-d,(3,1))))
            xyzs.append(xyz)
        if not remove_frame:
            xyzs = np.reshape(np.asarray(xyzs),(-1,45))
            transform_test.append(xyzs)
        else:
            remove_frame = False
    
    transform_test = np.squeeze(np.asarray(transform_test))
    return transform_test

class MyDataset_sbu(Dataset):
    def __init__(self, data_path, fold, train_or_val):

        self.root_dir = data_path
        self.fold = fold
        self.train_or_val = train_or_val

        self.all_gt, self.max_frame = get_ground_truth(self.root_dir)

        if self.train_or_val == 'train':
            self.gt = get_train_gt(self.fold, self.all_gt)
        elif self.train_or_val == 'val':
            self.gt = get_val_gt(self.fold, self.all_gt)
        else:
            raise  ValueError('no such flag')


        # data_npy = os.path.join(data_path, '{}_data.npy'.format(flag))
        # sample_label = os.path.join(data_path, '{}_label.pkl'.format(flag))
        # sample_len = os.path.join(data_path, '{}_lens.pkl'.format(flag))

        # with open(sample_label, "rb") as fin:
        #     label_list = pickle.load(fin)

        # with open(sample_len, "rb") as fin:
        #     lens_list = pickle.load(fin)

        
        # self.data = np.load(data_npy)
        # self.label_list = label_list
        # self.lens_list =  lens_list


    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):

        raw_data = parse_sbu_txt(self.gt.iloc[index].path)
        T, M, V, C = raw_data.shape
        raw_data = raw_data.transpose([0, -1, -2, 1]) # T, C, V, M
        raw_data = raw_data[:,:,:,0]


        raw_data = subtract(raw_data,0 ).reshape([T, -1]) # train loss 1.6w+
        # raw_data = ntu_tranform_skeleton(raw_data.reshape([T, -1])) # train loss 2w+


        sequence = torch.tensor(raw_data, dtype=torch.float) # C, T, V
        sequence, lens = downsample(sequence, 40)

        # data_numpy = np.zeros([C, self.max_frame, V, M])
        # data_numpy[:, :T , :, :] = raw_data

        label = self.gt.iloc[index].action 

        # lens = self.gt.iloc[index].frames

        # ===================
        # lens = self.lens_list[index]
        # sequence =  torch.tensor(self.data[index][:lens], dtype=torch.float)
        # sequence, lens = downsample(sequence)
        # label = self.label_list[index]
        

        return sequence,  lens, label, index

   


