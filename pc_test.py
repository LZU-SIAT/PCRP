# load file
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
import os
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import h5py
import numpy as np
import math
from torch.utils.data import random_split
import torchvision

import sys

from utilitiesPC import *
from data_loaderPC import *
from trainPC import *

# torch.manual_seed(10)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print('gpu num', torch.cuda.device_count(), 
torch.cuda.get_device_name(0) ,
torch.cuda.current_device() )

rnn_model = 'GRU'
if rnn_model == 'LSTM':
    from PCNet_LSTM import *
elif rnn_model == 'GRU':
    from PCNet import *

def density_copy(cluster_list, ele):
    if ele is None:
        return None
    density_list = []
    for i, num in enumerate(cluster_list):
        temp = np.array([ele] * int(num))
        density_list.append(temp)
    return density_list

def train():
    ## training procedure
    fix_weight = True
    fix_state = False
    teacher_force = False
    linear_eval = True 
    # hyperparameter
    hidden_size =1024
    
    en_num_layers = 3
    de_num_layers = 1
    print_every = 1
    learning_rate = 0.001


    # random_seed = 11
    
    dataset = 'ntu120' # ntu60, ntu120, ucla, uwa3d, sbu
    eval_flag = 'view' # 'subject', 'view'
    bi_flag = False
    reverse_flag = False
    prototype = True

    # [35, 40, 45], [40,50,60], [40, 70, 100], [90, 120, 150],[150,180,210]
    num_clusters =    [150,180,210]
    negative_r = 60
    
    batch_size = 64 # batch_size + negative_r < min(num_clusters)
    random_seed = 77
    epoch = 10
    model_save_every = 1

    density_value = None
    density_fixed = density_copy(num_clusters, density_value)


    # num_clusters = [100,150,200]
    # negative_r = 50
    # custom_flag = 'tea{}_epoch{}_batch{}_{}_Rev{}_LinEval{}'.format(str(teacher_force)[0], epoch, batch_size, rnn_model, 
    #                                                                 str(reverse_flag)[0], str(linear_eval)[0])
    



    # if prototype:
    #     if dataset == 'ntu60':
    #         batch_size = 32
    #     if dataset == 'ntu120':
    #         batch_size = 64
    # else:
    #     batch_size = 64


    if fix_weight:
        network = 'FW'

    if fix_state:
        network = 'FS'

    if not fix_state and not fix_weight:
        network = 'O'



    # global variable
    root = '/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/PCRP/pc_pytorch/'
    dataset_root = root + 'data_for_pytorch/'
    if 'ntu' in dataset:
        dataset_folder = '{}/cross_{}_data/'.format(dataset, eval_flag)
        feature_length = 75
        if '60' in dataset:
            num_class = 60
        elif '120' in dataset:
            num_class = 120
        else:
            raise ValueError
    elif  'ucla' in dataset:
        dataset_folder = '{}/'.format(dataset)
        feature_length = 60
        num_class = 10
    elif 'uwa3d' in dataset:
        dataset_folder = '{}/train12_test34/'.format(dataset)
        feature_length = 45
        num_class = 30
    elif 'sbu' in dataset:
        dataset_folder = '/data5/xushihao/data/SBU'
        feature_length = 45
        num_class = 8
    else:
        raise ValueError

    if 'sbu' in dataset:
        dataset_path = dataset_folder
    else:
        dataset_path = dataset_root + dataset_folder

    if 'ntu' in dataset:
        dataset_train = MyDataset_ntu(dataset_path, 'train')
        dataset_test = MyDataset_ntu(dataset_path, 'test')
    elif  'ucla' in dataset:
        data_path_train = dataset_path + 'UCLAtrain50.h5py'
        dataset_train = MyDataset(data_path_train)

        data_path_test = dataset_path + 'UCLAtest50.h5py'
        dataset_test = MyDataset(data_path_test)
    elif 'uwa3d' in dataset:
        uwa3d_train = ['12', '13', '14', '23', '24', '34', ]
        uwa3d_test1 = ['3','2','2','1','1','1']
        uwa3d_test2 = ['4','4','3','4','3','2']
        fold = 0 # index = 0 starts
        uwa3d_test_view = '3' # '3' or '4'

        data_path_train = dataset_path + 'training_12'
        dataset_train = MyDataset_uwa3d(data_path_train, 'train')

        data_path_test = dataset_path + 'test_{}'.format(uwa3d_test_view)
        dataset_test = MyDataset_uwa3d(data_path_test, 'val')
    elif 'sbu' in dataset:
        fold = 0 # 0-4
        dataset_train = MyDataset_sbu(dataset_path, fold, 'train')
        dataset_test = MyDataset_sbu(dataset_path, fold, 'val')







    shuffle_dataset = True
    dataset_size_train = len(dataset_train)
    dataset_size_test = len(dataset_test)

    indices_train = list(range(dataset_size_train))
    indices_test = list(range(dataset_size_test))


    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_test)

    print("training data length: %d, validation data length: %d" % (len(indices_train), len(indices_test)))
    # seperate train and validation

    train_sampler = SubsetRandomSampler(indices_train)
    valid_sampler = SubsetRandomSampler(indices_test)


    if 'ntu' in dataset:
        pad_style = pad_collate_ntu
    elif 'uwa3d' in dataset:
        pad_style = pad_collate_ntu
    elif 'ucla' in dataset_path:
        pad_style = pad_collate
    elif 'SBU' in dataset_path or 'sbu' in dataset_path:
        pad_style = pad_collate_ntu



    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                            sampler=train_sampler, collate_fn=pad_style )
    eval_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                sampler=valid_sampler, collate_fn=pad_style)


    # load model
    model = seq2seq(feature_length, hidden_size, feature_length, batch_size, 
                    en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force, 
                    bi_flag = bi_flag, negative_r = negative_r, reverse_flag = reverse_flag)



    # initilize weight
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    #                     nn.init.uniform_(param, a=-0.05, b=0.05)

    #check whether decoder gru weights are fixed
    if fix_weight:
        if rnn_model == 'GRU':
            print(model.decoder.gru.requires_grad)
        elif rnn_model == 'LSTM':
            print(model.decoder.lstm.requires_grad)



    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    criterion_seq = nn.L1Loss(reduction='none')

    last_file_name = 'bi{}_Rev{}_protp{}_{}_Rg{}_den{}_batch{}_epoch{}_{}_seed{}_LinEval{}'.format(str(bi_flag)[0], str(reverse_flag)[0], str(prototype)[0], 
                                                    str(num_clusters).replace(',', '_').replace(' ', ''), negative_r, density_value,
                                                    batch_size, epoch, network, random_seed, str(linear_eval)[0])
    # custom_flag = 'epoch{}_batch{}_{}_Rev{}_LinEval{}'.format( epoch, batch_size, rnn_model, 
    #                                                                 str(reverse_flag)[0], str(linear_eval)[0])
    # last_file_name = 'protp{}_bi{}_{}_{}_R{}_seed{}_{}'.format(str(prototype)[0], str(bi_flag)[0], network, 
    #                                     str(num_clusters).replace(',', '_').replace(' ', ''), negative_r, random_seed, custom_flag)

    log_path = root + '{}_{}/training_log_{}'.format( dataset, eval_flag, last_file_name)


    exist_dir(log_path)

    saved_name = '/{}sen{}d_hid{}d'.format(network, en_num_layers, hidden_size)
    file_output = open(log_path + '/{}.txt'.format(saved_name), 'w' ) 


    learned_model_path = training(epoch, train_loader, eval_loader, print_every,
                model, optimizer, criterion_seq,  file_output,
                root, network, en_num_layers, hidden_size, eval_flag = '{}_{}'.format(dataset, eval_flag), num_class=num_class, 
                loss_file_name = saved_name, prototype = prototype, bi_flag = bi_flag, 
                num_clusters = num_clusters, negative_r = negative_r, last_file_name = last_file_name, 
                model_save_every = model_save_every, reverse_flag = reverse_flag, my_custom = linear_eval, density_fixed = density_fixed
                )


    file_output.close()

    return learned_model_path


# =============== linear evaluation
def eval(trained_model):
    from trainPC import train_classifier, valid_classifier
    ## training procedure
    
    teacher_flag = trained_model[trained_model.find('tea') + 3]
    # if teacher_flag == 'T':
    #     teacher_force = True
    # elif teacher_flag == 'F':
    #     teacher_force = False
    # else:
    #     raise ValueError
    teacher_force = False
    fix_weight = True
    fix_state = False

    if fix_weight:
        network = 'FW'

    if fix_state:
        network = 'FS'

    if not fix_state and not fix_weight:
        network = 'O'

    # hyperparameter
    tsne_flag = False

    en_num_layers = 3
    de_num_layers = 1
    print_every = 1
    learning_rate = 0.01
    learning_rate_decay = 0.1
    epoch_decay = [5]

    hidden_size = 1024
    batch_size = 64
    epoch = 30

    random_seed = 11111
    

    # global variable
    root = '/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/PCRP/pc_pytorch/'

    # trained_model = '{}/{}/{}'.format(root,'pro{}_bi{}/seq2seq_model_{}'.format(str(prototype)[0],str(bi_flag)[0], dataset+"_"+ eval_flag), 'FWlayer3_hid1024_neg50_epoch50_trainLoss2.1381')
    # trained_model = '/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/PCRP/pc_pytorch/proT_biF/proT_biF_seq2seq_model_ntu60_subject/FWlayer3_hid1024_neg50_epoch50_trainLoss2.1381'
    
    if 'ucla' in trained_model:
        dataset = 'ucla'
        eval_flag = ''
    elif 'ntu' in trained_model:
        index = trained_model.find('ntu')
        if trained_model[index + 3] == '6':
            print(trained_model[index: index + 5])
            dataset = 'ntu60' # ntu60, ntu120, ucla
        elif trained_model[index + 3] == '1':
            print(trained_model[index: index+6])
            dataset = 'ntu120' # ntu60, ntu120, ucla

        if trained_model.find('subject') != -1:
            eval_flag = 'subject' # 'subject', 'view'
        else:
            eval_flag = 'view' # 'subject', 'view'

    elif 'uwa3d' in trained_model:
        dataset = 'uwa3d'
        eval_flag = ''
    elif 'sbu' in trained_model:
        dataset = 'sbu'
        eval_flag = ''
    else:
        raise ValueError


    if 'T' in trained_model[trained_model.find('protp')+5]:
        prototype = True
    elif 'F' in trained_model[trained_model.find('protp')+5]:
        prototype = False
    else:
        raise  ValueError


    if 'T' in trained_model[trained_model.find('bi')+2]:
        bi_flag = True
    elif 'F' in trained_model[trained_model.find('bi')+2]:
        bi_flag = False 
    else:
        raise  ValueError
    
    if bi_flag:
        lin_hidden_size = hidden_size * 2
    else:
        lin_hidden_size = hidden_size

    trained_name = trained_model.split('/')[-1]
    trained_epoch = trained_name[ trained_name.find('epoch')+5 : trained_name.find('epoch')+7]

    last_index = trained_model.find(trained_model.split('/')[-1])
    # last_file_name = trained_model[trained_model.find('protp'): last_index - 1]
    # last_file_name = "{}{}{}".format(last_file_name[0 : last_file_name.find('epoch') + 5], 
    #                                             trained_epoch, 
    #                                             last_file_name[last_file_name.find('epoch') + 7 :] ) 
    last_file_name = trained_model[trained_model.find('bi'): last_index - 1]
    last_file_name = "{}{}{}".format(last_file_name[0 : last_file_name.find('epoch') + 5], 
                                                trained_epoch, 
                                                last_file_name[last_file_name.find('epoch') + 7 :] ) 


    if 'ntu' in dataset:
        dataset_folder = 'data_for_pytorch/{}/cross_{}_data/'.format(dataset, eval_flag)
        feature_length = 75
        if '60' in dataset:
            num_class = 60
        elif '120' in dataset:
            num_class = 120
        else:
            raise ValueError
    elif  'ucla' in dataset:
        dataset_folder = 'data_for_pytorch/{}/'.format(dataset)
        feature_length = 60
        num_class = 10
    elif 'uwa3d' in dataset:
        dataset_folder = 'data_for_pytorch/{}/train12_test34/'.format(dataset)
        feature_length = 45
        num_class = 30
    elif 'sbu' in dataset:
        dataset_folder = '/data5/xushihao/data/SBU'
        feature_length = 45
        num_class = 8


    dataset_path = root + dataset_folder

    if 'ntu' in dataset:
        dataset_train = MyDataset_ntu(dataset_path, 'train')
        dataset_test = MyDataset_ntu(dataset_path, 'test')
    elif  'ucla' in dataset:
        data_path_train = dataset_path + 'UCLAtrain50.h5py'
        dataset_train = MyDataset(data_path_train)

        data_path_test = dataset_path + 'UCLAtest50.h5py'
        dataset_test = MyDataset(data_path_test)
    elif 'uwa3d' in dataset:
        uwa3d_train = ['12', '13', '14', '23', '24', '34', ]
        uwa3d_test1 = ['3','2','2','1','1','1']
        uwa3d_test2 = ['4','4','3','4','3','2']
        fold = 0 # index = 0 starts
        uwa3d_test_view = '3' # '3' or '4'

        data_path_train = dataset_path + 'training_12'
        dataset_train = MyDataset_uwa3d(data_path_train, 'train')

        data_path_test = dataset_path + 'test_{}'.format(uwa3d_test_view)
        dataset_test = MyDataset_uwa3d(data_path_test, 'val')
    elif 'sbu' in dataset:
        fold = 0 # 0-4
        dataset_train = MyDataset_sbu(dataset_path, fold, 'train')
        dataset_test = MyDataset_sbu(dataset_path, fold, 'val')

    shuffle_dataset = True
    dataset_size_train = len(dataset_train)
    dataset_size_test = len(dataset_test)

    indices_train = list(range(dataset_size_train))
    indices_test = list(range(dataset_size_test))


    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_test)

    print("training data length: %d, validation data length: %d" % (len(indices_train), len(indices_test)))
    # seperate train and validation
    train_sampler = SubsetRandomSampler(indices_train)
    # print(indices_test[0:4]) every time same
    valid_sampler = SubsetRandomSampler(indices_test)
    

    if 'ntu' in dataset:
        pad_style = pad_collate_ntu
    elif 'uwa3d' in dataset:
        pad_style = pad_collate_ntu
    elif 'ucla' in dataset_path:
        pad_style = pad_collate
    elif 'SBU' in dataset_path or 'sbu' in dataset_path:
        pad_style = pad_collate_ntu

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                            sampler=train_sampler, collate_fn=pad_style )
    eval_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                sampler=valid_sampler, collate_fn=pad_style)


    # load model
    model = seq2seq(feature_length, hidden_size, feature_length, batch_size, 
                    en_num_layers, de_num_layers, fix_state, fix_weight, teacher_force, bi_flag = bi_flag)
    classifier = LinearClassifier(lin_hidden_size , num_class)
    classifier = classifier.cuda()

    model_dict = model.state_dict()

    ckpt = torch.load(trained_model)
    load_state = ckpt['model_state_dict']



    # hard load
    model.load_state_dict(load_state)

    # soft load
    # state_dict = {k:v for k,v in load_state.items() if k in model_dict.keys()}
    # if len(state_dict.keys()) == 0:
    #     raise ImportError('load failure')
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)



    # initilize weight
    # with torch.no_grad():
    #     for child in list(model.children()):
    #         print(child)
    #         for param in list(child.parameters()):
    #             if param.dim() == 2:
    #                     nn.init.xavier_uniform_(param)
    #                     nn.init.uniform_(param, a=-0.05, b=0.05)

    #check whether decoder gru weights are fixed
    if fix_weight:
        if rnn_model == 'GRU':
            print(model.decoder.gru.requires_grad)
        elif rnn_model == 'LSTM':
            print(model.decoder.lstm.requires_grad)




    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    criterion_classifier = torch.nn.CrossEntropyLoss().cuda()

        
    model.eval()

    train_loss_list = []
    val_loss_list = []
    epoch_list  = []


    best_acc = 0

    # temp_path = '{}/pro{}_bi{}/lin_eval_{}_{}'.format( root, str(prototype)[0], str(bi_flag)[0], dataset,eval_flag)
    # temp_path = '{}/{}_{}/lin_eval_protp{}_bi{}'.format(root, dataset, eval_flag, str(prototype)[0], str(bi_flag)[0])
    temp_path = '{}/{}_{}/lin_eval_{}'.format(root, dataset, eval_flag, last_file_name)
    exist_dir(temp_path)

    # lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 10)
    # model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    saved_name = '{}sen{}d_hid{}d'.format(network, en_num_layers, hidden_size)
    delete_old_list = []
    for ith_epoch in range(1, epoch + 1):
        epoch_list.append(ith_epoch)

        adjust_learning_rate(ith_epoch, [epoch_decay, learning_rate, learning_rate_decay], optimizer)
        start = time.time()

        print("==> training...")

        time1 = time.time()

        train_acc, train_acc5, train_loss  = train_classifier(ith_epoch, train_loader, model, classifier, criterion_classifier, optimizer )



        train_loss_list.append(train_loss)

        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(ith_epoch, time2 - time1))

        print("==> testing...")

        test_acc, test_acc5, test_loss = valid_classifier(ith_epoch, eval_loader, model, classifier, criterion_classifier, optimizer, tsne_flag  )

        if tsne_flag:
            break

        val_loss_list.append(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                    'epoch': epoch,
                    'classifier': classifier.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
            save_name = '/{}_epoch_{}_bestAcc_top1_{:.1f}.pth'.format(saved_name, ith_epoch, float(best_acc.cpu().numpy()))
            save_name = temp_path + save_name
            # print(temp_path)
            # print(save_name)
            print('saving best model! best acc: {:.1f}'.format(float(best_acc.cpu().numpy())))
            if len(delete_old_list) > 0:
                os.remove(delete_old_list[-1])
            torch.save(state, save_name)
            delete_old_list.append(save_name)



    np.save(temp_path + '/{}_epochs_lin.npy'.format(saved_name), np.array(epoch_list))
    np.save(temp_path + '/{}_train_loss_lin.npy'.format(saved_name), np.array(train_loss_list))
    np.save(temp_path + '/{}_valid_loss_lin.npy'.format(saved_name), np.array(val_loss_list))

    return best_acc



if __name__ == "__main__":
    path = train()
    # path_temp = '/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/PCRP/pc_pytorch/ucla_/model_protpT_biF_FW_[100]_R60_seed11111_teaF_epoch100_batch32_GRU_RevT_LinEvalT/FWlayer3_hid1024_epoch50_trainLoss4.7777'
    # path = '/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/PCRP/pc_pytorch/ucla_/model_biF_RevT_protpT_[20]_Rg8_densityValNone_batch8_epoch90_FW_seed22_LinEvalT/FWlayer3_hid1024_epoch20_trainLoss6.7284'
    # path_best = '/data5/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/PCRP/pc_pytorch/ucla_/model_protpT_biF_FW_[40]_R20_seed11111_teaF_epoch50_batch16_GRU_RevT_LinEvalT/FWlayer3_hid1024_epoch50_trainLoss4.7426'
    print("===========Linear Evaluation")
    # eval(path)