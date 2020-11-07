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
# from PCNet_LSTM import *
from PCNet import autoencoder
from utilitiesPC import *
from torch import optim
import torch.nn.functional as F
from data_loaderPC import *
import h5py
import numpy as np

import time
import math
from torch.utils.data import random_split
import torchvision
import sys
from tqdm import tqdm
import faiss
from utilitiesPC import reverse_sequence
from pc_test import eval

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train_iter(input_tensor, seq_len, index,  model, optimizer, criterion_seq, cluster_result, reverse_flag):
    optimizer.zero_grad()
    
    en_hi, de_out, output_proto, target_proto = model(input_tensor, seq_len, index,  cluster_result)


    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    mask = torch.sum(mask, 1)

    if reverse_flag :
        input_tensor_reverse = reverse_sequence(input_tensor, seq_len)
        total_loss = torch.sum(criterion_seq(de_out, input_tensor_reverse), 2)
    else:
        total_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)

    total_loss = torch.mean(torch.sum(total_loss, 1) / mask)
    
    if cluster_result != None:
        # ProtoNCE loss
      if output_proto is not None:
          criterion = nn.CrossEntropyLoss().cuda()
          loss_proto = 0
          for proto_out,proto_target in zip(output_proto, target_proto):
              loss_proto += criterion(proto_out, proto_target)  
              # accp = accuracy(proto_out, proto_target)[0] 
              # acc_proto.update(accp[0], images[0].size(0))

          # average loss across all sets of prototypes
          loss_proto /= len(cluster_result) 
          total_loss += loss_proto 



    total_loss.backward()
    clip_grad_norm_(model.parameters(), 25 , norm_type=2)
    
    optimizer.step()
    return total_loss, en_hi

def train_iter_classifier(input_tensor, seq_len,  model, classifier, optimizer, criterion_seq):
    
    
    with torch.no_grad():
      en_hi, de_out = model(input_tensor, seq_len)
    output = classifier(en_hi)

    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    
    mask = torch.sum(mask, 1)

    total_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
    total_loss = torch.mean(torch.sum(total_loss, 1) / mask)
    
    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm_(model.parameters(), 25 , norm_type=2)
    
    optimizer.step()
    return total_loss, en_hi

def eval_iter(input_tensor, seq_len, model, criterion_seq):
    
    en_hi, de_out, _, _ = model(input_tensor, seq_len )

    mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
    for ith_batch in range(len(seq_len)):
        mask[ith_batch, 0:seq_len[ith_batch]] = 1
    mask = torch.sum(mask, 1)
    
    total_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
    total_loss = torch.mean(torch.sum(total_loss, 1) / mask)

    return total_loss, en_hi


def evaluation(validation_loader, model, criterion_seq):
    total_loss = 0
    flag = 0
    for ind, (eval_data, seq_len, label, _) in enumerate(validation_loader):
        # print('indddddddddddddd')
        input_tensor = eval_data.to(device)
        # if torch.isnan(input_tensor).sum()  > 0:
        #     flag += 1
        #     continue
        # print(torch.isnan(input_tensor).sum())
        loss, hid = eval_iter(input_tensor, seq_len, model, criterion_seq)
        total_loss += loss.item()
        # print(total_loss)

    # print('total number of batch has nan: {}x32'.format(flag) )
    ave_loss = total_loss / (ind + 1)
    return ave_loss



def test_extract_hidden(model, data_train, data_eval):
    for ith, (ith_data, seq_len, label, _) in enumerate(data_train):
        input_tensor = ith_data.to(device)
        
        en_hi, de_out, _, _ = model(input_tensor, seq_len,  )


        if ith == 0:
            label_train = label
            hidden_array_train = en_hi[0, :, :].detach().cpu().numpy()

        else:
            label_train = np.hstack((label_train, label))
            hidden_array_train = np.vstack((hidden_array_train, en_hi[0, :, :].detach().cpu().numpy()))

    for ith, (ith_data, seq_len, label, _) in enumerate(data_eval):

        input_tensor = ith_data.to(device)

        en_hi, de_out, _, _ = model(input_tensor, seq_len)

        if ith == 0:
            hidden_array_eval = en_hi[0, :, :].detach().cpu().numpy()
            label_eval = label
        else:
            label_eval =  np.hstack((label_eval, label))
            hidden_array_eval = np.vstack((hidden_array_eval, en_hi[0, :, :].detach().cpu().numpy()))

    return hidden_array_train, hidden_array_eval, label_train, label_eval


def train_autoencoder(hidden_train, hidden_eval, label_train,
                      label_eval, middle_size, criterion, lambda1, num_epoches):
  batch_size = 64
  auto = autoencoder(hidden_train.shape[1], middle_size).to(device)
  auto_optimizer = optim.Adam(auto.parameters(), lr = 0.001)
  auto_scheduler = optim.lr_scheduler.LambdaLR(auto_optimizer, lr_lambda=lambda1)
  criterion_auto = nn.MSELoss()

  autodataset = MyAutoDataset(hidden_train, label_train)
  trainloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

  autodataset = MyAutoDataset(hidden_eval, label_eval)
  evalloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

  for epoch in range(num_epoches):
    for (data, label) in trainloader:
      # img, _ = data
      # img = img.view(img.size(0), -1)
      # img = Variable(img).cuda()
      #data = torch.tensor(data.clone().detach(), dtype=torch.float).to(device)
      # ===================forward=====================
      data = data.to(device)
      output, _ = auto(data)
      loss = criterion(output, data)
      # ===================backward====================
      auto_optimizer.zero_grad()
      loss.backward()
      auto_optimizer.step()
      auto_scheduler.step()
  # ===================log========================
    for (data, label) in evalloader:
      data = data.to(device)
      # ===================forward=====================
      output, _ = auto(data)
      loss_eval = criterion(output, data)
    # if epoch % 200 == 0:
    #   print('epoch [{}/{}], train loss:{:.4f} eval loass:{:.4f}'
    #         .format(epoch + 1, num_epoches, loss.item(), loss_eval.item()))
      
   ## extract hidden train
  count = 0
  for (data, label) in trainloader:  
    data = data.to(device)
    _, encoder_output = auto(data)

    if count == 0:
      np_out_train = encoder_output.detach().cpu().numpy()
      label_train = label
    else:
      label_train = np.hstack((label_train, label))
      np_out_train = np.vstack((np_out_train, encoder_output.detach().cpu().numpy())) 
    count += 1
  
  ## extract hidden eval
  count = 0
  for (data, label) in evalloader:
    data = data.to(device)
    _, encoder_output = auto(data)

    if count == 0:
      np_out_eval = encoder_output.detach().cpu().numpy()
      label_eval = label

    else:
      label_eval = np.hstack((label_eval, label))
      np_out_eval = np.vstack((np_out_eval, encoder_output.detach().cpu().numpy()))
    count += 1
 
  return np_out_train, np_out_eval, label_train, label_eval



def clustering_knn_acc(model, train_loader, eval_loader, criterion , num_epoches = 50, middle_size = 125):
    hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader)
    #print(hi_train.shape)

    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 50)

    np_out_train, np_out_eval, au_l_train, au_l_eval = train_autoencoder(hi_train, hi_eval, label_train,
                      label_eval, middle_size, criterion, lambda1, num_epoches)


    
    # knn_acc_1 = knn(hi_train, hi_eval, label_train, label_eval, nn=1)
    knn_acc_1 = 0
    knn_acc_au = knn(np_out_train, np_out_eval, au_l_train, au_l_eval, nn=1)
    return knn_acc_1, knn_acc_au

def train_auto_directly(model, train_loader, eval_loader, criterion , num_epoches = 60, middle_size = 125):
    hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader)
    #print(hi_train.shape)

    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 50)
    np_out_train, np_out_eval, au_l_train, au_l_eval = train_autoencoder_linEval(hi_train, hi_eval, label_train,
                      label_eval, middle_size, criterion, lambda1, num_epoches)

    # knn_acc_1 = knn(hi_train, hi_eval, label_train, label_eval, nn=1)
    # knn_acc_1 = 0
    # knn_acc_au = knn(np_out_train, np_out_eval, au_l_train, au_l_eval, nn=1)
    return knn_acc_1, knn_acc_au

def compute_features(loader, model, arg_pcl):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(loader.dataset), arg_pcl['hidden_size']).cuda()
    for i, (data, lens, _, index) in enumerate(tqdm(loader)):
        with torch.no_grad():
            data = data.cuda(non_blocking=True)
            feat, _, _, _ = model(data, lens, index) 
            features[index] = feat
    # dist.barrier()        
    # dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()

def run_kmeans(x, args_kmeans, density_fixed = None):
    """
    Args:
        x: data to be clustered
        (len(eval_loader), low_dim)
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args_kmeans['num_cluster']):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args_kmeans['gpu']    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)     
        if density_fixed is  None:   
            density = np.zeros(k)
            for i,dist in enumerate(Dcluster):
                if len(dist)>1:
                    d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                    density[i] = d     
                    
            #if cluster only has one point, use the max to estimate its concentration        
            dmax = density.max()
            for i,dist in enumerate(Dcluster):
                if len(dist)<=1:
                    density[i] = dmax 

            density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
            density = args_kmeans['temperature'] * density/density.mean()  #scale the mean to temperature 
        else:
            density = density_fixed[seed]


        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids) # 3, k, d
        results['density'].append(density) # 3, k
        results['im2cluster'].append(im2cluster) # 3, len(all data)   
        
    return results

def training(epoch, train_loader, eval_loader, print_every,
             model, optimizer, criterion_seq, file_output,
             root_path, network, en_num_layers, hidden_size, 
             load_saved=False, eval_flag = None, num_class=10, 
             loss_file_name = None, few_knn=False, my_custom = True, 
             prototype = True, bi_flag = None, num_clusters = None, 
             negative_r = None, last_file_name = None, model_save_every = 10, reverse_flag = False, density_fixed = None):

    auto_criterion = nn.MSELoss()
    start = time.time()
    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 20)
    model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


    past_acc = 0

    # model_saved = 'pro{}_bi{}/seq2seq_model_{}/'.format(str(prototype)[0],str(bi_flag)[0],  eval_flag)
    model_saved = '{}/model_{}/'.format(eval_flag, last_file_name)
    exist_dir(model_saved)
    epoch_trainLoss_saved = model_saved 
    model_saved = root_path + model_saved

    # Load a saved model
    if load_saved:
        for item in os.listdir(model_saved):
          if item.startswith('%slayer%d_hid%d' % (network, en_num_layers, hidden_size)):
              # Pull the starting epoch from the file name
              start_epoch = int(item.split('epoch')[-1])
              epoch, ave_loss_train = load_checkpoint(model, optimizer, model_saved + item)

    train_loss_list = []
    epoch_list = []
    best_eval_list = []
    eval_epoch_list = []

    for ith_epoch in range(1, epoch + 1):
        epoch_list.append(ith_epoch)
        # ========== prototypical learning
        if prototype:
          if bi_flag:
            arg_pcl = {'hidden_size': hidden_size * 2}
          else:
            arg_pcl = {'hidden_size': hidden_size }

          features = compute_features(train_loader, model, arg_pcl)
          model.train()
          # num_clusters = [80,90,100]

          # placeholder for clustering result
          cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
          for num_cluster in num_clusters:
              cluster_result['im2cluster'].append(torch.zeros(len(train_loader),dtype=torch.long).cuda())
              cluster_result['centroids'].append(torch.zeros(int(num_cluster),hidden_size).cuda())
              cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda()) 

          # if args.gpu == 0:
          features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
          features = features.numpy()
          args_kmeans = {'num_cluster': num_clusters, 'gpu': 0, 'temperature': 0.2 }
          cluster_result = run_kmeans(features, args_kmeans, density_fixed)  #run kmeans clustering on master node
          # save the clustering result
          
              # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
        # ========== 

        if ith_epoch % print_every == 0 or ith_epoch == 1:
            
            if my_custom:
                pass
            #   with torch.no_grad():
                # ave_loss_train = evaluation(train_loader, model, criterion_seq)
              # knn_acc_1, knn_acc_au = train_auto_directly(model, train_loader, eval_loader, criterion= auto_criterion)
            else:
              ave_loss_train = evaluation(train_loader, model, criterion_seq)
              ave_loss_eval = evaluation(eval_loader,  model, criterion_seq)
              knn_acc_1, knn_acc_au = clustering_knn_acc(model, train_loader, eval_loader, criterion= auto_criterion)
              print('%s (%d %d%%) TrainLoss %.4f EvalLoss %.4f KnnACC W/O-AEC: %.4f W-AEC: %.4f' % (
                  timeSince(start, ith_epoch / epoch),
                  ith_epoch, ith_epoch / epoch * 100, ave_loss_train, ave_loss_eval, knn_acc_1, knn_acc_au))
              file_output.writelines('%.4f %.4f %.4f %.4f\n' %
                                    (ave_loss_train, ave_loss_eval, knn_acc_1,
                                      knn_acc_au))

              if knn_acc_1 > past_acc:
                  past_acc = knn_acc_1
                  for item in os.listdir(model_saved):
                      if item.startswith('%slayer%d_hid%d' % (network, en_num_layers, hidden_size)):
                          open(model_saved + item, 'w').close()  # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                          os.remove(model_saved + item)

                  path_model = model_saved + '%slayer%d_hid%d_epoch%d' % ( network, en_num_layers, hidden_size, ith_epoch)
                  save_checkpoint(model, epoch, optimizer, ave_loss_train, path_model)
        

        train_losses = AverageMeter()
        for it, (data, seq_len, label, index) in enumerate(train_loader):
            input_tensor = data.to(device)

            if not prototype:
                # print('no prototype')
                cluster_result = None
            
            total_loss, en_hid = train_iter(input_tensor, seq_len, index,  model, optimizer, criterion_seq, cluster_result, reverse_flag)
            train_losses.update(total_loss.item(), input_tensor.size(0))

        train_loss_list.append(train_losses.avg)

        if ith_epoch % print_every == 0 or ith_epoch == 1:
            print('%s (%d %d%%) TrainLoss %.2f' % (timeSince(start, ith_epoch / epoch),
                    ith_epoch, ith_epoch / epoch * 100, train_losses.avg))
            file_output.writelines('epoch%d %.2f \n' %
                                        (ith_epoch, train_losses.avg))

        if ith_epoch % model_save_every == 0:
            torch.save(cluster_result, epoch_trainLoss_saved + '{}_epoch{}_clusters.pth'.format(loss_file_name, ith_epoch) )
            path_model = model_saved + '%slayer%d_hid%d_epoch%d_trainLoss%.4f' % ( network, en_num_layers, hidden_size,  ith_epoch, train_losses.avg)
            save_checkpoint(model, ith_epoch, optimizer, train_losses.avg, path_model)  
            temp_best_eval_acc =  eval(path_model)
            best_eval_list.append(temp_best_eval_acc.cpu().numpy())
            eval_epoch_list.append(ith_epoch)

        
        model_scheduler.step()

        if ith_epoch % 10 == 0:
          filename = file_output.name 
          file_output.close()
          file_output = open(filename, 'a')
    
    np.save(epoch_trainLoss_saved + '{}_train_epochs.npy'.format(loss_file_name), np.array(epoch_list))
    np.save(epoch_trainLoss_saved + '{}_train_loss.npy'.format(loss_file_name), np.array(train_loss_list))
    np.save(epoch_trainLoss_saved + '{}_lin_eval_acc.npy'.format(loss_file_name), np.array(best_eval_list))
    np.save(epoch_trainLoss_saved + '{}_lin_eval_epoch.npy'.format(loss_file_name), np.array(eval_epoch_list))



    # return total_loss, en_hid
    print('=======last training model saved path')
    print(path_model.split('/')[-3], '/', path_model.split('/')[-2], '/',  path_model.split('/')[-1])
    epoch_at_max_lin_acc = np.argmax(np.array(best_eval_list))
    print("Best evaluation acc: epoch %d, lin eval acc %.1f"%(eval_epoch_list[epoch_at_max_lin_acc], np.max(np.array(best_eval_list))))
    return path_model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




def train_classifier(ith_epoch, train_loader, model, classifier, 
                   criterion_classifier, optimizer, tsne_flag = False ):
    
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for it, (data, seq_len, label, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_tensor = data.to(device)
        label = torch.from_numpy(label).cuda()
        with torch.no_grad():
            # print(input_tensor[0][0])
            en_hi, de_out, _, _ = model(input_tensor, seq_len) 

            en_hi = en_hi.squeeze().detach()
          
            if tsne_flag:
                from t_sne_test import tsne_draw
                tsne_draw(en_hi.cpu().numpy(), label.cpu().numpy())
                break

        output = classifier(en_hi)


        loss = criterion_classifier(output, label)

        # print('label: ',label)
        # print('loss', loss)
        acc1, acc5 = accuracy(output, label, topk=(1,5))

        losses.update(loss.item(), input_tensor.size(0))
        top1.update(acc1[0], input_tensor.size(0))
        top5.update(acc5[0], input_tensor.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if it % 500 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   ith_epoch, it, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()



    return top1.avg, top5.avg, losses.avg


def valid_classifier(ith_epoch, train_loader, model, classifier, 
                   criterion_classifier, optimizer, tsne_flag = False ):
    
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    with torch.no_grad():
      for it, (data, seq_len, label, _) in enumerate(train_loader):
          
          data_time.update(time.time() - end)

          input_tensor = data.to(device)
          label = torch.from_numpy(label).cuda()

          
          en_hi, de_out, _, _ = model(input_tensor, seq_len) 

          en_hi = en_hi.squeeze().detach()
            
          if tsne_flag:
            from t_sne_test import tsne_draw
            tsne_draw(en_hi.cpu().numpy(), label.cpu().numpy())
            break
          
          output = classifier(en_hi)

          loss = criterion_classifier(output, label)
          acc1, acc5 = accuracy(output, label, topk=(1,5))
          losses.update(loss.item(), input_tensor.size(0))
          top1.update(acc1[0], input_tensor.size(0))
          top5.update(acc5[0], input_tensor.size(0))



          batch_time.update(time.time() - end)
          end = time.time()

          if it % 500 == 0:
              print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    ith_epoch, it, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
              sys.stdout.flush()

      print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg




