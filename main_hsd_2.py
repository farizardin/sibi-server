#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
import json
from pathlib import Path
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Shift Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--data-path',
        default=None,
        help="the directory of the dataset"
    )

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg, num_fold, data_train_files, data_test_files):
        self.num_fold = num_fold
        arg.model_saved_name = "./save_models/"+arg.Experiment_name+"_fold_"+str(self.num_fold)
        arg.work_dir = "./work_dir/"+arg.Experiment_name+"_fold_"+str(self.num_fold)
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.data_train_files = data_train_files
        self.data_test_files = data_test_files
        
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_epoch = 0
        

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.print_log("NUM WORKER: "+str(self.arg.num_worker))
        if self.arg.phase == 'train':
            datagen = DatasetGenerator(
                 self.arg.data_path,
                 self.data_train_files,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=1,
                 num_person_out=1
            )
            data_train, sample_name_train, label_train = datagen.generate()
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args, 
                    data=data_train, 
                    sample_name=sample_name_train, 
                    label=label_train),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

        datagen = DatasetGenerator(
                 self.arg.data_path,
                 self.data_test_files,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=1,
                 num_person_out=1
            )
        data_test, sample_name_test, label_test = datagen.generate()
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                data=data_test, 
                sample_name=sample_name_test, 
                label=label_test),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0
                    
                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
            
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir+'/eval_results')
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        # self.print_log('before tqdm')
        process = tqdm(loader)
        # self.print_log('after tqdm')
        acc_all = []
        if epoch >= self.arg.only_train_epoch:
            for key, value in self.model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = True
                    self.print_log(key + '-require grad')
        else:
            for key, value in self.model.named_parameters():
                if 'PA' in key:
                    value.requires_grad = False
                    self.print_log(key + '-not require grad')
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            start = time.time()
            output = self.model(data)
            network_time = time.time()-start

            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data)
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_all.append(acc)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}  network_time: {:.4f}'.format(
                        batch_idx, len(loader), loss.data, self.lr, network_time))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        acc_all = torch.tensor(acc_all)
        acc_final = torch.mean(acc_all)
        self.print_log('\tTraining Accuracy: {:.2f}%'.format(acc_final*100))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        y_pred = []
        y_true = []
        labels = np.arange(26)
        labels = [chr(lab+ord("A")) for lab in labels]

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                print(type(data))
                data = Variable(
                    data.float().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)
                label = Variable(
                    label.long().cuda(self.output_device),
                    requires_grad=False,
                    volatile=True)

                with torch.no_grad():
                    output = self.model(data)

                loss = self.loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.cpu().numpy())

                _, predict_label = torch.max(output.data, 1)
                step += 1
                
                print("data inside ", data.data.cpu().numpy().shape)

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())

                    y_pred.append(predict)
                    y_true.append(true)

                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            if "hsd" in arg.Experiment_name and 'test' in arg.phase:
                y_pred = [i for row in y_pred for i in row]
                y_true = [i for row in y_true for i in row]
                # print(y_pred)
                # print(y_true)
                y_pred_chr = [chr(char+ord("A")) for char in y_pred]
                y_true_chr = [chr(char+ord("A")) for char in y_true]

                fig, ax = plt.subplots(figsize=(15, 15))
                # plot_confusion_matrix(slf_4, X_test, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
                # plt.show()

                cm = confusion_matrix(y_true_chr, y_pred_chr,labels=labels)
                cmd_obj = ConfusionMatrixDisplay(cm, display_labels=labels)
                cmd_obj.plot(ax=ax, cmap=plt.cm.Blues)
                cmd_obj.ax_.set(
                    title='Confusion Matrix', 
                    xlabel='Predicted', 
                    ylabel='Actual')
                plt.show()
                self.print_log(classification_report(y_true_chr, y_pred_chr, target_names=labels))

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_epoch = epoch
                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))

                with open('./work_dir/' + arg.Experiment_name+"_fold_"+str(self.num_fold) + '/eval_results/best_acc' +'.pkl'.format(
                        epoch, accuracy), 'wb') as f:
                    pickle.dump(score_dict, f)

            print('Eval Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            with open('./work_dir/' + arg.Experiment_name+"_fold_"+str(self.num_fold) + '/eval_results/epoch_' + str(epoch) + '_' + str(accuracy) + '.pkl'.format(
                    epoch, accuracy), 'wb') as f:
                pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                self.train(epoch, save_model=True)

                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

            print('best accuracy: ', self.best_acc, 'best epoch:', self.best_epoch, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test' or self.arg.phase == 'full_test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name+"_fold_"+str(self.num_fold) + '_wrong.txt'
                rf = self.arg.model_saved_name+"_fold_"+str(self.num_fold) + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

num_joint = 21
max_frame = 300
num_person_out = 1
num_person_in = 1


class DatasetGenerator(Dataset):
    """ Feeder for skeleton-based hand sign recognition in HandSign dataset
    # 21 keypoints list:
    # https://google.github.io/mediapipe/images/mobile/hand_landmarks.png

    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 sample_name,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=2,
                 num_person_out=5):
        self.data_path = data_path
        self.sample_name = sample_name
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.label_info = self.make_label()
        self.load_data()

    def load_data(self):
        # load file list
        # self.sample_name = os.listdir(self.data_path)

        # load label
        # label_path = self.label_path
        # with open(label_path) as f:
        #     label_info = json.load(f)

        # self.sample_name = [key+".json" for key in label_info.keys()]
        print(self.sample_name)
        sample_id = [name.split('.')[0] for name in self.sample_name]
        # sample_id = [name for name in self.sample_name]
        self.label = np.array([self.label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array([self.label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                # score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::3]
                data_numpy[1, frame_index, :, m] = pose[1::3]
                data_numpy[2, frame_index, :, m] = pose[2::3]

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info['label_index']
        # print(label)
        # print(self.label[index])
        assert (self.label[index] == label)

        ## sort by score
        # sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        # for t, s in enumerate(sort_index):
        #     data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
        #                                                                0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label

    def make_label(self):
        dict_data = {}
        ord_A = ord("A")

        for f in self.sample_name:
            # print(file)
            f = f.replace(".json","")
            class_name = f.split("_")[0]
            cls_index = ord(class_name) - ord_A
            dict_data[f] = {"has_skeleton":True,"label":class_name,"label_index":cls_index}
            
        return dict_data


    def generate(self):
        sample_name = self.sample_name
        sample_label = []

        fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)

        for i, _ in enumerate(tqdm(sample_name)):
            data, label = self[i]
            fp[i, :, 0:data.shape[1], :, :] = data
            sample_label.append(label)

        # with open(label_out_path, 'wb') as f:
        #     pickle.dump((sample_name, list(sample_label)), f)

        # np.save(data_out_path, fp)

        return fp, sample_name, sample_label


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)

    if arg.phase == 'full_test':
        src_path = arg.data_path
        files = sorted(os.listdir(src_path))
        files = np.array(files)
        processor = Processor(arg, 0, None, files)
        processor.start()

    else:
        # kfold separator -> train jsons + test jsons
        skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        src_path = arg.data_path
        files = sorted(os.listdir(src_path))
        labels = [ord(label.split("_")[0]) - ord("A") for label in files]
        files = np.array(files)

        skfold_json_path = src_path[:-1]+"_skfold/"
        skfold_json_name = "skfold.json"
        skfold_json_final_path = skfold_json_path + skfold_json_name
        
        if  os.path.exists(skfold_json_final_path):
            print("EXIST")
            with open(skfold_json_final_path, 'r') as f:
                kfolds = json.load(f)

        else:
            print("NOT EXIST")
            skf_split = skf.split(files, labels)
            kfolds = {}

            for i, (train_index, test_index) in enumerate(skf_split):
                kfolds['fold_{}'.format(i+1)] = {}
                kfolds['fold_{}'.format(i+1)]['train'] = files[train_index].tolist()
                kfolds['fold_{}'.format(i+1)]['test'] = files[test_index].tolist()

            Path(skfold_json_path).mkdir(parents=True, exist_ok=True)
            with open(skfold_json_final_path, 'w') as f:
                json.dump(kfolds,f)

        for i, (key,val) in enumerate(kfolds.items()):        
            ## create sample_name and label for each dataset
            ## generate with DatasetGenerator
            ## provide to processor
            train = np.array(val['train'])
            test = np.array(val['test'])
            processor = Processor(arg, i+1, train, test)
            processor.start()
