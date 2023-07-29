from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import sys

from cv2 import cv2
import mediapipe as mp
import json
import glob
from pathlib import Path
import pprint as pp
from turtle import right
from keypoints_normalization.image_cropping import ImageCroppingNormalization
from keypoints_normalization.recalculate_keypoints import RecalculateNormalization
from keypoints_normalization.recalculate_keypoints2 import RecalculateNormalization2
from moviepy.editor import VideoFileClip

class Feeder_hsd(Dataset):
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
                 data,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=1,
                 num_person_out=1,
                 max_frame = 300,
                 num_joint = 115,
                 ):
        self.data = data
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample
        self.max_frame = max_frame
        self.num_joint = num_joint

        self.load_data()

    def load_data(self):
        self.sample_name = ["test_webcam"]
        # print(self.sample_name)

        # output data shape (N, C, T, V, M)
        self.N = 1  # sample
        self.C = 3  # channel
        self.T = self.max_frame  # frame
        self.V = self.num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        video_info = self.data

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

        ## sort by score
        # sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        # for t, s in enumerate(sort_index):
        #     data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
        #                                                                0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./save_models/"+arg.Experiment_name
        arg.work_dir = "./work_dir/"+arg.Experiment_name
        self.arg = arg
        # self.save_arg()
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
        self.load_model()
        self.load_optimizer()
        # self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_epoch = 0

    def import_class(self, name):
        components = name.split('.')
        mod = __import__(components[0])  # import return model
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod
    
    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = self.import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            try:
                # print(weights)
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
    def eval(self, data, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        self.model.eval()
        loss_value = []
        score_frag = []
        right_num_total = 0
        total_num = 0
        loss_total = 0
        data = torch.from_numpy(data)
        data = Variable(
            data.float().cuda(self.output_device),
            requires_grad=False,
            volatile=True)

        with torch.no_grad():
            output = self.model(data)
        score_frag.append(output.data.cpu().numpy())
        _, predict_label = torch.max(output.data, 1)
        score = np.concatenate(score_frag)
        return predict_label, score

class Eval:
    max_frame = 300
    num_person_out = 1
    num_person_in = 1
    keypoints_normalization_method = "image_cropping"

    def __init__(self, config_path, video_path):
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        self.num_joint = 115
        self.max_frame = 300
        self.num_person_out = 1
        self.num_person_in = 1
        self.config_path = config_path
        self.video_path = video_path
        self.labels = ['akan',
                        'anda',
                        'apa',
                        'atau',
                        'baca',
                        'bagaimana',
                        'bahwa',
                        'beberapa',
                        'besar',
                        'bisa',
                        'buah',
                        'dan',
                        'dari',
                        'dengan',
                        'dia',
                        'haus',
                        'ingin',
                        'ini',
                        'itu',
                        'jadi',
                        'juga',
                        'kami',
                        'kata',
                        'kecil',
                        'kumpul',
                        'labuh',
                        'lain',
                        'laku',
                        'lapar',
                        'main',
                        'makan',
                        'masing',
                        'mereka',
                        'milik',
                        'minum',
                        'oleh',
                        'pada',
                        'rumah',
                        'satu',
                        'saya',
                        'sebagai',
                        'tambah',
                        'tangan',
                        'tetapi',
                        'tidak',
                        'tiga',
                        'udara',
                        'untuk',
                        'waktu',
                        'yang']

    def str2bool(self, v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def get_parser(self):
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
            type=self.str2bool,
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
            type=self.str2bool,
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
            '--nesterov', type=self.str2bool, default=False, help='use nesterov or not')
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
        parser.add_argument('--video', default=None)
        parser.add_argument('--data_path', default=None)
        return parser

    def config_preprocess(self, config):
        if "data_path" not in config:
            config["data_path"] = None
        if "video" not in config:
            config["video"] = None
        if "warm_up_epoch" not in config:
            config["warm_up_epoch"] = 0
        if "only_train_epoch" not in config:
            config["only_train_epoch"] = 0
        if "only_train_part" not in config:
            config["only_train_part"] = True
        if "weight_decay" not in config:
            config["weight_decay"] = float(0.0005)
        if "num_epoch" not in config:
            config["num_epoch"] = int(80)
        if "start_epoch" not in config:
            config["start_epoch"] = 0
        if "test_batch_size" not in config:
            config["test_batch_size"] = int(256)
        if "batch_size" not in config:
            config["batch_size"] = int(256)
        if "nesterov" not in config:
            config["nesterov"] = False
        if "optimizer" not in config:
            config["optimizer"] = 'SGD'
        if "device" not in config:
            config["device"] = [0]
        if "step" not in config:
            config["step"] = [20, 40, 60]
        if "base_lr" not in config:
            config["base_lr"] = float(0.01)
        if "ignore_weights" not in config:
            config["ignore_weights"] = []
        if "weights" not in config:
            config["weights"] = None
        if "model_args" not in config:
            config["model_args"] = dict()
        if "model" not in config:
            config["model"] = None
        if "test_feeder_args" not in config:
            config["test_feeder_args"] = dict()
        if "train_feeder_args" not in config:
            config["train_feeder_args"] = dict()
        if "num_worker" not in config:
            config["num_worker"] = 32
        if "feeder" not in config:
            config["feeder"] = 'feeder.feeder'
        if "show_topk" not in config:
            config["show_topk"] = [1, 5]
        if "print_log" not in config:
            config["print_log"] = True
        if "eval_interval" not in config:
            config["eval_interval"] = 5
        if "save_interval" not in config:
            config["save_interval"] = 2
        if "log_interval" not in config:
            config["log_interval"] = 100
        if "seed" not in config:
            config["seed"] = 1
        if "save_score" not in config:
            config["save_score"] = False
        if "phase" not in config:
            config["phase"] = 'train'
        if not self.config_path:
            config["config"] = './config/nturgbd-cross-view/test_bone.yaml'
        if self.config_path:
            config["config"] = self.config_path
        if "Experiment_name" not in config:
            config["Experiment_name"] = ''
        if "model_saved_name" not in config:
            config["model_saved_name"] = ''
        if "work_dir" not in config:
            config["work_dir"] = './work_dir/temp'
        return config

    def gendata(self, data,
                num_person_in=num_person_in,  # observe the first 1 persons
                num_person_out=num_person_out,  # then choose 1 persons with the highest score
                max_frame=max_frame):
        # print(num_person_in, num_person_out, max_frame)
        feeder = Feeder_hsd(
                    data=data,
                    num_person_in=num_person_in,
                    num_person_out=num_person_out,
                    window_size=max_frame,
                    max_frame = 300,
                    num_joint = 115,
                )

        sample_name = feeder.sample_name
        sample_label = []

        fp = np.zeros((1, 3, max_frame, self.num_joint, num_person_out), dtype=np.float32)

        for i, _ in enumerate(sample_name):
            data = feeder[i]
            fp[i, :, 0:data.shape[1], :, :] = data


        # np.save(data_out_path, fp)
        return fp


    def make_json(self, data, cls, is_kinetics):
        ord_A = ord("A")
        dict_data = {"data":[],"label":cls,"label_index":ord(cls)-ord_A}
        z_data = []
        for i,frame in enumerate(data):
            dict_data["data"].append({"frame_index":i+1, "skeleton":[{"pose":[]}]})
            for keypoint in frame:
                dict_data["data"][i]["skeleton"][0]["pose"].append(round(keypoint[0],4))
                dict_data["data"][i]["skeleton"][0]["pose"].append(round(keypoint[1],4))
                z = round(abs(keypoint[2]),4)
                if not is_kinetics:
                    dict_data["data"][i]["skeleton"][0]["pose"].append(z)
                else:
                    z_data.append(z)

                if is_kinetics:
                    dict_data["data"][i]["skeleton"][0]["score"] = z_data

        return dict_data

    def get_rank(self, data,index):
        ord_A = ord("A")
        return [[chr(ord_A+index[0]),data[index[0]]],[chr(ord_A+index[1]),data[index[1]]],[chr(ord_A+index[2]),data[index[2]]],[chr(ord_A+index[3]),data[index[3]]],[chr(ord_A+index[4]),data[index[4]]]]


    def levenshtein(self, a, b):
        if not a: return len(b)
        if not b: return len(a)
        return min(self.levenshtein(a[1:], b[1:])+(a[0] != b[0]),
                self.levenshtein(a[1:], b)+1,
                self.levenshtein(a, b[1:])+1)

    def get_video_duration(self, video_path):
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            return duration
        except Exception as e:
            print("Error getting video duration:", str(e))
            return None

    def eval(self):
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        config = self.config_preprocess(config)
        config = Struct(**config)
        processor = Processor(config)
        frame_counter = 0
        xyz = []
        ord_A = ord("A")
        dict_data = {"data":[],"label": "?","label_index":ord("?")-ord_A}
        duration = self.get_video_duration(self.video_path)
        # print(duration, self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        
        frame_process = 120
        frame_counter = 0
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic()
        image_crop_normalization = ImageCroppingNormalization(holistic)
        recalculation_normalization = RecalculateNormalization2(mp_holistic)
        # print(self.keypoints_normalization_method)
        predicted_array = []
        normalization_duration = 0
        prediction_duration = 0
        with holistic:
            while cap.isOpened():
                ret,frame = cap.read()
                frame_counter += 1
                if ret == True:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    start_normalization_process = time.time()
                    if self.keypoints_normalization_method == "image_cropping":
                        coordinates_data = image_crop_normalization.normalize(image)
                    elif self.keypoints_normalization_method == "recalculate_coordinates":
                         coordinates_data = recalculation_normalization.normalize(image)
                    end_normalization_process = time.time()
                    normalization_duration += end_normalization_process - start_normalization_process
                    dict_data["data"].append({"frame_index": frame_counter, "skeleton":[{"pose":[]}]})
                    dict_data["data"][frame_counter - 1]["skeleton"][0]["pose"] = coordinates_data
                    key = cv2.waitKey(1)
                    if frame_counter == frame_process:
                        ord_A = ord("A")
                        start_predict_process = time.time()
                        np_data = self.gendata(dict_data,max_frame=self.max_frame,num_person_in=self.num_person_in,num_person_out=self.num_person_out)
                        predicted, score = processor.eval(np_data)
                        end_predict_process = time.time()
                        prediction_duration += end_predict_process - start_predict_process
                        print("PREDICTED:", self.labels[predicted])
                        # print("SCORE:",score)
                        
                        if len(predicted_array) != 0:
                            if predicted != predicted_array[-1]:
                                predicted_array.append(predicted)
                        else:
                            predicted_array.append(predicted)

                        frame_counter = 0
                        xyz=[]
                        dict_data = {"data":[],"label": "?","label_index":ord("?")-ord_A}
                    if key == ord("q"):
                        break
                else:
                    break
        cap.release()
        cv2.destroyAllWindows()
        pred_str = ""
        if len(predicted_array) > 0:
            for i in predicted_array:
                pred_str += self.labels[i] + " "
        if self.video_path:
            print("FINAL PREDICTION:", pred_str)
            data = {
                "predicted": pred_str,
                "video_duration": duration,
                "normalization_elapsed_time": round(normalization_duration, 2),
                "prediction_elapsed_time": round(prediction_duration, 2),
            }
            return data

class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)