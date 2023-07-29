import argparse
import os
import numpy as np
import json
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

num_joint = 115
# num_joint = 107
max_frame = 300
num_person_out = 1
num_person_in = 1


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
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=2,
                 num_person_out=5):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        # self.sample_name = os.listdir(self.data_path)

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        self.sample_name = [key+".json" for key in label_info.keys()]
        print(self.sample_name)
        sample_id = [name.split('.')[0] for name in self.sample_name]
        # sample_id = [name for name in self.sample_name]
        self.label = np.array([label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])

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
        assert (self.label[index] == label)

        ## sort by score
        # sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        # for t, s in enumerate(sort_index):
        #     data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
        #                                                                0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=num_person_in,  # observe the first 1 persons
            num_person_out=num_person_out,  # then choose 1 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder_hsd(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)

    for i, _ in enumerate(tqdm(sample_name)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HandSignDataset Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/hsd_raw')
    parser.add_argument(
        '--out_folder', default='../data/hsd')
    arg = parser.parse_args()

    part = ['train','val']#['val', 'train']
    for p in part:
        print('hsd ', p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = '{}/hsd_{}'.format(arg.data_path, p)
        label_path = '{}/hsd_{}_label.json'.format(arg.data_path, p)
        # label_path = '{}/hsd_lite_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        gendata(data_path, label_path, data_out_path, label_out_path)
