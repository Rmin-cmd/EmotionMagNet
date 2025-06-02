import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import os
from glob import glob
import re
from tqdm import tqdm


def read_mat_file(mat_path):

    if mat_path.endswith('.mat'):

        arr = sio.loadmat(mat_path)['DTF']['matrix'][0][0]

        arr_norm = np.mean(arr, axis=2)
        #
        arr_norm /= np.max(arr_norm, axis=0)

        np.fill_diagonal(arr_norm, 0)

    return arr_norm


def str_num(str):
    number = re.search(r'\d+', str)
    return int(number.group())


def matfile_loader(data_path, vid_ord):

    data_out = []

    for vid in tqdm(vid_ord):

        paths = sorted(glob(os.path.join(data_path, vid) + '\*'))

        freq_sub_data = []

        for path in paths:

            path_ = sorted(glob(path + '\*'))

            subs = np.unique([os.path.basename(path)[:6] for path in path_])

            sub_data = []

            for sub in subs:

                specified_ = [sub in path for path in path_]

                specified_paths = [item for item, flag in zip(path_, specified_) if flag]

                unique_vids = np.unique([str_num(os.path.basename(path_sp)[7:9]) for path_sp in specified_paths]).tolist()

                data_sub = np.array([read_mat_file(path_data) for path_data in specified_paths])

                sub_data.append(data_sub.reshape([len(unique_vids), data_sub.shape[0] // len(unique_vids),
                                                  data_sub.shape[1], data_sub.shape[2]]))

            freq_sub_data.append(sub_data)

        data_out.append(np.array(freq_sub_data).swapaxes(1, 0))

    return np.concatenate(data_out, axis=2)


# path_root = 'D:\proposal and thesis\Emotion recognition dataset\FACED dataset\Clisa_data\PDC_connectivity_30'
path_root = 'D:\proposal and thesis\Emotion recognition dataset\FACED dataset\Clisa_data\PDC_connectivity_30_mod'

out_path = 'preprocessed_data\preprocessed_connectivity\processed_conn_30_mod_3.mat'

vid_order = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']

out = matfile_loader(data_path=path_root, vid_ord=vid_order)

dic = {'data':out, 'label':'FACED_PDC'}

sio.savemat(out_path, dic)
