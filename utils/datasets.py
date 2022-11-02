import numpy as np


def load_preproc_data(name='src'):
    names_dict = {
        'src': "df1/preproc_dataset.npz",
        'tar1': "df2/preproc_dataset.npz",
        'tar2': "df3/preproc_dataset.npz",
        'tar3': "df4/preproc_dataset.npz",
        'bpm10_src': "BeijingPM10Quality/src.npz",
        'bpm10_tar': "BeijingPM10Quality/tar.npz",
    }
    filename = '../Data/'+names_dict[name]
    return np.load(filename)