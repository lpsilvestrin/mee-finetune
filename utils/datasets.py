import numpy as np


def load_preproc_data(name='src', path='../Data/'):
    """
    the sequencial data should be in the tensorflow order (samples, steps, features)
    Args:
        name:
        path:

    Returns:

    """
    names_dict = {
        'src': "df1/preproc_dataset.npz",
        'tar1': "df2/preproc_dataset.npz",
        'tar2': "df3/preproc_dataset.npz",
        'tar3': "df4/preproc_dataset.npz",
        'bpm10_src': "BeijingPM10Quality/src.npz",
        'bpm10_tar': "BeijingPM10Quality/tar.npz",
    }
    assert name in names_dict, f"{name} are not in the know dataset list: {', '.join(names_dict.keys())}"
    filename = path+'/'+names_dict[name]
    return np.load(filename)