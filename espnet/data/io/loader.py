import h5py
import torchaudio
import numpy as np
import torch
from torchaudio import kaldi_io


def to_tensor(array):
    return torch.from_numpy(array)

def load_hdf5(filepath):
    # e.g.
    #    {"input": [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
    #                "filetype": "hdf5",
    # -> filepath = "some/path.h5", key = "F01_050C0101_PED_REAL"
    filepath, key = filepath.split(':', 1)
    with h5py.File(filepath, 'r') as h5_file:
        array = h5_file[key][()]
    return to_tensor(array)


def load_soundhdf5(filepath):
    # e.g.
    #    {"input": [{"feat": "some/path.h5:F01_050C0101_PED_REAL",
    #                "filetype": "sound.hdf5",
    # -> filepath = "some/path.h5", key = "F01_050C0101_PED_REAL"
    filepath, key = filepath.split(':', 1)
    with SoundHDF5File(filepath, 'r', dtype='int16') as h5_file:
        array, rate = h5_file[key]
    return to_tensor(array)


def load_audiofile(filepath):
    # e.g.
    #    {"input": [{"feat": "some/path.wav",
    #                "filetype": "sound"},
    # Assume PCM16
    tensor, _ = torchaudio.load(filepath, dtype='int16')
    return tensor

def load_compressed_numpy(filepath):
    # TODO: dataset: load files if not loaded (use cache) and return tensors[key] if loaded
    # e.g.
    #    {"input": [{"feat": "some/path.npz:F01_050C0101_PED_REAL",
    #                "filetype": "npz",
    filepath, key = filepath.split(':', 1)
    data = np.load(filepath)
    return data[key]

def load_numpy(filepath):
    # e.g.
    #    {"input": [{"feat": "some/path.npy",
    #                "filetype": "npy"},
    return np.load(filepath)

def load_mat(filepath):
    # TODO: dataset: load files if not loaded (use cache) and return tensors[key] if loaded    # e.g.
    #    {"input": [{"feat": "some/path.ark:123",
    #                "filetype": "mat"}]},
    # In this case, "123" indicates the starting points of the matrix
    # load_mat can load both matrix and vector 
    # --> TODO: kaldi_io.read_mat_ark vs kaldi_io.read_vec_flt_ark
    return kaldi_io.read_mat_ark(filepath)  ## generator!!!

def load_vec(filepath):
    return kaldi_io.read_vec_flt_ark(filepath)

def load_scp(filepath):
    # e.g.
    #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
    #                "filetype": "scp",
    filepath, key = filepath.split(':', 1)
    return kaldi_io.read_mat_scp(filepath)


LOADERS = {
    'hdf5': load_hdf5,
    'sound.hdf5': load_soundhdf5,
    'sound': load_audiofile,
    'npz': load_compressed_numpy,
    'npy': load_numpy,
    'mat': load_mat,
    'vec': load_vec,
    'scp': load_scp
}


def get_loader(self, filetype, filepath):
    """Return loader function

    In order to make the fds to be opened only at the first referring,
    the loader are stored in self._loaders

    >>> ndarray = loader.get_from_loader(
    ...     'some/path.h5:F01_050C0101_PED_REAL', filetype='hdf5')

    :param: str filetype:
    :return:
    :rtype: function
    """
    if filetype in LOADERS:
        return LOADERS[filetype](filepath)
    raise NotImplementedError(
        'Not supported: loader_type={}'.format(filetype)
    )
