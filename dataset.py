import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        # super(DatasetFromHdf5, self).__init__()
        self.file_path = file_path

    def open_hdf5(self):
        hf = h5py.File(self.file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')
    
    def len_hdf5(self):
        hf = h5py.File(self.file_path)
        self.total = hf.get('data').shape[0]

    def __getitem__(self, index):
        if not hasattr(self, 'data'):
            self.open_hdf5()
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        if not hasattr(self, 'total'):
            self.len_hdf5()
        return self.total