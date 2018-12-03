import os
import numpy as np
from torch.utils import data
import nibabel as nib

nRows = 96
nCols = 96
nSlices = 60

output_x = 96
output_y = 96
output_z = 64

# labels = [0, 45]

# labels = [0, 4,11,23,30,31,32,35,36,37,38,39,40,41,44,45,47,48,49,50,51,52,55,56,57,58,59,60,61,62,71,72,73,75,76,100,101,102,103,104,105,106,107,108,109,112,113,114,115,116,117,118,119,120,121,122,123,124,125,128,129,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207]
class pytorch_loader_allpiece(data.Dataset):
    def __init__(self, subdict, num_channels, piece,piece_map):
        self.subdict = subdict
        self.source_subs = subdict['source_subs']
        self.source_files = subdict['source_files']
        self.piece = piece
        self.piece_map = piece_map
        if subdict.has_key('target_subs'):
            self.target_subs_subs = subdict['target_subs']
            self.target_files = subdict['target_files']
        else:
            self.target_subs = None
            self.target_files = None
        self.num_channels = num_channels

    def __getitem__(self, index):
        num_channels = self.num_channels
        sub_name = self.source_subs[index]
        x = np.zeros((num_channels, output_z, output_x, output_y))
        source_file = self.source_files[index]
        source_3d = nib.load(source_file)
        source = source_3d.get_data()
        inds = self.piece_map[self.piece]
        source = source[inds[0]:inds[1],inds[2]:inds[3],inds[4]:inds[5]]
        # source = (source - source.min())/(source.max()-source.min())
        # source = source*255.0
        source = np.transpose(source,(3, 2, 0, 1))
        x[:,0:60,0:output_x,0:output_y] = source[:,:,:,:]
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        x = x.astype('float32')

        if (self.target_files == None):
            y = x
        else:
            y = np.zeros((num_channels, output_z, output_x, output_y))
            target_file = self.target_files[index]
            target_3d = nib.load(target_file)
            target = target_3d.get_data()
            target = target[inds[0]:inds[1],inds[2]:inds[3],inds[4]:inds[5]]
            target = np.transpose(target,(3, 2, 0, 1))
            y[:,0:60, 0:output_x, 0:output_y] = target[:,:,:,:]
            y[np.isnan(y)] = 0
            y[np.isinf(y)] = 0
            y = y.astype('float32')

        return x, y, sub_name

    def __len__(self):
        return len(self.source_subs)