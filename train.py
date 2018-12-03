import torch
import subjectlist as subl
import os
import argparse
import torchsrc

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', required=True, help='location of training data')
parser.add_argument('--test_data_dir', required=True, help='loacation of testing data')
parser.add_argument('--working_dir', required=True, help='loacation of working directory')
parser.add_argument('--piece', default='1_1_1', help='1_1_1 | 1_1_3 | 3_3_3 etc.')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for, default=100')
parser.add_argument('--batch_size', type=int, default=1, help='batch size, default=1')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')


opt = parser.parse_args()
print(opt)

# hyper parameters
epoch_num = opt.epoch
train_data_dir = opt.train_data_dir
test_data_dir = opt.test_data_dir
working_root_dir =  opt.working_dir
piece = opt.piece
batch_size = opt.batch_size
learning_rate = opt.lr  #0.0001


num_channels = 15

piece_map = {}
piece_map['1_1_1'] = [0, 	96, 		0,	96, 		0,	60]

train_source_dir = os.path.join(train_data_dir,'source')
train_target_dir = os.path.join(train_data_dir,'target')
test_source_dir = os.path.join(test_data_dir,'source')
working_dir = os.path.join(working_root_dir,piece)

# define paths
out = os.path.join(working_dir, 'finetune_out')
mkdir(out)
train_source_subs,train_source_files = subl.get_sub_list(train_source_dir)
train_target_subs,train_target_files = subl.get_sub_list(train_target_dir)
train_dict = {}
train_dict['source_subs'] = train_source_subs
train_dict['source_files'] = train_source_files
train_dict['target_subs'] = train_target_subs
train_dict['target_files'] = train_target_files


test_source_subs,test_source_files = subl.get_sub_list(test_source_dir)
test_dict = {}
test_dict['source_subs'] = test_source_subs
test_dict['source_files'] = test_source_files



# load image
train_set = torchsrc.imgloaders.pytorch_loader_allpiece(train_dict,num_channels=num_channels,piece=piece,piece_map=piece_map)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1)
test_set = torchsrc.imgloaders.pytorch_loader_allpiece(test_dict,num_channels=num_channels,piece=piece,piece_map=piece_map)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=1)

# load network
model = torchsrc.models.UNet3D(in_channel=num_channels, n_classes=num_channels)
# model = torchsrc.models.VNet()

# print_network(model)
#
# load optimizor
optim = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
# optim = torch.optim.SGD(model.parameters(), lr=learning_curve() _rate, momentum=0.9)

# load CUDA
cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()

# load trainer
trainer = torchsrc.Trainer(
	cuda=cuda,
	model=model,
	optimizer=optim,
	train_loader=train_loader,
	test_loader=test_loader,
	out=out,
	max_epoch = epoch_num,
	batch_size = batch_size,
	lmk_num = num_channels
)


print("==start training==")

start_epoch = 0
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.train_epoch()







