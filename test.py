import torch
import subjectlist as subl
import os
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

# hyper parameters
epoch_num = 1000
batch_size = 1
lmk_num = 133
learning_rate = 0.0001

# 5000 mas
# train_root_dir = '/share4/huoy1/Deep_5000_Brain/working_dir/test_out'
# out = '/share4/huoy1/Deep_5000_Brain/working_dir/MAS5000/testing/'

#45 truth
# train_root_dir = '/share3/huoy1/3DUnet/working_dir/test_out_lr=0.0001'
# out = '/share4/huoy1/Deep_5000_Brain/working_dir/True45/testing'
# test_img_dir = '/share4/huoy1/Deep_5000_Brain/testing/resampled'

#1_1_1
train_root_dir = '/share4/huoy1/Deep_5000_Brain/working_dir/1_1_1/test_out'
out = '/share4/huoy1/Deep_5000_Brain/working_dir/1_1_1/testing/'
test_img_dir = '/share4/huoy1/Deep_5000_Brain/testing/part_1_1_1/croped'






mkdir(out)

# make img list

test_img_subs,test_img_files = subl.get_sub_list(test_img_dir)
test_dict = {}
test_dict['img_subs'] = test_img_subs
test_dict['img_files'] = test_img_files


# load image
test_set = torchsrc.imgloaders.pytorch_loader(test_dict,num_labels=lmk_num)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=1)

# load network
model = torchsrc.models.UNet3D(in_channel=1, n_classes=lmk_num)
# model = torchsrc.models.VNet()

# print_network(model)
#
# load optimizor
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
	test_loader=test_loader,
    train_root_dir = train_root_dir,
	out=out,
	max_epoch = epoch_num,
	batch_size = batch_size,
	lmk_num = lmk_num,
)


print("==start testing==")

start_epoch = 0
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.test_epoch()







