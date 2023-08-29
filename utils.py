import torch
import torch.nn as nn
import numpy as np
import random
import sys
import logging
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.serialization import default_restore_location
from sklearn.mixture import GaussianMixture as g
from torch.nn.utils.rnn import pad_sequence
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression

def setup_seed(seed):
	# Set random seeds to ensure reproducibility
	 torch.manual_seed(seed)
	 random.seed(seed)
	 np.random.seed(seed)
	 torch.cuda.manual_seed(seed)
	 torch.cuda.manual_seed_all(seed)


def load_and_split(args):
	with open('training.csv') as f:
			ncols = len(f.readline().split(','))

	train = torch.from_numpy(np.loadtxt('training.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	valid = torch.from_numpy(np.loadtxt('valid.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	test = torch.from_numpy(np.loadtxt('test.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))

	# calculate the maximum counts of different obejct classes
	maxcounts = torch.cat((train.max(0).values.unsqueeze(0), valid.max(0).values.unsqueeze(0), 
		test.max(0).values.unsqueeze(0)),0).max(0).values
	
	train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True)

	# the GPU should be able to load the whole validation/test set without gradient if not using the SSVAT
	return train_loader, valid, test, maxcounts

def load_and_split_trans(args):
	with open('training.csv') as f:
			ncols = len(f.readline().split(','))

	train = torch.from_numpy(np.loadtxt('training.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	valid = torch.from_numpy(np.loadtxt('valid.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	test = torch.from_numpy(np.loadtxt('test.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True)
	val_loader = DataLoader(valid, batch_size = args.batch_size, shuffle = True)
	test_loader = DataLoader(test, batch_size = args.batch_size, shuffle = False)

	# the seuqnece data can be large due to padding, so the validation/test loss can not be calculated in one-shot.
	return train_loader, val_loader, test_loader



def save_checkpoint(save_dir, save_name, state_dict):
	# save the checkpoint file and update the best mean validation loss
	os.makedirs(save_dir, exist_ok=True)
	last_val = state_dict['val_hist'][-1]
	torch.save(state_dict, os.path.join(save_dir, save_name))

	if state_dict['best_val'] == last_val:
		torch.save(state_dict, os.path.join(save_dir, save_name.replace('last','best')))

def load_checkpoint(checkpoint_path, model, optimizer):
	# load the check point file when training the model from the latest checkpoint
	state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
	model.load_state_dict(state_dict['model'])
	optimizer.load_state_dict(state_dict['optimizer'])
	logging.info('Loaded checkpoint {}'.format(checkpoint_path))
	return state_dict

def init_logging(args):
	# logging information in the command line
	handlers = [logging.StreamHandler()]

	if hasattr(args, 'log_file') and args.log_file is not None:
		handlers.append(logging.FileHandler(args.log_file, mode='w'))

	logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
						level=logging.INFO)
	logging.info('COMMAND: %s' % ' '.join(sys.argv))
	logging.info('Arguments: {}'.format(vars(args)))


def init_para(m):
	# initialize the weights and biases
	for layer in m.modules():
		if isinstance(layer, nn.Linear):
			nn.init.xavier_uniform_(layer.weight)
			layer.bias.data.fill_(0)


class MAF(nn.Module):
	# MAF prior (revserse variable order after adding each MADE layer to increase flexibility)
	def __init__(self, dim, parity=True):
		super().__init__()
		self.dim = dim
		self.net = MADE(dim, [5*dim, 5*dim], 2*dim)
		self.parity = parity
	
	def infer(self, z):
		x = torch.zeros_like(z)
		log_det = torch.zeros(z.shape[0],z.shape[1],device=z.device)
		z = z.flip(dims=[2]) if self.parity else z
		for i in range(self.dim):
			st = self.net(x.clone()) 
			s, t = st.split(self.dim, dim=-1)
			x[:, :, i] = (z[:, :, i] - t[:, :, i]) * torch.exp(-s[:, :, i])
			log_det -= s[:,:, i]

		x=torch.where(torch.isnan(x), torch.randn_like(x),x)
		log_det=torch.where(torch.isnan(log_det), torch.randn_like(log_det),log_det)
		# avoid extreme values to keep a healthy gradient flow
		x=torch.where(x>1e2, 1e2,x)
		x=torch.where(x<-1e2, -1e2,x)
		log_det=torch.where(log_det>1e2, 1e2,log_det)
		log_det=torch.where(log_det<-1e2, -1e2,log_det)

		return x, log_det

	def sample(self, x):
		st = self.net(x)
		s, t = st.split(self.dim, dim=-1)
		z = x * torch.exp(s) + t
		z = z.flip(dims=[1]) if self.parity else z
		log_det = torch.sum(s, dim=1)
		return z, log_det


class MaskedLinear(nn.Linear):
	# masks to build the MADE layer
	def __init__(self, in_features, out_features, bias=True):
		super().__init__(in_features, out_features, bias)        
		self.register_buffer('mask', torch.ones(out_features, in_features))
		
	def set_mask(self, mask):
		self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
		
	def forward(self, input):
		return nn.functional.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
	# a single MADE layer for the MAF prior
	def __init__(self, nin, hidden_sizes, nout, natural_ordering=True):
		
		super().__init__()
		self.nin = nin
		self.nout = nout
		self.hidden_sizes = hidden_sizes
		
		self.net = []
		hs = [nin] + hidden_sizes + [nout]
		for h0,h1 in zip(hs, hs[1:]):
			self.net.extend([MaskedLinear(h0, h1), nn.ReLU()])
		self.net.pop() 
		self.net = nn.Sequential(*self.net)
		init_para(self.net)
		self.natural_ordering = natural_ordering
		
		self.m = {}
		self.update_masks()
		
	def update_masks(self):
		L = len(self.hidden_sizes)
		
		self.m[-1] = np.arange(self.nin) if self.natural_ordering else np.random.permutation(self.nin)
		for l in range(L):
			self.m[l] = np.random.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])
		
		masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
		masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
		
		if self.nout > self.nin:
			k = int(self.nout / self.nin)
			masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
		
		layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
		for l,m in zip(layers, masks):
			l.set_mask(m)
	
	def forward(self, x):
		return self.net(x)


def fit_mixture(args):
	# fit mixture model on extracted embeddings based on given conditional distribution 
	train_file=np.loadtxt('train_'+args.loss_function.lower()+'.csv',delimiter=',')
	model = g(n_components=args.prior_hyper, covariance_type='diag').fit(train_file)
	np.save('means'+args.loss_function.lower()+str(args.prior_hyper)+'.npy', model.means_)
	np.save('cov'+args.loss_function.lower()+str(args.prior_hyper)+'.npy', model.covariances_)
	np.save('weights'+args.loss_function.lower()+str(args.prior_hyper)+'.npy', model.weights_)

def generate_indexs(args, sample):
	# generate forward sequence, backward sequences and the numbers of possible squences for different count vectors
	# [BOS] = 80, [EOS]=81, [PAD]=82
	bos_index = args.visible_dimension
	eos_index = args.visible_dimension + 1
	pad_index = args.visible_dimension + 2
	nonzero_indices = torch.nonzero(sample)
	row_indices = nonzero_indices[:, 0]
	col_indices = nonzero_indices[:, 1]
	
	indexs = torch.repeat_interleave(col_indices, repeats=sample[row_indices, col_indices])
	indexs=list(indexs.split(sample.sum(-1).tolist()))
	indexs_forward=[index[torch.randperm(index.size(0))] for index in indexs]
	indexs_backward = [index.flip(dims=[0]) for index in indexs_forward]

	indexs_forward=[torch.cat((index, torch.tensor(eos_index,dtype=index.dtype,device=index.device).reshape(1)),dim=0) for index in indexs_forward]
	indexs_backward=[torch.cat((index, torch.tensor(eos_index,dtype=index.dtype,device=index.device).reshape(1)),dim=0) for index in indexs_backward]

	indexs_forward=pad_sequence(indexs_forward,padding_value=pad_index,batch_first=True)
	indexs_backward=pad_sequence(indexs_backward,padding_value=pad_index,batch_first=True)

	indexs_forward=torch.cat((torch.empty(args.batch_size, 1,dtype=indexs_forward.dtype,device=indexs_forward.device).fill_(bos_index),indexs_forward),dim=-1)
	indexs_backward=torch.cat((torch.empty(args.batch_size, 1,dtype=indexs_backward.dtype,device=indexs_backward.device).fill_(bos_index),indexs_backward),dim=-1)
	
	nonzero_indices = torch.nonzero(sample, as_tuple=True)
	single=torch.tensor(gammaln((sample[nonzero_indices]+1).cpu().numpy()))
	single=list(single.split(sample.ne(0).sum(-1).tolist()))
	overall=torch.tensor(gammaln((sample.sum(-1)+1).cpu().numpy()))
	permutation=torch.tensor([overall[i]-single[i].sum() for i in range(args.batch_size)])

	permutation=permutation.to(sample.device)

	return indexs_forward, indexs_backward, permutation


class dis_MADE(nn.Module):
	# MADE layer for DAF (revserse variable order after adding each MADE layer to increase flexibility)
	def __init__(self, hidden_sizes, maxcounts,reverse_order=False):
		super().__init__()
		self.nin = len(maxcounts)
		self.maxcounts = maxcounts
		self.hidden_sizes = hidden_sizes
		self.reverse_order = reverse_order
		self.net = []
		hs = [(maxcounts+1).sum()] + hidden_sizes
		for h0,h1 in zip(hs, hs[1:]):
			self.net.extend([MaskedLinear(h0, h1), nn.ReLU()])
		self.net = nn.Sequential(*self.net)
		init_para(self.net)
		self.project=nn.ModuleList()
		for i in maxcounts:
			project_i = MaskedLinear(hidden_sizes[-1],i+1)
			init_para(project_i)
			self.project.append(project_i)
		

		self.m={}
		self.update_masks()

	def update_masks(self):
		L=len(self.hidden_sizes)
		order=np.arange(start=self.nin-1,stop=-1,step=-1) if self.reverse_order else np.arange(self.nin)
		self.m[-1] = order.repeat(self.maxcounts+1)
		for i in range(L):
			self.m[i]=np.arange(self.hidden_sizes[i])%(self.nin-1)

		masks = [self.m[i-1][:,None] <= self.m[i][None,:] for i in range(L)]

		layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
		for l,m in zip(layers, masks):
			l.set_mask(m)

		for i in range(self.nin):
			mask = self.m[L-1][:,None] < ((self.nin-1-i) if self.reverse_order else i)
			self.project[i].set_mask(mask)

	def forward(self,x):
		features=self.net(x)
		one_hots=[self.project[i](features) for i in range(self.nin)]

		return one_hots

def one_hot(inputs, vocab_size):
	# change inputs to one-hot embeddings
	input_shape = inputs.shape
	inputs = inputs.flatten().unsqueeze(1).long()
	z = torch.zeros(len(inputs), vocab_size,device=inputs.device)
	z.scatter_(1, inputs, 1.)
	return z.view(*input_shape, vocab_size)

def one_hot_argmax(inputs, temperature):
	vocab_size = inputs.shape[-1]
	z = one_hot(torch.argmax(inputs, dim=-1), vocab_size) 
	soft = nn.functional.softmax(inputs / temperature, dim=-1)
	outputs = soft + (z - soft).detach()
	return outputs


def one_hot_add(inputs, shift):
	# differentiable addition+mod in one-hot embedding space
	fft_inputs = torch.fft.fft(inputs, dim=-1)
	fft_shift = torch.fft.fft(shift, dim=-1)
	complex_result = torch.fft.ifft(fft_inputs * fft_shift, dim=-1)
	return complex_result.real

def one_hot_minus(inputs, shift):
	# differentiable minus+mod in one-hot embedding space
	shift = shift.type( inputs.dtype)
	vocab_size = inputs.shape[-1]
	shift_matrix = torch.stack([torch.roll(shift, i, dims=-1) for i in range(vocab_size)], dim=-2)
	outputs = torch.einsum('...v,...uv->...u', inputs, shift_matrix)
	return outputs

def get_mask(x):
	# get random masks for training DAD
	batch = x.size(0)
	dimensions = x.size(1)
	pre = [torch.ones(dimensions) for i in range(batch)]
	positions=torch.randint(dimensions,(batch,))+1
	for i in range(batch):
		pre[i][:positions[i]]=0
	masks = [i[torch.randperm(dimensions)][None,:] for i in pre]
	masks = torch.cat(masks,dim=0)


	return masks

def get_orders(x):
	# get random orders for DAD sample generation
	sample_size=x.size(0)
	dimensions=x.size(1)
	orders=[torch.arange(dimensions)[torch.randperm(dimensions)][None,:] for i in range(sample_size)]
	orders=torch.cat(orders,dim=0)

	return(orders)

def neuro_dis(sample,train, valid, test,num_exp=5,seed=42,clip_norm=4.0,init_lr=0.005,max_epoch=50,patience=10,lr_patience=5,batch_size=100,ratio=0.1):
	# train a discriminator to distinguish whether a point is generated or observed, the same as the main experiment
	setup_seed(seed)
	seeds = np.random.choice(range(10000),size=num_exp,replace=False)
	train_size, visible =train.size()
	valid_size = valid.size(0)
	test_size = test.size(0)
	
	sample_split=torch.split(sample,[train_size,valid_size])

	labels_train=torch.cat([torch.zeros(train_size)[:,None],torch.ones(train_size)[:,None]],dim=0)
	labels_valid=torch.cat([torch.zeros(valid_size)[:,None],torch.ones(valid_size)[:,None]],dim=0)
	train_data=torch.cat([torch.cat([sample_split[0],train],dim=0),labels_train],dim=-1)
	valid_data=torch.cat([torch.cat([sample_split[1],valid],dim=0),labels_valid],dim=-1)

	train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
	state=[]
	best_loss=[]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	for i in seeds:
		setup_seed(i)
		model=discriminator(visible,[visible])
		init_para(model)
		model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), init_lr)
		bad_epochs = 0
		best_val = float('inf')

		for epoch in range(max_epoch):
			if bad_epochs == lr_patience:
				optimizer.param_groups[0]['lr']/=2
			for j in train_loader:
				model.train()
				optimizer.zero_grad()
				loss=model(j.to(device))
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
				optimizer.step()
			model.eval()
			with torch.no_grad():
				val_loss=model(valid_data.to(device))
			if val_loss < best_val:
				bad_epochs=0
				best_val=val_loss
				best_state=model.state_dict()
			else:
				bad_epochs+=1

			if bad_epochs>patience:
				break
		state.append(best_state)
		best_loss.append(best_val)

	index=int(torch.tensor(best_loss).argmin())
	model.load_state_dict(state[index])
	with torch.no_grad():
		logits=model.net(test.float().to(device))
		loss_fn=nn.CrossEntropyLoss(reduction='none')
		pre=loss_fn(logits,torch.ones(test_size,device=device).long())
	torch.save(state[index],'state.pt')
	# save losses on the test set
	np.savetxt('neuro_predictions.csv',pre.detach().cpu().numpy(),delimiter=',')

def logis_dis(sample,train, valid, test, max_iter=1000):
	# a logistic discriminator (not useful)
	train_size=train.size(0)+valid.size(0)
	labels_train=torch.cat([torch.zeros(train_size)[:,None],torch.ones(train_size)[:,None]],dim=0).numpy()
	train_data=torch.cat([sample,train,valid],dim=0).numpy()
	
	m=LogisticRegression(max_iter=max_iter)
	m.fit(train_data,labels_train)
	
	pre=-m.predict_log_proba(test)[:,1]
	np.savetxt('logis_predictions.csv',pre,delimiter=',')
	


class discriminator(nn.Module):
	# the neural network discriminator
	def __init__(self,visible,hidden):
		super().__init__()
		structure=[visible]+hidden+[2]
		net=[]
		for i in range(len(structure)-1):
			net.append(nn.Linear(structure[i],structure[i+1]))
			net.append(nn.ReLU())
		net.pop()
		self.net=nn.Sequential(*net)
		self.visible=visible
		self.loss=nn.CrossEntropyLoss()

	def forward(self,x):
		logits=self.net(x[:,:self.visible].float())
		loss=self.loss(logits, x[:,self.visible].long())

		return loss




	














	




