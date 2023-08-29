import math
import torch
import torch.nn as nn
import pyro.distributions as dis
import utils
import numpy as np

class vae(nn.Module):
	# variational autoencoder
	def __init__(self, args):
		super().__init__()
		self.encoder=encoder(args)
		self.decoder=decoder(args)
		self.args = args
		# set hyperparameter if using expressive priors. ae=AEA
		if args.prior == 'vamp':
			self.mixture_components = nn.Parameter(torch.randn(args.prior_hyper, args.visible_dimension))
			self.mixture_weight = nn.Parameter(torch.randn(args.prior_hyper))
		elif args.prior == 'maf':
			self.maf = nn.ModuleList([utils.MAF(dim=args.hidden_dimension) for i in range(args.prior_hyper)])
		elif args.prior == 'ae':
			self.register_buffer('ae_means',torch.tensor(np.load('means'+args.loss_function.lower()+str(args.prior_hyper)+'.npy')))
			self.register_buffer('ae_cov',torch.tensor(np.load('cov'+args.loss_function.lower()+str(args.prior_hyper)+'.npy')).log_())
			self.register_buffer('ae_weights',torch.tensor(np.load('weights'+args.loss_function.lower()+str(args.prior_hyper)+'.npy')))
			
	def forward(self, x):
		y=x.to(torch.float)
		mean, logvar = self.encoder(y)
		samples = reparameterization(mean, logvar, self.args.sample_size)
		pzero, strength, ratio = self.decoder(samples)
		# calculate the mean reconstuction loss over samples
		likelihood = evaluate(x, pzero, strength, ratio, self.args.loss_function).mean(dim = 0)
		# calculate the mean regularization loss over samples
		if self.args.prior == 'vamp':
			prior_means, prior_logvars = self.encoder(self.mixture_components)
			mixture_weight = torch.softmax(self.mixture_weight,dim=0)
			klloss = kld(mean, logvar, self.args, samples, prior_means, prior_logvars,mixture_weight)
		elif self.args.prior=='maf':
			change=torch.zeros(samples.shape[0],samples.shape[1],device=samples.device)
			for i in range(self.args.prior_hyper):
				samples, log_det = self.maf[self.args.prior_hyper-i-1].infer(samples)
				change+=log_det
			klloss =kld(mean, logvar, self.args, samples, change)
		elif self.args.prior == 'ae':
			klloss =kld(mean, logvar,self.args,samples, self.ae_means,self.ae_cov,self.ae_weights)
		else:
			klloss = kld(mean, logvar, self.args)

		return likelihood, klloss

	def generate(self, sample_size):
		# generating samples using samples from the prior
		with torch.no_grad():
			if self.args.prior == 'vamp':
				prior_means, prior_logvars = self.encoder(self.mixture_components)
				mixture_weight = torch.softmax(self.mixture_weight,dim=0)
				mixture = np.random.choice(np.arange(self.args.prior_hyper),sample_size,replace=True,p=mixture_weight.detach().cpu().numpy())
				latent=[]
				for i in range(len(mixture_weight)):
					num=(mixture==i).sum()
					latent.append(torch.randn(num,self.args.hidden_dimension)*torch.exp(0.5*prior_logvars[i])+prior_means[i])
				latent=torch.cat(latent,dim=0)
			elif self.args.prior == 'maf':
				latent = torch.randn(sample_size,self.args.hidden_dimension)
				for i in range(self.args.prior_hyper):
					latent, log_det = self.maf[i].sample(latent)
			elif self.args.prior == 'ae':
				mixture = np.random.choice(np.arange(self.args.prior_hyper),sample_size,replace=True,p=self.ae_weights.detach().cpu().numpy())
				latent=[]
				for i in range(len(self.ae_weights)):
					num=(mixture==i).sum()
					latent.append(torch.randn(num,self.args.hidden_dimension)*torch.exp(0.5*self.ae_cov[i])+self.ae_means[i])
				latent=torch.cat(latent,dim=0).float()
			else:
				latent=torch.randn(sample_size,self.args.hidden_dimension)

			latent=latent[torch.randperm(sample_size)]

			pzero, strength, ratio = self.decoder(latent)
			if self.args.loss_function == 'ZIP':
				prob = dis.ZeroInflatedPoisson(strength, gate = pzero)
			elif self.args.loss_function == 'ZINB':
				prob = dis.ZeroInflatedNegativeBinomial(strength, probs=ratio, gate = pzero)
			elif self.args.loss_function == 'P':
				prob = torch.distributions.Poisson(strength)
			elif self.args.loss_function == 'NB':
				prob = torch.distributions.NegativeBinomial(strength, probs=ratio)

		return prob.sample([1])



class encoder(nn.Module):
	# encoder for VAE/AE model
	def __init__(self, args):
		super().__init__()

		visible = args.visible_dimension
		hidden = args.hidden_dimension
		inter = args.inter_layer
		final = [visible] + inter + [hidden]

		layers = []
		# construct the encoder
		for i in range(len(final) - 2):
			layers.append(nn.Linear(final[i], final[i + 1]))
			layers.append(nn.ReLU())
		self.model = nn.Sequential(*layers)

		# the two output layers of encoder, representing the mean and the log diagonal variances
		self.mean = nn.Linear(final[-2], final[-1])
		self.logvar = nn.Linear(final[-2], final[-1])

	def forward(self, x):
		features = self.model(x)
		mean = self.mean(features)
		logvar = self.logvar(features)

		return mean, logvar

class decoder(nn.Module):
	# decoder for VAE/AE model with non-categorical conditional distribution
	def __init__(self, args):
		super().__init__()
		
		visible = args.visible_dimension
		hidden = args.hidden_dimension
		inter = args.inter_layer

		final = ([visible] + inter + [hidden])
		final.reverse()

		layers = []
		# construct the decoder
		for i in range(len(final) - 2):
			layers.append(nn.Linear(final[i], final[i + 1]))
			layers.append(nn.ReLU())
		# the three output layers, some are not used, depending on the conditional distribution
		self.model = nn.Sequential(*layers)
		self.pzero = nn.Linear(final[-2], final[-1])
		self.strength = nn.Linear(final[-2], final[-1])   
		self.ratio = nn.Linear(final[-2], final[-1])
	def forward(self,x):
		features = self.model(x)
		pzero = torch.sigmoid(self.pzero(features))
		strength = torch.exp(self.strength(features))+1e-10
		ratio = torch.sigmoid(self.ratio(features))

		# avoid extreme values to keep a healthy gradient flow
		strength = torch.where(strength > 50, 50, strength)
		pzero = torch.where(pzero > 0.99, 0.99, pzero)
		ratio = torch.where(ratio > 0.99, 0.99, ratio)
		pzero = torch.where(pzero < 0.01, 0.01, pzero)
		ratio = torch.where(ratio < 0.01, 0.01, ratio)
		
		return pzero, strength, ratio


def reparameterization(mean, logvar, sample_size = 1):
	# reparameterization trick to keep a continuous gradient flow
	std = torch.exp(0.5*logvar)
	eps=torch.randn(torch.Size([sample_size]) + std.shape, dtype=std.dtype, device=std.device)
	z = mean + eps*std
	return z.to(device = std.device)

def kld(mean, logvar, args, *others):
	# calculating the regularization loss. Have a closed-form solution for gaussian prior. 
	# the divergence is the difference between the differential entropy of the variational posterior (closed form) and the cross entropy
	# between the prior and the variational posterior, the latter can use samples to approximate if no closed form solution
	if args.prior == 'gaussian':
		var = torch.exp(logvar)
		kld=-0.5*(1 + logvar - mean**2 - var).sum(dim = -1)
	elif args.prior == 'vamp':
		samples = others[0].unsqueeze(2)
		prior_means = others[1]
		prior_logvars = others[2]
		mixture_weight=others[3]
		selfentropy = 0.5*(logvar+1).sum(dim=-1)
		single= 0.5*(prior_logvars+(samples-prior_means)**2/torch.exp(prior_logvars)
			).sum(dim=-1)+torch.log(mixture_weight)
		crossentropy = torch.logsumexp(single, dim=-1).mean(dim=0)
		kld = crossentropy - selfentropy
	elif args.prior == 'maf':
		samples = others[0]
		change = others[1]
		# avoid extreme values to keep a healthy gradient flow
		samples = torch.where(samples>1e2,1e2,samples)
		samples = torch.where(samples<-1e2,-1e2,samples)
		selfentropy = 0.5*(logvar+1).sum(dim=-1)
		crossentropy = 0.5*(samples**2).sum(dim=-1)
		crossentropy-=change
		kld=crossentropy.mean(dim=0)-selfentropy
	elif args.prior == 'ae':
		samples = others[0].unsqueeze(2)

		ae_means = others[1]
		ae_cov = others[2]
		ae_weights = others[3]
		selfentropy=0.5*(logvar+1).sum(dim=-1)
		single=0.5*(ae_cov+(samples-ae_means)**2/torch.exp(ae_cov)).sum(dim=-1)+torch.log(ae_weights)
		crossentropy = torch.logsumexp(single,dim=-1).mean(dim=0)
		kld = crossentropy - selfentropy

	return kld



def evaluate(x, pzero, strength, ratio, loss):
	# calculate the reconstruction loss

	if loss == 'ZIP':
		prob = dis.ZeroInflatedPoisson(strength, gate = pzero)
		
	elif loss == 'ZINB':
		prob = dis.ZeroInflatedNegativeBinomial(strength, probs=ratio, gate = pzero)

	elif loss == 'P':
		prob = torch.distributions.Poisson(strength)

	elif loss == 'NB':
		prob = torch.distributions.NegativeBinomial(strength, probs=ratio)

	return prob.log_prob(x).sum(dim=-1)


class ae(nn.Module):
	# autoencoder with non-categorical conditional distributions
	def __init__(self, args):
		super().__init__()
		self.encoder=encoder(args)
		self.decoder=decoder(args)
		self.args = args

	def forward(self, x):
		y=x.to(torch.float)
		mean, logvar = self.encoder(y)
		# only the mean vector is used since no stochastic layers
		pzero, strength, ratio = self.decoder(mean)
		likelihood = evaluate(x, pzero, strength, ratio, self.args.loss_function)

		return likelihood

class cate_ae(nn.Module):
	# autoencoder with categorical conditional distribution
	def __init__(self, args,maxcounts):
		super().__init__()
		self.encoder=encoder(args)
		self.decoder=cate_decoder(args,maxcounts)
		self.args = args
		self.loss = nn.CrossEntropyLoss(reduction='none')

	def forward(self,x):
		y=x.to(torch.float)
		mean, logvar = self.encoder(y)
		cates=self.decoder(mean)
		loss_all=[]
		# label smoothing during training
		self.loss.label_smoothing = 1e-5 if self.training else 0
		# calculate losses per object class
		losses = [self.loss(cates[i], x.long()[:,i]).unsqueeze(1) for i in range(self.args.visible_dimension)]
		likelihood = -torch.cat(losses, dim=1).sum(dim=1)

		return likelihood



class cate_vae(nn.Module):
	# variational autoencoder with categorical conditional distribution
	def __init__(self, args, maxcounts):
		super().__init__()
		self.encoder=encoder(args)
		self.decoder=cate_decoder(args, maxcounts)
		self.args = args
		self.loss = nn.CrossEntropyLoss(reduction='none')
		if args.prior == 'vamp':
			self.mixture_components = nn.Parameter(torch.randn(args.prior_hyper, args.visible_dimension))
			self.mixture_weight = nn.Parameter(torch.randn(args.prior_hyper))
		elif args.prior == 'maf':
			self.maf = nn.ModuleList([utils.MAF(dim=args.hidden_dimension) for i in range(args.prior_hyper)])
		elif args.prior == 'ae':
			self.register_buffer('ae_means',torch.tensor(np.load('means'+args.loss_function.lower()+str(args.prior_hyper)+'.npy')))
			self.register_buffer('ae_cov',torch.tensor(np.load('cov'+args.loss_function.lower()+str(args.prior_hyper)+'.npy')).log_())
			self.register_buffer('ae_weights',torch.tensor(np.load('weights'+args.loss_function.lower()+str(args.prior_hyper)+'.npy')))

	def forward(self,x):
		y=x.to(torch.float)
		mean, logvar = self.encoder(y)
		samples = reparameterization(mean, logvar, self.args.sample_size)
		cates = self.decoder(samples)
		
		loss_all = []
		for j in range(self.args.sample_size):
			self.loss.label_smoothing = 1e-5 if self.training else 0
			losses = [self.loss(cates[i][j], x.long()[:,i]).unsqueeze(1) for i in range(self.args.visible_dimension)]
			loss_all.append(torch.cat(losses, dim=1).sum(dim=1).unsqueeze(0))

		likelihood = -torch.cat(loss_all,dim=0).mean(dim=0)
		if self.args.prior == 'vamp':
			prior_means, prior_logvars = self.encoder(self.mixture_components)
			mixture_weight = torch.softmax(self.mixture_weight,dim=0)
			klloss = kld(mean, logvar, self.args, samples, prior_means, prior_logvars,mixture_weight)
		elif self.args.prior == 'maf':
			change=torch.zeros(samples.shape[0],samples.shape[1],device=samples.device)
			for i in range(self.args.prior_hyper):
				samples, log_det = self.maf[self.args.prior_hyper-i-1].infer(samples)
				change+=log_det
			klloss =kld(mean, logvar, self.args, samples, change)
		elif self.args.prior == 'ae':
			klloss =kld(mean, logvar,self.args,samples, self.ae_means,self.ae_cov,self.ae_weights)
		else:
			klloss = kld(mean, logvar, self.args)


		return likelihood, klloss

	def generate(self, sample_size):
		with torch.no_grad():
			if self.args.prior == 'vamp':
				prior_means, prior_logvars = self.encoder(self.mixture_components)
				mixture_weight = torch.softmax(self.mixture_weight,dim=0)
				mixture = np.random.choice(np.arange(self.args.prior_hyper),sample_size,replace=True,p=mixture_weight.detach().cpu().numpy())
				latent=[]
				for i in range(len(mixture_weight)):
					num=(mixture==i).sum()
					latent.append(torch.randn(num,self.args.hidden_dimension)*torch.exp(0.5*prior_logvars[i])+prior_means[i])
				latent=torch.cat(latent,dim=0)
			elif self.args.prior == 'maf':
				latent = torch.randn(sample_size,self.args.hidden_dimension)
				for i in range(self.args.prior_hyper):
					latent, log_det = self.maf[i].sample(latent)
			elif self.args.prior == 'ae':
				mixture = np.random.choice(np.arange(self.args.prior_hyper),sample_size,replace=True,p=self.ae_weights.detach().cpu().numpy())
				latent=[]
				for i in range(len(self.ae_weights)):
					num=(mixture==i).sum()
					latent.append(torch.randn(num,self.args.hidden_dimension)*torch.exp(0.5*self.ae_cov[i])+self.ae_means[i])
				latent=torch.cat(latent,dim=0).float()
			else:
				latent=torch.randn(sample_size,self.args.hidden_dimension)

			latent=latent[torch.randperm(sample_size)]

			cates=self.decoder(latent)

			samples=[]
			for i in range(self.args.visible_dimension):
				prob=torch.distributions.categorical.Categorical(logits=cates[i])
				samples.append(prob.sample([1]).T)

			return torch.cat(samples,dim=-1)


class cate_decoder(nn.Module):
	# decoder for AE/VAE with categorical conditional distribution
	def __init__(self, args, maxcounts):
		super().__init__()
		
		hidden = args.hidden_dimension
		inter = args.inter_layer

		final = (inter + [hidden])
		final.reverse()

		layers = []
		for i in range(len(final) - 1):
			layers.append(nn.Linear(final[i], final[i + 1]))
			layers.append(nn.ReLU())
		self.model = nn.Sequential(*layers)
		self.cate = nn.ModuleList([nn.Linear(final[-1], num_cate + 1) for num_cate in maxcounts])

	def forward(self,x):
		features = self.model(x)
		# different output layers
		cates = [cate_layer(features) for cate_layer in self.cate]

		return cates

class base_NB(nn.Module):
	# baseline-NB/ZINB model (using GD to optimize)
	def __init__(self, args):
		super().__init__()
		self.pzero = nn.Parameter(torch.randn(args.visible_dimension))
		self.strength = nn.Parameter(torch.randn(args.visible_dimension))
		self.ratio = nn.Parameter(torch.randn(args.visible_dimension))
		self.args = args

	def forward(self, x):
		pzero = torch.sigmoid(self.pzero)
		strength = torch.exp(self.strength)+1e-10
		ratio = torch.sigmoid(self.ratio)
		likelihood = evaluate(x, pzero, strength,ratio,self.args.loss_function)

		return likelihood

	def generate(self, sample_size):
		pzero = torch.sigmoid(self.pzero)
		strength = torch.exp(self.strength)+1e-10
		ratio = torch.sigmoid(self.ratio)
		if self.args.loss_function=='NB':
			prob=torch.distributions.NegativeBinomial(strength, probs=ratio)
		else:
			prob = dis.ZeroInflatedNegativeBinomial(strength, probs=ratio, gate = pzero)

		return prob.sample([sample_size])


class mixture_model(nn.Module):
	# Mixture model with non-categorical conditional distributions
	def __init__(self, args):
		super().__init__()
		self.mixture_weight = nn.Parameter(torch.randn(args.mixture_number))
		self.pzero = nn.ParameterList([nn.Parameter(torch.randn(args.visible_dimension)) for i in range(args.mixture_number)])
		self.strength = nn.ParameterList([nn.Parameter(torch.randn(args.visible_dimension)) for i in range(args.mixture_number)])
		self.ratio = nn.ParameterList([nn.Parameter(torch.randn(args.visible_dimension)) for i in range(args.mixture_number)])
		self.args = args

	def forward(self, x):
		mixture_weight = torch.softmax(self.mixture_weight,dim=0)
		likelihood = torch.randn(len(mixture_weight),x.shape[0],device=mixture_weight.device)

		for i in range(len(mixture_weight)):
			pzero = torch.sigmoid(self.pzero[i])
			strength = torch.exp(self.pzero[i])+1e-10
			ratio = torch.sigmoid(self.ratio[i])
			likelihood[i,:] = evaluate(x, pzero, strength,ratio,self.args.loss_function) + torch.log(mixture_weight[i])

		likelihood = torch.logsumexp(likelihood,dim=0)

		return likelihood

	def generate(self,sample_size):

		mixture_weight = torch.softmax(self.mixture_weight,dim=0)
		mixture = np.random.choice(np.arange(self.args.mixture_number),sample_size,replace=True,p=mixture_weight.detach().cpu().numpy())
		samples=[]
		for i in range(len(mixture_weight)):
			num=(mixture==i).sum()
			pzero = torch.sigmoid(self.pzero[i])
			strength = torch.exp(self.pzero[i])+1e-10
			ratio = torch.sigmoid(self.ratio[i])
			if self.args.loss_function == 'ZIP':
				prob = dis.ZeroInflatedPoisson(strength, gate = pzero)
			elif self.args.loss_function == 'ZINB':
				prob = dis.ZeroInflatedNegativeBinomial(strength, probs=ratio, gate = pzero)
			elif self.args.loss_function == 'P':
				prob = torch.distributions.Poisson(strength)
			elif self.args.loss_function == 'NB':
				prob = torch.distributions.NegativeBinomial(strength, probs=ratio)
			sample=prob.sample([num])
			samples.append(sample)

		return (torch.cat(samples,dim=0))[torch.randperm(sample_size)]


class cate_mixture_model(nn.Module):
	# Mixture model with categorical conditional distribution
	def __init__(self, args,maxcounts):
		super().__init__()
		self.mixture_weight=nn.Parameter(torch.randn(args.mixture_number))
		self.loss = nn.CrossEntropyLoss(reduction='none')
		self.cate = nn.ParameterList()
		# different output layers
		for i in range(args.mixture_number):
			self.cate.append(nn.ParameterList([nn.Parameter(torch.randn(num+1)) for num in maxcounts]))
		self.args = args

	def forward(self, x):
		mixture_weight = torch.softmax(self.mixture_weight,dim=0)
		likelihood = torch.randn(len(mixture_weight),x.shape[0],device=mixture_weight.device)
		# label smoothing during training
		self.loss.label_smoothing = 1e-5 if self.training else 0

		for i in range(len(mixture_weight)):
			single = self.cate[i]
			losses=[self.loss(single[j].repeat(x.shape[0],1), x[:,j].long()).unsqueeze(1) for j in range(self.args.visible_dimension)]
			loss_all = -torch.cat(losses,dim=1).sum(dim=1)
			likelihood[i:] = loss_all+torch.log(mixture_weight[i])

		likelihood=torch.logsumexp(likelihood,dim=0)

		return likelihood
	def generate(self,sample_size):

		mixture_weight = torch.softmax(self.mixture_weight,dim=0)
		mixture = np.random.choice(np.arange(self.args.mixture_number),sample_size,replace=True,p=mixture_weight.detach().cpu().numpy())
		samples=[]
		for i in range(len(mixture_weight)):
			num=(mixture==i).sum()
			if num>0:
				paras=self.cate[i]
				subsample=[]
				for j in range(self.args.visible_dimension):
					prob=torch.distributions.categorical.Categorical(logits=paras[j])
					subsample.append(prob.sample([num])[:,None])
				samples.append(torch.cat(subsample,dim=-1))

		return (torch.cat(samples,dim=0))[torch.randperm(sample_size)]

class transformer(nn.Module):
	# SSVAT model
	def __init__(self, args):
		super().__init__()
		# [BOS]/[EOS]/[PAD] three extra tokens
		num_tokens=args.visible_dimension+3
		self.embedding = nn.Embedding(num_tokens, args.embedding_dimension, padding_idx=num_tokens-1)
		self.encoder=trans_encoder(args)
		self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=args.embedding_dimension,nhead=args.num_heads
			,dropout=args.dropout,dim_feedforward = 2*args.embedding_dimension), num_layers=args.num_decoders)
		self.out=nn.Linear(args.embedding_dimension,num_tokens)
		self.num_tokens = num_tokens
		self.args=args

		if args.prior == 'maf':
			self.maf = nn.ModuleList([utils.MAF(dim=args.embedding_dimension) for i in range(args.prior_hyper)])
		elif args.prior == 'ae':
			self.register_buffer('ae_means',torch.tensor(np.load('means_transformer'+str(args.prior_hyper)+'.npy')))
			self.register_buffer('ae_cov',torch.tensor(np.load('cov_transformer'+str(args.prior_hyper)+'.npy')).log_())
			self.register_buffer('ae_weights',torch.tensor(np.load('weights_transformer'+str(args.prior_hyper)+'.npy')))
		

	def forward(self, src, tgt):
		# transform tokens to embeddings
		src_pad_mask = src.eq(self.num_tokens-1)
		tgt_pad_mask = tgt.eq(self.num_tokens-1)
		src=self.embedding(src)*math.sqrt(self.args.embedding_dimension)
		tgt=self.embedding(tgt)*math.sqrt(self.args.embedding_dimension)
		src=src.permute(1,0,2)
		tgt=tgt.permute(1,0,2)
		tgt_mask = self.tgt_mask(tgt.size(0)).to(src.device)

		pooling_out = self.encoder(src, src_pad_mask)[0]
		pooling_out_mean=pooling_out[0].unsqueeze(0)
		pooling_out_logvar=pooling_out[1].unsqueeze(0)
		
		pooling_out_std = torch.exp(0.5*pooling_out_logvar)
		eps=torch.randn(pooling_out_std.shape, dtype=pooling_out_std.dtype, device=pooling_out_std.device)
		pooling_out_sample = pooling_out_mean + eps*pooling_out_std

		if self.args.prior == 'maf':
			change=torch.zeros(pooling_out_sample.shape[0],pooling_out_sample.shape[1],device=pooling_out_sample.device)
			for i in range(self.args.prior_hyper):
				pooling_out_sample, log_det = self.maf[self.args.prior_hyper-i-1].infer(pooling_out_sample)
				change+=log_det
			klloss =kld(pooling_out_mean, pooling_out_logvar, self.args, pooling_out_sample, change).squeeze(0)
		elif self.args.prior == 'ae':
			klloss =kld(pooling_out_mean, pooling_out_logvar,self.args,pooling_out_sample, self.ae_means,self.ae_cov,self.ae_weights).squeeze(0)
		else:
			klloss =kld(pooling_out_mean,pooling_out_logvar,self.args).squeeze(0)

		# directly use the mean vector if training the AEA prior
		if self.args.model == 'ae':
			pooling_out_sample=pooling_out_mean

		decoder_out = self.decoder(tgt,pooling_out_sample, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_pad_mask)
		
		out = self.out(decoder_out)
		# forbid predicting [EOS] and [PAD] token
		out[:,:,[self.num_tokens-1,self.num_tokens-3]]=-float('inf')

		return out, klloss

	def tgt_mask(self, size):
		# masks in decoder input for parallel training
		mask = torch.tril(torch.ones(size, size) == 1).float()
		mask = mask.masked_fill(mask == 0, float('-inf')) 
		mask = mask.masked_fill(mask == 1, float(0.0))

		return mask

	def generate(self, sample_size):
		with torch.no_grad():
			if self.args.prior == 'maf':
				samples=torch.randn(sample_size,self.args.embedding_dimension)
				for i in range(self.args.prior_hyper):
					samples, log_det = self.maf[i].sample(samples)
				samples=samples.unsqueeze(0)
			elif self.args.prior=='ae':
				mixture = np.random.choice(np.arange(self.args.prior_hyper),sample_size,replace=True,p=self.ae_weights.detach().cpu().numpy())
				latent=[]
				for i in range(len(self.ae_weights)):
					num=(mixture==i).sum()
					latent.append(torch.randn(num,self.args.embedding_dimension)*torch.exp(0.5*self.ae_cov[i])+self.ae_means[i])
				latent=torch.cat(latent,dim=0).float()
				samples=(latent[torch.randperm(sample_size)]).unsqueeze(0)
			else:
				samples=torch.randn(sample_size,self.args.embedding_dimension).unsqueeze(0)
			tgt=(torch.ones(sample_size)*(self.num_tokens-3))[:,None]
			# to reduce computational cost, the maximum length is 100 for generated sequences
			for i in range(100):
				tgt_eb=(self.embedding(tgt.long())*math.sqrt(self.args.embedding_dimension)).permute(1,0,2)
				decoder_out=self.decoder(tgt_eb,samples)
				out=self.out(decoder_out)
				out[:,:,[self.num_tokens-1,self.num_tokens-3]]=-float('inf')
				prob=torch.distributions.categorical.Categorical(logits=out[-1,:,:])
				tgt=torch.cat([tgt,prob.sample([1]).T],dim=-1)
			pad_eos=(torch.ones(sample_size)*(self.num_tokens-2))[:,None]
			tgt=torch.cat([tgt[:,1:],pad_eos],dim=-1)
			final=[]
			for i in range(sample_size):
				# get sequences before the first [EOS] token
				eos=torch.where(tgt[i]==(self.num_tokens-2))[0][0]
				pre=tgt[i][:eos].long()
				one_hots=nn.functional.one_hot(pre,self.args.visible_dimension).sum(dim=0)
				final.append(one_hots[None,:])

		return torch.cat(final,dim=0)

class trans_encoder(nn.Module):
	# encoder for the SSVAT
	def __init__(self, args):
		super().__init__()
		num_tokens=args.visible_dimension+3
		self.embedding = nn.Embedding(num_tokens, args.embedding_dimension, padding_idx=num_tokens-1)
		self.encoder_layer=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.embedding_dimension,nhead=args.num_heads
			,dropout=args.dropout,dim_feedforward = 2*args.embedding_dimension), num_layers=args.num_encoders)
		# trainable query matrix and the multi-head attention layer for pooling
		self.pooling=nn.Parameter(torch.randn(2,args.embedding_dimension).reshape(2,1,args.embedding_dimension))
		self.pooling_layer = nn.MultiheadAttention(args.embedding_dimension,num_heads=args.num_heads,dropout=args.dropout)
		self.batch_size=args.batch_size

	def forward(self, src, src_pad_mask):
		trans_encoder_out = self.encoder_layer(src, src_key_padding_mask = src_pad_mask)
		query=self.pooling.repeat(1,self.batch_size,1)
		pooling_out = self.pooling_layer(query,trans_encoder_out,trans_encoder_out,key_padding_mask=src_pad_mask,need_weights=False)

		return pooling_out


class dis_flow_model(nn.Module):
	# DAF model
	def __init__(self, args, maxcounts):
		super().__init__()
		# reverse variable order after adding each MADE layer to increase flexibility
		self.flows = nn.ModuleList([dis_flow(args.net_width,args.net_depth,args.temperature,(i+1)%2==0,maxcounts) for i in range(args.flow_number)])
		self.probs = nn.ParameterList([torch.randn(i+1) for i in maxcounts])
		self.unit=len(maxcounts)
		self.maxcounts=maxcounts
		self.temperature=args.temperature
		self.args=args

	def forward(self,x):
		# for the discrete flow, it does not matter whether the factorized categorical distribution is the base or the transformed
		# distribution since there is no volume change
		y = [nn.functional.one_hot(x[:,i].long(), (self.maxcounts[i]+1).long()) for i in range(self.unit)]
		# flatten the inputs and pass it to several MADE layer
		y = torch.cat(y,dim=-1).float()
		for i in range(self.args.flow_number):
			y = self.flows[i](y)
		splits=list(torch.split(y,(self.maxcounts+1).tolist(),dim=-1))
		splits = [utils.one_hot_argmax(splits[i],self.temperature) for i in range(self.unit)]
		# label smoothing during training
		if self.training:
			splits = [(1-1e-5)*splits[i]+1e-5/(self.maxcounts[i]+1) for i in range(self.unit)]
		likelihood = [torch.mm(splits[i].float(),nn.functional.log_softmax(self.probs[i],dim=-1)[:,None]) for i in range(self.unit)]
		likelihood = torch.cat(likelihood,dim=-1).sum(dim=-1)
		return likelihood

	def generate(self, sample_size):
		with torch.no_grad():
			samples=[]
			for i in range(self.unit):
				prob=torch.distributions.one_hot_categorical.OneHotCategorical(logits=self.probs[i])
				samples.append(prob.sample([sample_size]).float())
			for i in range(self.args.flow_number):
				samples=self.flows[self.args.flow_number-i-1].sample(samples)
			samples=[torch.argmax(i,dim=-1)[:,None] for i in samples]

		return torch.cat(samples,dim=-1)



class dis_flow(nn.Module):
	# a single MADE layer
	def __init__(self, net_width, net_depth, temperature, reverse, maxcounts):
		super().__init__()
		unit = len(maxcounts)
		hidden = [net_width*unit for i in range(net_depth)]
		self.made = utils.dis_MADE(hidden, maxcounts,reverse)
		self.temperature=temperature
		self.unit=unit
		self.maxcounts=maxcounts
		self.reverse=reverse

	def forward(self, x):
		one_hots = self.made(x)
		# predicted location change
		locs=[utils.one_hot_argmax(i, self.temperature).float() for i in one_hots]
		x=list(torch.split(x,(self.maxcounts+1).tolist(),dim=-1))
		# addition+mod in the one-hot embedding space
		outputs=[utils.one_hot_add(x[i], locs[i]) for i in range(self.unit)]
		outputs=torch.cat(outputs, dim=-1)

		return outputs

	def sample(self, z):
		# sample from the categorical and change it to sample from the model
		order = range(self.unit-1,-1,-1) if self.reverse else range(self.unit)
		for j in order:
			z_trans=torch.cat(z,dim=-1)
			one_hots=self.made(z_trans)
			locs=[utils.one_hot_argmax(q, self.temperature).float() for q in one_hots]
			z[j]=utils.one_hot_minus(z[j],locs[j])

		return z



		

class ard(nn.Module):
	# DAD model
	def __init__(self, args, maxcounts):
		super().__init__()
		self.unit=len(maxcounts)
		hidden=[args.net_width*self.unit for i in range(args.net_depth)]
		# the input layer is wider due to the mask state
		nin = sum(maxcounts+2)
		hs = [nin] + hidden
		self.net=[]
		for h0,h1 in zip(hs, hs[1:]):
			self.net.extend([nn.Linear(h0,h1), nn.ReLU()])
		self.net = nn.Sequential(*self.net)
		self.project=nn.ModuleList()
		for i in (maxcounts+1):
			self.project.append(nn.Linear(hs[-1],i))
		self.args=args
		self.maxcounts=maxcounts
		self.loss = nn.CrossEntropyLoss(reduction='none')

	def forward(self,x):
		# get random order and position (equivalent to random masks)
		masks=utils.get_mask(x).to(x.device)
		y=(x+1)*masks
		one_hots= [nn.functional.one_hot(y[:,i].long(),(self.maxcounts[i]+2).long()) for i in range(self.unit)]
		one_hots=torch.cat(one_hots,dim=-1)
		features=self.net(one_hots.float())
		res=[i(features) for i in self.project]
		# label smoothing during training
		self.loss.label_smoothing = 1e-5 if self.training else 0
		losses=[self.loss(res[i],x[:,i].long()).unsqueeze(1) for i in range(self.unit)]
		losses=(torch.cat(losses,dim=-1)*(1-masks)).sum(dim=-1)
		# reweight the losses
		weights=self.unit/(1-masks).sum(dim=-1)
		weighted_losses = losses*weights 

		return weighted_losses

	def generate(self, sample_size):
		x=torch.zeros(sample_size,self.unit).to(torch.long)
		orders=utils.get_orders(x)
		# generate samples autoregressively
		with torch.no_grad():
			for i in range(self.unit):
				tomask=orders.clone()[:,i:]
				tomask=tomask.T.tolist()
				y=x+1
				y[list(range(sample_size)),tomask]=0
				one_hots=[nn.functional.one_hot(y[:,j].long(),(self.maxcounts[j]+2).long()) for j in range(self.unit)]
				one_hots=torch.cat(one_hots,dim=-1)
				features=self.net(one_hots.float())
				res=[j(features) for j in self.project]
				samples=[]
				for j in range(self.unit):
					prob=torch.distributions.categorical.Categorical(logits=res[j])
					samples.append(prob.sample([1]).T)
				res=torch.cat(samples,dim=-1).to(torch.long)
				index=(orders[:,i]).tolist()
				x[list(range(sample_size)),index]=res[list(range(sample_size)),index]

		return x






		







