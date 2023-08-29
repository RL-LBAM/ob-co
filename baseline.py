import torch
import torch.nn as nn
import os
import logging
import argparse
import utils
import numpy as np
from scipy.special import lambertw
from collections import Counter
from model import base_NB
import pyro.distributions as dis

def get_args():
	""" hyper-parameters. """
	parser = argparse.ArgumentParser('Baseline')

	# Add data arguments
	parser.add_argument("--seed", type=int, default=42, help="primary random seed")
	parser.add_argument("--experiment_num", type=int, default=5, help="number of experiments to conduct")
	parser.add_argument("--sample_size", type=int, default=90000, help="sample size for generation")

	# Add model arguments
	parser.add_argument('--visible_dimension', default=80, type=int, help='number of object classes')	
	parser.add_argument('--loss_function', default='ZIP', type=str, help='distribution function')

	# Add training arguments
	parser.add_argument('--batch_size', default=100, type=int, help='batch size for training (not actually used)')
	parser.add_argument('--max_epoch', default=1000, type=int, help='force stop training at specified epoch')
	parser.add_argument('--clip_norm', default=10, type=float, help='clip threshold of gradients')
	parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
	parser.add_argument('--patience', default=10, type=int,
						help='number of epochs without improvement on validation set before early stopping')
	parser.add_argument('--patience_lr', default=5, type=int,
						help='number of epochs without improvement on validation set before halving the learning rate')

	# Add checkpoint arguments
	parser.add_argument('--log_file', default='logging.log', help='path to save logs')
	
	args = parser.parse_args()
	return args

def main(args):
	utils.init_logging(args)
	savedir = 'Baseline'+str(args.loss_function)
	os.makedirs(savedir, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info('using '+str(device))
	# even for NB and ZINB, I use GD rather than SGD, so no batch is required
	with open('training.csv') as f:
		ncols = len(f.readline().split(','))
	train_data = torch.from_numpy(np.loadtxt('training.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	train_loader, val_data, test_data, maxcounts = utils.load_and_split(args)
	
	if 'NB' not in args.loss_function:
	# directly calcualte the analytical solutions for ZIP/Poisson/Categorical distributions
		training = train_data.T
		allpara=[]
		if args.loss_function == 'C':
			maxmax = torch.max(maxcounts)
			for i in range(len(maxcounts)):
				allpara.append(cateMLE(training[i],maxmax,maxcounts[i]))
			allpara=torch.tensor(allpara)
			loss_test = -torch.distributions.categorical.Categorical(probs=allpara).log_prob(test_data).sum(-1)
			samples = torch.distributions.categorical.Categorical(probs=allpara).sample([args.sample_size])

		elif args.loss_function == 'ZIP':
			allpara = torch.tensor([ZIPMLE(column) for column in training],dtype=float).T
			strength = allpara[0]
			pzero = allpara[1]
			loss_test = -dis.ZeroInflatedPoisson(strength, gate=pzero).log_prob(test_data).sum(-1)
			samples=dis.ZeroInflatedPoisson(strength, gate=pzero).sample([args.sample_size])

		elif args.loss_function == 'P':
			# for Poisson distribution, the analytic solution is the mean
			strength = torch.tensor([column.float().mean() for column in training])
			loss_test = -torch.distributions.Poisson(strength).log_prob(test_data).sum(-1)
			samples=torch.distributions.Poisson(strength).sample([args.sample_size])

	else:
		# GD for NB/ZINB distributions
		utils.setup_seed(args.seed)
		seeds = np.random.choice(range(10000),size=args.experiment_num,replace=False)
		best_model = float('inf')

		for i in range(args.experiment_num):
			logging.info('This is experiment {:01d}'.format(i+1))
			utils.setup_seed(seeds[i])
			model = base_NB(args)

			model.to(device)
			optimizer = torch.optim.Adam(model.parameters(), args.lr)
			bad_epochs = 0
			best_val = float('inf')
			

			for epoch in range(args.max_epoch):
				model.train()
				optimizer.zero_grad()

				if bad_epochs == args.patience_lr:
					optimizer.param_groups[0]['lr']/=2

				likelihood = model(train_data.to(device))
				loss=-likelihood.mean(dim = 0)
				loss.backward()
				grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
				optimizer.step()

				model.eval()
				with torch.no_grad():
					val_loss = -model(val_data.to(device)).mean(dim=0)

				logging.info('current_val:{:.4g}, previous_best_val:{:.4g}'.format(val_loss, best_val))

				if val_loss < best_val:
					best_val = val_loss
					bad_epochs = 0
					state=model.state_dict()
				else:
					bad_epochs += 1

				if bad_epochs >= args.patience:
					logging.info('No validation loss improvements observed for {:d} epochs. Early stop!'.format(args.patience))
					break

			if best_val < best_model:
				best_model=best_val
				with torch.no_grad():
					model.load_state_dict(state)
					loss_test = -model(test_data.to(device))
					samples=model.generate(args.sample_size)
				

	loss_test=loss_test.cpu().detach().numpy()
	samples=samples.cpu().detach().numpy()
	np.savetxt(os.path.join(savedir,'loss.csv'),loss_test,delimiter=',')
	np.savetxt(os.path.join(savedir,'samples.csv'),samples,delimiter=',')
	
def ZIPMLE(data):
	# analytical maximum likelihood estimates for ZIP distribution
	t1 = torch.sign(data).int().sum()
	t2 = torch.sum(data)
	ratio = t2/t1
	strength = (lambertw(-ratio*torch.exp(-ratio))+ratio).real
	pzero = 1-t2/(strength*len(data))

	if pzero < 0:
		pzero = 0

	return [strength, pzero]

def cateMLE(data, maxmax, smoothing):
	# analytical maximum likelihood estimates for categorical distribution
	fre = Counter(data.tolist()+list(range(smoothing+1)))
	rela_fre = [fre.get(value, 0)/(len(data)+smoothing) for value in range(maxmax)]

	return rela_fre



if __name__ == '__main__':
	args = get_args()

	logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
						format='%(levelname)s: %(message)s')
	if args.log_file is not None:
		
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		logging.getLogger('').addHandler(console)

	main(args)





