import torch
import torch.nn as nn
import os
import logging
import argparse
from tqdm import tqdm
from collections import OrderedDict
import utils
import numpy as np
from model import ard



def get_args():
	""" hyper-parameters. """
	parser = argparse.ArgumentParser('DAD model')

	# Add data arguments
	parser.add_argument("--seed", type=int, default=42, help="primary random seed")
	parser.add_argument("--experiment_num", type=int, default=5, help="number of experiments to conduct")

	# Add model arguments
	parser.add_argument('--net_width', default=4, type=int, help='ratio between hidden layer width and input layer width')
	parser.add_argument('--net_depth', default=2, type=int, help='number of hidden layers')

	# Add training arguments
	parser.add_argument('--batch_size', default=100, type=int, help='batch size for training')
	parser.add_argument('--max_epoch', default=50, type=int, help='force stop training at specified epoch')
	parser.add_argument('--clip_norm', default=4.0, type=float, help='clip threshold of gradients')
	parser.add_argument('--lr', default=0.005, type=float, help='intial learning rate')
	parser.add_argument('--patience_lr', default=5, type=int,
						help='number of epochs without improvement on validation set before halving the learning rate')
	parser.add_argument('--patience', default=10, type=int,
						help='number of epochs without improvement on validation set before early stopping')

	# Add checkpoint arguments
	parser.add_argument('--log_file', default='logging.log', help='path to save logs')
	parser.add_argument('--save_dir', default='checkpoint', help='path to save checkpoints')
	parser.add_argument('--restore_file', default='checkpoint_last', help='filename to load checkpoint')
	parser.add_argument('--save_interval', type=int, default=1, help='save a checkpoint every N epochs')
	parser.add_argument('--continue_last', action='store_true', help='continue training from the latest checkpoint')
	
	args = parser.parse_args()
	return args

def main(args):
	utils.init_logging(args)
	args.save_dir = 'ard'+args.save_dir+str(args.net_width)+' '+str(args.net_depth)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info('using '+str(device))

	utils.setup_seed(args.seed)
	seeds = np.random.choice(range(10000),size=args.experiment_num,replace=False)

	if os.path.exists(args.save_dir):
		files = os.listdir(args.save_dir)
		current_seed = sum(['last' in file for file in files])-1
		current_save = args.restore_file + str(current_seed) + '.pt'
		checkpoint_path = os.path.join(args.save_dir, current_save)
	else:
		checkpoint_path = ''


	if os.path.isfile(checkpoint_path) and args.continue_last:
		utils.setup_seed(current_seed)
		train_loader, val, test, maxcounts = utils.load_and_split(args)

		model=dis_flow_model(args, maxcounts)
		model=nn.DataParallel(model)
		model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), args.lr)
		state_dict = utils.load_checkpoint(checkpoint_path, model, optimizer)
		last_epoch=state_dict['epoch']
		bad_epochs = state_dict['bad_epochs']
		best_val=state_dict['best_val']
		val_hist=state_dict['val_hist']
		train_hist=state_dict['train_hist']
		logging.info("You are training the model from the latest checkpoint. Make sure you use the same arguments before interruption")
		continue_last = True

	else:
		current_seed = 0
		logging.info('No checkpoint, training the model from scratch')
		continue_last = False


	for experiment in range(current_seed, args.experiment_num):
		logging.info('This is experiment {:01d}'.format(experiment+1))

		save_name=args.restore_file + str(experiment) + '.pt'
		utils.setup_seed(seeds[experiment])
		if continue_last == False:
			train_loader, val, test, maxcounts = utils.load_and_split(args)

			model=ard(args, maxcounts)
			model=nn.DataParallel(model)
			model.to(device)
			optimizer = torch.optim.Adam(model.parameters(), args.lr)
			last_epoch = -1
			bad_epochs = 0
			best_val = float('inf')
			train_hist = []
			val_hist = []

		state_dict={'args':args}

		for epoch in range(last_epoch + 1, args.max_epoch):
			state_dict['epoch'] = epoch

			model.train()
			stats = OrderedDict()
			stats['loss'] = 0
			stats['grad_norm'] = 0
			stats['clip'] = 0
			stats['lr'] = 0
			if bad_epochs == args.patience_lr:
				optimizer.param_groups[0]['lr']/=2
			
			progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

			# Iterate over the training set
			for i, sample in enumerate(progress_bar):

				model.train()
				optimizer.zero_grad()

				losses = model(sample.to(device))
				loss = losses.mean(dim=0)
				
				loss.backward()
				for para in model.parameters():
					if para.grad is not None and (torch.isnan(para.grad).any()):
						noise = torch.randn_like(para.grad)
						para.grad = torch.where(torch.isnan(para.grad),noise, para.grad)

				grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
			
				optimizer.step()
				stats['loss'] += loss.detach().item() 
				stats['grad_norm'] += grad_norm
				stats['lr'] += optimizer.param_groups[0]['lr']
				stats['clip'] += 1 if grad_norm > args.clip_norm else 0
				progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
										 refresh=True)
			logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
				value / len(progress_bar)) for key, value in stats.items())))

			train_hist.append(stats['loss']/len(progress_bar))
			
			state_dict['train_hist']=train_hist
			state_dict['model']=model.state_dict()
			state_dict['optimizer']=optimizer.state_dict()
			model.eval()
			with torch.no_grad():
				losses = model(val.to(device))
				val_loss = losses.mean(dim = 0)
				
			val_hist.append(val_loss)
			state_dict['val_hist']=val_hist
			logging.info('current_val:{:.4g}, previous_best_val:{:.4g}'.format(val_loss, best_val))


			if val_loss < best_val - 0.01:
				best_val = val_loss
				bad_epochs = 0
			else:
				bad_epochs += 1

			state_dict['bad_epochs']=bad_epochs
			state_dict['best_val'] = best_val

			# Save checkpoints
			if epoch % args.save_interval == 0:
				utils.save_checkpoint(args.save_dir, save_name, state_dict)

			if bad_epochs >= args.patience:
				logging.info('No validation loss improvements observed for {:d} epochs. Early stop!'.format(args.patience))
				break

		logging.info('Experiment {:01d} finished'.format(experiment+1))
		continue_last = False



if __name__ == '__main__':
	args = get_args()

	logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
						format='%(levelname)s: %(message)s')
	if args.log_file is not None:
		
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		logging.getLogger('').addHandler(console)

	main(args)


