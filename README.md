# Modeling Object-Coocurences in Images
## Dataset
The first two columns are the image source and ID (reordered after downloading).  
The rest columns (column name = object class) are the original data. Each row corresponds to a count vector.
## Codes
The model.py saves the structure of each model. The utils.py are auxiliary functions to build the models.  
The rest py files are for the training of different models. The file name indicates the model name. These files are similar, so I mainly annotate the vae.py file, please see it first.

Example command to train a VAE model:
python vae.py --loss_function C --inter_layer 160 160 --hidden_dimension 8

See the py files for more hyperparameters.

For each model, it will build a folder saving the best and last checkpoints of each experiment, and a logging file. If the training is interrupted, 
adding --continue_last to your commands to train it from the latest checkpoint (can be useful for mixture models with hundreds of components).
