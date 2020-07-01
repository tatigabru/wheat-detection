"""
Experiment configs
"""
experiment_name = 'effdet4'
experiment_tag = '_run1'

fold = 3
our_image_size = 512

num_workers = 2
train_batch_size = 4
inf_batch_size = 16

n_epochs=60 
factor=0.2
start_lr=1e-3
min_lr=1e-6 
lr_patience=2
overall_patience=10 
loss_delta=1e-4
gpu_number=1
