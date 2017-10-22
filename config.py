import os
import glob


input_shape = [32, 32, 3]

batch_size = 256
num_classes = 10

checkpoint_steps = 100
save_steps = 1000
num_steps = 250000

model_name = 'net'
log_dir = 'log'

data_dir = '/home/brady/data/cifar-10-batches-py/'
train_path = list(sorted(glob.glob(os.path.join(data_dir, 'data_batch*'))))
valid_path = list(sorted(glob.glob(os.path.join(data_dir, 'test_batch*'))))
