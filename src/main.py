import neptune
import numpy as np
# The init() function called this way assumes that 
# NEPTUNE_API_TOKEN environment variable is defined.
neptune.__version__
print(neptune.__version__)

neptune.init('blonde/wheat')

# Define parameters
PARAMS = {'decay_factor' : 0.5,
          'n_iterations' : 117}

# Create experiment with defined parameters
neptune.create_experiment (name='example_with_parameters',
                          params=PARAMS, 
                          tags=['examples'],
                          upload_source_files=['main.py', 'train.py'])
# add tags 
neptune.append_tags('metric_logging', 'exploration')    

# log some metrics
for i in range(100):
    neptune.log_metric('loss', 0.95**i)
neptune.log_metric('AUC', 0.96)

# Log image data
array = np.random.rand(10, 10, 3)*255
array = np.repeat(array, 30, 0)
array = np.repeat(array, 30, 1)
neptune.log_image('mosaics', array)

# Log text data
neptune.log_text('top questions', 'what is machine learning?')

# log some file
# replace this file with your own file from local machine
neptune.log_artifact('folds/folds.csv')  

#