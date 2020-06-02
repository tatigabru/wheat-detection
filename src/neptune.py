import neptune

# The init() function called this way assumes that 
# NEPTUNE_API_TOKEN environment variable is defined.
# Windows
# set NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTExN2QzMzQtMDJlYi00ODkzLTk5YTktYWNhNzg4MWFjZGQ3In0="
# Linux
# export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTExN2QzMzQtMDJlYi00ODkzLTk5YTktYWNhNzg4MWFjZGQ3In0="

neptune.init('blonde/sandbox')
neptune.create_experiment(name='minimal_example')

# log some metrics

for i in range(100):
    neptune.log_metric('loss', 0.95**i)

neptune.log_metric('AUC', 0.96)