import os
from tools import run_net

# Paths to transformer
path_timesformer = '/ifs/CS/replicated/home/syamamo1/course/robust_fp/TimeSformer'

# runner program 
runner = path_timesformer + '/tools/run_net.py'

# yaml config file
model = path_timesformer + '/configs/my_config.yaml'

# Make command string
com = f'''
python3 {runner} \
  --cfg {model} \
'''

# Call + Logging
print('\n\nSTARTING HERE\n')
# print(com, '\n\n')
# os.system(com)
print('\n\nFINISHED IN train_model.py')










