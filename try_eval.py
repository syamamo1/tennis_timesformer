print('at start')
import torch
from timesformer.models.vit import TimeSformer

path_timesformer = '/ifs/CS/replicated/home/syamamo1/course/robust_fp/TimeSformer'
path_data = '/ifs/CS/replicated/home/syamamo1/course/robust_fp/data'
model_file = '/checkpoints/TimeSformer_divST_96x4_224_K600.pyth'
model_file = path_timesformer + model_file

print('loading model')

model = TimeSformer(img_size=224, num_classes=2, num_frames=96, attention_type='divided_space_time',  pretrained_model=str(model_file))
print('loaded')

# (batch x channels x frames x height x width)
dummy_video = torch.randn(8, 3, 96, 224, 224) 
print('making pred')
pred = model(dummy_video,) # (batch, classes)
print(pred)
print(pred.shape)