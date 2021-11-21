from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
def gpu_limit():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.per_process_gpu_memory_fraction=0.7
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)