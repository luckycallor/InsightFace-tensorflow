import io
import os
from scipy import misc

import numpy as np
import pickle


read_path = r'F:\FaceDataset\faces_vgg_112x112\lfw.bin'
save_dir = r'F:\FaceDataset\faces_vgg_112x112\lfw_img_sample'


bins, issame_list = pickle.load(open(read_path, 'rb'), encoding='bytes')
cnt = 0
for bin in bins:
    img = misc.imread(io.BytesIO(bin))
    print('============================================')
    print(img.dtype)
    print(np.max(img))
    print(np.min(img))
    print('============================================')
    misc.imsave(os.path.join(save_dir, str(cnt)+'.jpg'), img)
    cnt += 1
    if cnt >= 10:
        break