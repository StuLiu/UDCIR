
# Code Partially Forked from NTIRE2020 Denoising Challenge @ CVPR
# Revised by Yuqian Zhou
import numpy as np
import os.path
import shutil
import torch
from scipy.io.matlab.mio import savemat, loadmat
from model import Restorer
from torch import from_numpy

print('begin')
import cv2
def restoration(udc):
    # TODO: plug in your method here
    print('udc.shape', udc.shape)
    # cv2.imshow('img', udc)
    # cv2.waitKey(0)
    data_batch = from_numpy(np.array([udc]).transpose((0, 3, 1, 2))).float()
    model = Restorer(image_c=3, N=64)
    model.load_state_dict(torch.load(
        f='pkls/model_1441.pkl',
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    output_batch = model(data_batch).cpu().numpy()
    result = output_batch.transpose((0, 2, 3, 1))[0,:,:,:]
    print('result.shape', result.shape)
    # cv2.imshow('img', result)
    # cv2.waitKey(0)
    return result


# TODO: update your working directory; it should contain the .mat file containing noisy images
work_dir = './'

# load noisy images
udc_fn = 'toled_val_display.mat'  # or poled_val_display.mat
udc_key = 'val_display'
udc_mat = loadmat(os.path.join(work_dir, udc_fn))[udc_key]

# restoration
n_im, h, w, c = udc_mat.shape
results = udc_mat.copy()
for i in range(n_im):
    udc = np.reshape(udc_mat[i, :, :, :], (h, w, c))
    restored = restoration(udc)
    results[i, :, :, :] = restored

# create results directory
res_dir = 'res_dir'
os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

# save denoised images in a .mat file with dictionary key "results"
res_fn = os.path.join(work_dir, res_dir, 'results.mat')
res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
savemat(res_fn, {res_key: results})

# submission indormation
# TODO: update the values below; the evaluation code will parse them
runtime = 0.0  # seconds / megapixel
cpu_or_gpu = 0  # 0: GPU, 1: CPU
method = 1  # 0: traditional methods, 1: deep learning method
other = '(optional) any additional description or information'

# prepare and save readme file
readme_fn = os.path.join(work_dir, res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
with open(readme_fn, 'w') as readme_file:
    readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
    readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
    readme_file.write('Method: %s\n' % str(method))
    readme_file.write('Other description: %s\n' % str(other))

# compress results directory
res_zip_fn = 'results_dir'
shutil.make_archive(os.path.join(work_dir, res_zip_fn), 'zip', os.path.join(work_dir, res_dir))
print('finished')