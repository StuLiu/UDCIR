
# Code Partially Forked from NTIRE2020 Denoising Challenge @ CVPR
# Revised by Yuqian Zhou
import numpy as np
import os.path
import shutil
import torch
from scipy.io.matlab.mio import savemat, loadmat
from model import UNet, CNN
from torch import from_numpy
import sys
# import cv2
# from preprocess import image_crop, image_splice

print('begin')
# python prep_result.py UNet-32 pkls/UNet-32/model_16000.pkl cuda
model_name = sys.argv[1]
pkl_path = sys.argv[2]
dev = sys.argv[3]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and dev=='cuda' else "cpu")
print(model_name, pkl_path, DEVICE)
if model_name == 'UNet-16':
    model = UNet(N=16)
elif model_name == 'UNet-32':
    model = UNet(N=32)
elif model_name == 'UNet-64':
    model = UNet(N=64)
elif model_name == 'CNN':
    model = CNN(N=32)
else:
    model = None
    print('>>> Model name:{} invalid!')
    exit(-1)

def restoration(udc, model):
    # TODO: plug in your method here
    print('udc.shape', udc.shape)
    # cv2.imshow('img', udc)
    # cv2.waitKey(0)
    data_batch = from_numpy(np.array([udc]).transpose((0, 3, 1, 2))).float().to(DEVICE)
    with torch.no_grad():
        output_batch = model(data_batch).cpu().numpy()
        output_batch = np.where(output_batch < 0, 0, output_batch)
        output_batch = np.where(output_batch > 255, 255, output_batch)
    result = output_batch.transpose((0, 2, 3, 1))[0,:,:,:].astype('uint8')
    print('result.shape', result.shape)
    # cv2.imshow('img', result)
    # cv2.waitKey(0)
    return result

# TODO: update your working directory; it should contain the .mat file containing noisy images
work_dir = './'

# load model
model = model.to(DEVICE)
model.load_state_dict(torch.load(f=pkl_path, map_location=DEVICE))

# load noisy images
udc_fn = 'toled_val_display.mat'  # or poled_val_display.mat
udc_key = 'val_display'
udc_mat = loadmat(os.path.join(work_dir, udc_fn))[udc_key]
# print(type(udc_mat), udc_mat[0,0:5,0:5,:])

# restoration
n_im, h, w, c = udc_mat.shape
results = udc_mat.copy()
for i in range(n_im):
    udc = np.reshape(udc_mat[i, :, :, :], (h, w, c))
    restored = restoration(udc, model)
    results[i, :, :, :] = restored
# print(type(results), results[0,0:5,0:5,:])
# exit(0)

# create results directory
res_dir = 'res_dir'
os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

# save denoised images in a .mat file with dictionary key "results"
res_fn = os.path.join(work_dir, res_dir, 'results.mat')
res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
savemat(res_fn, {res_key: results})

# submission information
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

# def restoration(udc, model):
#     print('udc.shape', udc.shape)   # (h, w, c)
#     cropped_imgs = np.array(image_crop(udc))
#     data_batch = from_numpy(cropped_imgs.transpose((0, 3, 1, 2))).float().to(DEVICE)
#     with torch.no_grad():
#         output_batch = model(data_batch).cpu().numpy()
#         output_batch = np.where(output_batch < 0, 0, output_batch)
#         output_batch = np.where(output_batch > 255, 255, output_batch)
#     enhanced_imgs = output_batch.transpose((0, 2, 3, 1))
#     enhanced_img = image_splice(enhanced_imgs)
#     print('result.shape', enhanced_img.shape)
#     # cv2.imshow('img', enhanced_img)
#     # cv2.waitKey(0)
#     return enhanced_img