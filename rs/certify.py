'''
Anisotropic Randomized Smoothing Certification: Hard-RS and Soft-RS
Soft-RS uses empirical Bernstein bound

Acknowledgements:
[1] https://github.com/RuntianZ/macer
[2] https://github.com/locuslab/smoothing/blob/master/code/certify.py

References:
[1] J. Cohen, E. Rosenfeld and Z. Kolter. 
Certified Adversarial Robustness via Randomized Smoothing. In ICML, 2019.
'''

import numpy as np
import scipy.io as sio
from rs.core import C_Smooth as Smooth
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def sigma_comparison(sigma, ref):
  return (sigma >= ref).sum() / sigma.size

def certify_gen(model, generator, device, dataset, transform, num_classes, experiment_name,
                matfile=None, mode='hard', start_img=0, num_img=500, skip=1, N0=100, N=100000,
                alpha=0.001, batch=1000, verbose=False,
                grid=(0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25), beta=1.0):
  print('===certify(N={}, mode={})==='.format(N, mode))

  model.eval()
  generator.eval()
  smoothed_net = Smooth(model, generator, num_classes,
                        device, mode, beta)

  radius_hard = np.zeros((num_img,), dtype=np.float)
  radius_soft = np.zeros((num_img,), dtype=np.float)
  num_grid = len(grid)
  cnt_grid_hard = np.zeros((num_grid + 1,), dtype=np.int)
  cnt_grid_soft = np.zeros((num_grid + 1,), dtype=np.int)
  s_hard = 0.0
  s_soft = 0.0
  sigma_records = {'per_sample_sigma_min': [],
                   'min': 1e5,
                   'mean': 0,
                   'max':0,
                   'l2_min': 1e5,
                   'l2_mean': 0,
                   'l2_max':0}
  for i in tqdm(range(num_img)):
    img, target = dataset[start_img + i * skip]
    img = img.to(device)
    smoothed_net.sigma = generator(img[None,...])
    if mode == 'both':
      p_hard, r_hard, p_soft, r_soft, sigma_stats = smoothed_net.certify(
          img, N0, N, alpha, batch)
      sigma_records['per_sample_sigma_min'].append(sigma_stats['min'])
      sigma_records['min'] = min(sigma_records['min'], sigma_stats['min'])
      sigma_records['mean'] += sigma_stats['mean']
      sigma_records['max'] = max(sigma_records['max'], sigma_stats['max'])
      sigma_records['l2_min'] = min(sigma_records['l2_min'], sigma_stats['l2_min'])
      sigma_records['l2_mean'] += sigma_stats['l2_mean']
      sigma_records['l2_max'] = max(sigma_records['l2_max'], sigma_stats['l2_max'])
      
      correct_hard = int(p_hard == target)
      correct_soft = int(p_soft == target)
      if verbose:
        if correct_hard == 1:
          print('Hard Correct: 1. Radius: {}.'.format(r_hard))
        else:
          print('Hard Correct: 0.')
        if correct_soft == 1:
          print('Soft Correct: 1. Radius: {}.'.format(r_soft))
        else:
          print('Soft Correct: 0.')
      radius_hard[i] = r_hard if correct_hard == 1 else -1
      radius_soft[i] = r_soft if correct_soft == 1 else -1
      if correct_hard == 1:
        cnt_grid_hard[0] += 1
        s_hard += r_hard
        for j in range(num_grid):
          if r_hard >= grid[j]:
            cnt_grid_hard[j + 1] += 1
      if correct_soft == 1:
        cnt_grid_soft[0] += 1
        s_soft += r_soft
        for j in range(num_grid):
          if r_soft >= grid[j]:
            cnt_grid_soft[j + 1] += 1
    else:
      prediction, radius, sigma_stats = smoothed_net.certify(img, N0, N, alpha, batch)
      #print(f'Prediction: {prediction}, Target: {target}, Radius: {radius}')
      sigma_records['per_sample_sigma_min'].append(sigma_stats['min'])
      sigma_records['min'] = min(sigma_records['min'], sigma_stats['min'])
      sigma_records['mean'] += sigma_stats['mean']
      sigma_records['max'] = max(sigma_records['max'], sigma_stats['max'])
      sigma_records['l2_min'] = min(sigma_records['l2_min'], sigma_stats['l2_min'])
      sigma_records['l2_mean'] += sigma_stats['l2_mean']
      sigma_records['l2_max'] = max(sigma_records['l2_max'], sigma_stats['l2_max'])
    
      correct = int(prediction == target)
      if verbose:
        if correct == 1:
          print('Correct: 1. Radius: {}.'.format(radius))
        else:
          print('Correct: 0.')
      radius_hard[i] = radius if correct == 1 else -1
      if correct == 1:
        cnt_grid_hard[0] += 1
        s_hard += radius
        for j in range(num_grid):
          if radius >= grid[j]:
            cnt_grid_hard[j + 1] += 1
  sigma_records['mean'] /= i+1
  sigma_records['l2_mean'] /= i+1
  sigma_records['per_sample_sigma_min'] = np.array(sigma_records['per_sample_sigma_min'])
  ref=1.0
  winning_ratio = sigma_comparison(sigma_records['per_sample_sigma_min'], ref)
  print('===Certify Summary===')
  print('Total Image Number: {}'.format(num_img))
  print(f"===Sigma stats=== \n Overall Min: {sigma_records['min']}  Mean: {sigma_records['mean']}  Max: {sigma_records['max']}")
  print(f'{winning_ratio*100} % samples has minimal sigma greater than {ref}')
  if mode == 'both':
    print('===Hard certify===')
    print('Normalized radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
      print('Normalized radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_hard / num_img))
    print('===Soft certify===')
    print('Normalized radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_soft[0], cnt_grid_soft[0] / num_img * 100))
    for j in range(num_grid):
      print('Normalized radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_soft[j + 1], cnt_grid_soft[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_soft / num_img))
    if matfile is not None:
      sio.savemat(matfile, {'hard': radius_hard, 'soft': radius_soft})
  else:
    print('Normalized radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
      print('Normalized radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_hard / num_img))
    if matfile is not None:
      sio.savemat(matfile, {mode: radius_hard})