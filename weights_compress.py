#!/usr/bin/env python
import matplotlib
matplotlib.use('agg') #Remove me and all is well!
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/daiyan/Caffe/caffe/python')
import caffe
import numpy as np
from sklearn.cluster import KMeans
from scipy import fftpack

def uniform_global_quantize(weight, bits):
  weight_min = weight.min()
  weight_max = weight.max()
  quantized_range = float(2 ** (bits - 1) - 1)
  scale = quantized_range / abs(weight_max) if abs(weight_max) > abs(weight_min) else quantized_range / abs(weight_min)
  quantized_weight = np.around(weight * scale).astype(np.int8)
  dequantized_weight = quantized_weight.astype(np.float) / scale
  abs_diff = np.absolute((weight - dequantized_weight))
  return (quantized_weight, np.sum(abs_diff), np.sum(abs_diff) / np.sum(np.absolute(weight)))

def dct_2d(spatial):
  return fftpack.dct(fftpack.dct(spatial.T, norm='ortho').T, norm='ortho')

def idct_2d(freq):
  return fftpack.idct(fftpack.idct(freq.T, norm='ortho').T, norm='ortho')

def dct_compress(weight, ratio):
  shape_y, shape_x = weight.shape
  freq = dct_2d(weight)
  sorted_freq = np.sort(np.absolute(freq.flatten()))
  threshold = sorted_freq[min(int(freq.size * ratio), weight.size - 1)] 
  index = np.absolute(freq) >= threshold
  truncated_freq = np.multiply(freq, index) 
  dct_weight = idct_2d(truncated_freq)
  abs_diff = np.absolute((weight - dct_weight))
  return (dct_weight, np.sum(abs_diff), np.sum(abs_diff) / np.sum(np.absolute(weight)))
  
def uniform_local_quantize(weight, bits):
  shape_y, shape_x = weight.shape
  weight_min = weight.min(1)
  weight_max = weight.max(1)
  scale = np.zeros(weight_min.shape)
  quantized_range = float(2 ** (bits - 1) - 1)
  quantized_weight = np.zeros(weight.shape, dtype=np.int8)
  dequantized_weight = np.zeros(weight.shape, dtype=np.float)
  for y in range(shape_y):
    scale[y] = quantized_range / abs(weight_max[y]) if abs(weight_max[y]) > abs(weight_min[y]) else quantized_range / abs(weight_min[y])
    quantized_weight[y] = np.around(weight[y] * scale[y]).astype(np.int8)
  for y in range(shape_y):
    dequantized_weight[y] = quantized_weight[y].astype(np.float) / scale[y]
  abs_diff = np.absolute((weight - dequantized_weight))
  return (quantized_weight, np.sum(abs_diff), np.sum(abs_diff) / np.sum(np.absolute(weight)))
  
def kmeans_global_quantize(weight, n):
  src_shape = weight.shape
  dst_shape = (-1, 1)
  reshaped_weights = weight.reshape(dst_shape)
  kmeans = KMeans(n_clusters=n, init='random', random_state=0, n_init=1)
  kmeans.fit(reshaped_weights)
  centers = kmeans.cluster_centers_
  kmeans_weight = centers[kmeans.predict(reshaped_weights)]
  kmeans_weight.resize(src_shape)
  abs_diff = np.absolute((weight - kmeans_weight))
  return (kmeans_weight, np.sum(abs_diff), np.sum(abs_diff) / np.sum(np.absolute(weight)))

def kmeans_local_quantize(weight, n):
  shape_y, shape_x = weight.shape
  kmeans_weight = np.zeros(weight.shape)
  dst_shape = (-1, 1)
  for y in range(shape_y):
    kmeans = KMeans(n_clusters=n, init='random', random_state=0, n_init=1)
    local_weight = weight[y].reshape(dst_shape)
    kmeans.fit(local_weight)
    centers = kmeans.cluster_centers_
    kmeans_weight[y] = centers[kmeans.predict(local_weight)].reshape(shape_x, )
  abs_diff = np.absolute((weight - kmeans_weight))
  return (kmeans_weight, np.sum(abs_diff), np.sum(abs_diff) / np.sum(np.absolute(weight)))

def pca_compress(weight, ratio):
    shape_y, shape_x = weight.shape
    u, s, v = np.linalg.svd(weight, full_matrices=False)
    final_channel = int(shape_y * ratio)
    ul = u[:, :final_channel]
    sl = s[:final_channel]
    vl = v[:final_channel, :]
    pca_weight = np.dot(np.diag(sl), vl)
    depca_weight = np.dot(ul, pca_weight)
    abs_diff = np.absolute((weight - depca_weight))
    return (pca_weight, np.sum(abs_diff), np.sum(abs_diff) / np.sum(np.absolute(weight)))

"""
net = caffe.Net('/home/daiyan/Caffe/caffe/models/bvlc_reference_caffenet/deploy.prototxt',
                '/home/daiyan/Caffe/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)
"""
net = caffe.Net('/home/daiyan/Caffe/caffe/models/bn_alexnet/deploy.prototxt',
                '/home/daiyan/Caffe/caffe/models/bn_alexnet/alexnet_cvgj_iter_320000.caffemodel',
                caffe.TEST)

fc6_weights = net.params['fc6'][0].data
fc7_weights = net.params['fc7'][0].data
fc8_weights = net.params['fc8'][0].data

print(dct_compress(fc6_weights, 1.0)[2])
print(dct_compress(fc6_weights, 0.75)[2])
print(dct_compress(fc6_weights, 0.5)[2])
print(dct_compress(fc6_weights, 0.25)[2])
print(dct_compress(fc6_weights, 0)[2])
print('local quantization:')
print(uniform_local_quantize(fc6_weights, 4)[2])
print(uniform_local_quantize(fc6_weights, 5)[2])
print(uniform_local_quantize(fc6_weights, 6)[2])
print(uniform_local_quantize(fc6_weights, 7)[2])
print(uniform_local_quantize(fc6_weights, 8)[2])
print('global quantization:')
print(uniform_global_quantize(fc6_weights, 4)[2])
print(uniform_global_quantize(fc6_weights, 5)[2])
print(uniform_global_quantize(fc6_weights, 6)[2])
print(uniform_global_quantize(fc6_weights, 7)[2])
print(uniform_global_quantize(fc6_weights, 8)[2])
print('pca quantization:')
print(pca_compress(fc6_weights, 0.8)[2])
print(pca_compress(fc6_weights, 0.9)[2])
print(pca_compress(fc6_weights, 1.0)[2])
print('local kmeans centers:')
print(kmeans_local_quantize(fc6_weights, 8)[2])
print(kmeans_local_quantize(fc6_weights, 16)[2])
print(kmeans_local_quantize(fc6_weights, 32)[2])
print(kmeans_local_quantize(fc6_weights, 64)[2])
print(kmeans_local_quantize(fc6_weights, 128)[2])
print(kmeans_local_quantize(fc6_weights, 256)[2])
