import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


class QuantizationLayer(nn.Module):

	def __init__(self, num_code_words, max_labels, max_samples):
		'''
		initialize quantization layer

		:param num_code_words: define the qunatization resolution , log(num_code_words) is the number of bits used in quantization
		:param max_labels: maximum value of the target values
		:param max_samples: maximum value of the samples
		'''
		super(QuantizationLayer, self).__init__()
		self.a = torch.nn.Parameter(
			data=torch.from_numpy(
				np.ones(num_code_words - 1) * max_labels / num_code_words
			), requires_grad=False)
		self.b = torch.nn.Parameter(
			data=torch.from_numpy(
				np.linspace(-1, 1, num_code_words - 1) * max_samples),
			requires_grad=True)
		if len(self.b) > 1:
			self.c = torch.nn.Parameter(
				data=torch.from_numpy(
					(15 / np.mean(np.diff(self.b.data.numpy()))) * np.ones(num_code_words - 1)
				), requires_grad=False)
		else:
			self.c = torch.nn.Parameter(data=torch.from_numpy(15 / int(self.b.data) * np.ones(num_code_words - 1)),
			                            requires_grad=False)
		self.num_code_words = num_code_words

	def forward(self, x):
		print(x.shape)
		z = torch.zeros(self.num_code_words - 1, x.shape[0], x.shape[1]).double()
		for i in range(self.num_code_words - 1):
			z[i, :, :] = self.a[i] * torch.tanh(self.c[i] * (x - self.b[i]))
		return torch.sum(z, dim=0)

	def plot(self, ro):
		bdiff = torch.max(torch.from_numpy(np.diff(self.b.data.numpy())))
		x_vals = np.linspace(np.min(self.b.data.numpy()) - bdiff, np.max(self.b.data.numpy()) + bdiff, 1000)
		quant = []
		for x_val in x_vals:
			quant.append(torch.sum(self.a * torch.tanh(self.c * (x_val - self.b))))
		plt.figure(ro)
		plt.title(str(ro))
		plt.plot(x_vals, quant)


import tensorflow as tf
from tensorflow.keras import layers


class KerasQuantizationLayer(layers.Layer):

	def __init__(self, num_code_words, max_labels, max_samples):
		'''
		initialize quantization layer

		:param num_code_words: define the qunatization resolution , log(num_code_words) is the number of bits used in quantization
		:param max_labels: maximum value of the target values
		:param max_samples: maximum value of the samples
		'''
		super(KerasQuantizationLayer, self).__init__()
		self.init_a = np.ones(num_code_words - 1) * max_labels / num_code_words
		self.a = tf.Variable(initial_value=self.init_a,
		                     trainable=False)
		self.init_b = np.linspace(-1, 1, num_code_words - 1) * max_samples
		self.b = tf.Variable(initial_value=self.init_b, trainable=True)

		if len(self.init_b) > 1:
			self.init_c = (15 / np.mean(np.diff(self.init_b))) * np.ones(num_code_words - 1)
			self.c = tf.Variable(initial_value=self.init_c,
			                     trainable=False)
		else:
			self.init_c = 15 / int(self.init_b) * np.ones(num_code_words - 1)
			self.c = tf.Variable(initial_value=self.init_c, trainable=False)
		self.num_code_words = num_code_words

	def call(self, x):
		outputs = []
		for i in range(self.num_code_words - 1):
			quant_step = self.init_a[i] * tf.math.tanh(self.init_c[i] * (x - self.init_b[i]))
			outputs.append(quant_step)
		z = tf.stack(outputs)
		return tf.math.reduce_sum(z, axis=0)


class HardQuantizationLayer(nn.Module):

	def __init__(self, a, b, c, num_code_words):
		super(HardQuantizationLayer, self).__init__()
		self.a = a
		self.b = b
		self.c = c
		self.num_code_words = num_code_words

	def quant_in_b(self, x):
		z = torch.zeros(self.num_code_words - 1, len(x)).double()
		x = x.detach().numpy()
		b = np.sort(self.b.detach().numpy())
		ind = np.digitize(x, b)
		x = (b[ind - 1] + b[ind]) / 2
		x = torch.from_numpy(x)
		for i in range(self.num_code_words - 1):
			z[i, :] = self.a[i] * torch.tanh(self.c[i] * (x - self.b[i]))
		return torch.sum(z, dim=0)

	def forward(self, x):
		z = torch.zeros(x.shape).double()
		z[x <= self.b[0]] = -torch.sum(self.a)
		z[x > self.b[-1]] = torch.sum(self.a)
		z[(x > self.b[0]) & (x <= self.b[-1])] = self.quant_in_b(x[(x > self.b[0]) & (x <= self.b[-1])])
		return z


if __name__ == '__main__':
	x = tf.ones((4, 4), dtype=tf.float64)
	print('**********************')
	tf.keras.backend.print_tensor(
		x, message='hello'
	)
	print('**********************')

	# quant_layer = KerasQuantizationLayer(8, 15, 15)
	# y = quant_layer(x)
