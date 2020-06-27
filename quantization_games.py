import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers

dense_sampling_L = 20  # Observed discretized time frame
num_of_adc_p = 4  # Number of scalar quantizers
num_samples_L_tilde = 4  # Number of samples to produce over time frame
gaussian_sampling_std = 0.4


class SamplingLayer(nn.Module):
	'''
	Sampling Pytorch version
	The input shape to this layer is (batch size, number_of_adc * dense sampling)
	'''

	def __init__(self):
		super(SamplingLayer, self).__init__()
		self.weight = torch.nn.Parameter(
			data=torch.from_numpy(
				np.arange(num_samples_L_tilde, dense_sampling_L, num_samples_L_tilde, dtype=np.float64)),
			requires_grad=True)

	def forward(self, x):
		out = torch.zeros((len(x), num_samples_L_tilde * num_of_adc_p)).double()
		t = torch.from_numpy(np.arange(1, dense_sampling_L + 1, dtype=np.double))
		for v, j in enumerate(range(0, dense_sampling_L * num_of_adc_p, dense_sampling_L)):
			for k in range(num_samples_L_tilde):
				out[:, v * num_samples_L_tilde + k] = torch.sum(
					x[:, j:j + dense_sampling_L] * torch.exp(
						-(t - self.weight[k]) ** 2 / gaussian_sampling_std ** 2), dim=1)
		return out


class KerasSamplingLayer(tf.keras.layers.Layer):
	'''
	Sampling Tensorflow version
	The input shape to this layer is (batch size, number_of_adc * dense sampling)
	'''

	def __init__(self):
		'''
		initialize sampling layer
		'''
		super(KerasSamplingLayer, self).__init__()
		self.init_weight = np.arange(num_samples_L_tilde, dense_sampling_L, num_samples_L_tilde, dtype=np.float32)
		self.weight = tf.Variable(initial_value=self.init_weight,
		                          trainable=True, dtype=tf.float32)

	def call(self, x):
		# out = tf.Variable(initial_value=np.zeros((len(inp), num_samples_L_tilde * num_of_adc_p), dtype=np.float32))
		outputs = []
		t = tf.Variable(initial_value=np.arange(1, dense_sampling_L + 1), dtype=tf.float32)
		for v, j in enumerate(range(0, dense_sampling_L * num_of_adc_p, dense_sampling_L)):
			for k in range(num_samples_L_tilde):
				sampling = tf.math.reduce_sum(
					x[:, j:j + dense_sampling_L] * tf.exp(
						-(t - self.weight[k]) ** 2 / gaussian_sampling_std ** 2), axis=1)
				outputs.append(sampling)
		return tf.stack(outputs, axis=1)

class HardSamplingLayer(nn.Module):
	'''
	Hard sampling Pytorch version
	TODO : Implement Tensorflow version
	'''
	def __init__(self, weight):
		super(HardSamplingLayer, self).__init__()
		rounded_weight = torch.round(weight)
		rounded_weight[rounded_weight < 1] = 1
		rounded_weight[rounded_weight > dense_sampling_L] = dense_sampling_L
		self.weight = rounded_weight.long()

	def forward(self, x):
		out = torch.zeros((len(x), num_samples_L_tilde * num_of_adc_p)).double()
		for i in range(num_of_adc_p):
			for j in range(num_samples_L_tilde):
				out[:, i * num_samples_L_tilde + j] = x[:, i * dense_sampling_L + self.weight[j]]
		return out


class QuantizationLayer(nn.Module):
	'''
		Quantization Pytorch version
		The input shape to this layer is (batch size, number_of_adc * num_samples_L_tilde)
	'''

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


class KerasQuantizationLayer(tf.keras.layers.Layer):
	'''
	Quantization Tensorflow version
	The input shape to this layer is (batch size, number_of_adc * num_samples_L_tilde)
	'''

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
	'''
	Hard quantization Pytorch version
	TODO : Implement Tensorflow version
	'''

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
	# inp = tf.ones((1, num_of_adc_p * dense_sampling_L), dtype=tf.float32)
	# print(inp)
	inp2 = tf.range(1, 81, dtype=tf.float32)
	rep = tf.constant([4])
	inp2 = tf.reshape(tf.tile(inp2, rep), [4, 80])
	print(inp2.shape)
	k_smp = KerasSamplingLayer()
	out = k_smp(inp2)
	print(out)
