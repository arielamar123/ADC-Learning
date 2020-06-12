import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy

###########################
# Parmeters Intialization #
###########################


num_rx_antenas = 6  # Number of Rx antennas
dense_sampling_L = 20  # Observed discretized time frame
num_transmitted_symbols = 4  # Number of transmitted symbols
valid_percent = 0.2  # Percentage of training data used for validation
train_size = int(1e4)  # Training size
validation_size = int(train_size * valid_percent)
test_size = int(1e5)  # Test data size
num_of_adc_p = 4  # Number of scalar quantizers
num_samples_L_tilde = 4  # Number of samples to produce over time frame
num_of_channels = 1  # 1 - Gaussian channel
BPSK_symbols = [-1, 1]  # BPSK symbols
error_variance = 0.1  # error variance for csi uncertainty
frame_size = 200
f_0 = 1e3
w = 2 * np.pi * f_0
snr_vals = 10 ** (0.1 * np.arange(13))
num_code_words = 8
epochs = 30
batch_size = 200
time_vec = np.arange(1, dense_sampling_L + 1) / dense_sampling_L
channel_matrix_cos = 1 + 0.5 * np.cos(time_vec)
gaussian_sampling_std = 0.4 * torch.ones(num_samples_L_tilde).double()
noise_vector = 1 + 0.3 * np.cos(1.5 * (np.arange(1, dense_sampling_L + 1)) + 0.2)

###########################
# channel matrix creation #
###########################


channel_matrix_exp = np.exp(-np.abs(np.ones((num_rx_antenas, 1)) * np.arange(num_transmitted_symbols) - (
		np.ones((num_transmitted_symbols, 1)) * [np.arange(num_rx_antenas)]).T))


def create_received_signal(data, ro=1):
	'''
	Creating x(t) = G(t)*s+w(t) the signal received by the antenna
	n - number of antennas
	L - number of dense samples
	N - data size
	:param data: sequence of symbols to pass through the channel (4, N)
	:param ro: SNR
	:return: x(t) = G(t)*s+w(t) (n*L, N)
	'''
	channel_matrix = np.zeros((num_rx_antenas * dense_sampling_L, data.shape[1]))
	for t in range(dense_sampling_L):
		signal = np.dot(channel_matrix_cos[t] * channel_matrix_exp, data)
		power_of_signal = signal.var()
		noise_power = power_of_signal / ro
		channel_matrix[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.sqrt(
			noise_power) * signal + np.random.randn(
			num_rx_antenas,
			data.shape[1])
	return channel_matrix

def create_received_signal_debug(data, ro=1):
	'''
	Creating x(t) = G(t)*s+w(t) the signal received by the antenna
	n - number of antennas
	L - number of dense samples
	N - data size
	:param data: sequence of symbols to pass through the channel (4, N)
	:param ro: SNR
	:return: x(t) = G(t)*s+w(t) (n*L, N)
	'''
	channel_matrix = np.zeros((num_rx_antenas * dense_sampling_L, data.shape[1]))
	for t in range(dense_sampling_L):
		signal = np.dot(channel_matrix_cos[t] * channel_matrix_exp, data)
		power_of_signal = signal.var()
		noise_power = power_of_signal / ro
		channel_matrix[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.sqrt(
			noise_power) * signal + np.random.randn(
			num_rx_antenas,
			data.shape[1])
	return channel_matrix


def create_received_signal_uncertanity_error(data, ro=1):
	'''
	Creating x(t) = G(t)*s+w(t) the signal received by the antenna
	n - number of antennas
	L - number of dense samples
	N - data size
	:param data: sequence of symbols to pass through the channel (4, N)
	:param ro: SNR
	:return: x(t) = G(t)*s+w(t) (n*L, N)
	'''

	noisy_channel_matrix = np.zeros((num_rx_antenas * dense_sampling_L, data.shape[1]))
	for t in range(dense_sampling_L):
		noisy_channel_matrix[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.random.randn(
			num_rx_antenas,
			data.shape[1])

	noisy_recieved_signal = np.zeros((num_rx_antenas * dense_sampling_L, data.shape[1]))
	for t in range(dense_sampling_L):
		signal = np.zeros((num_rx_antenas, data.shape[1]))
		noisy_channel_matrix_exp = channel_matrix_exp * (
				1 + np.sqrt(error_variance) * np.random.randn(channel_matrix_exp.shape[0],
				                                              channel_matrix_exp.shape[1]))
		for f in range(data.shape[1] // frame_size):
			signal[:, f * frame_size:(f + 1) * frame_size] = np.dot(channel_matrix_cos[t] * noisy_channel_matrix_exp,
			                                                        data[:, f * frame_size:(f + 1) * frame_size])
		noisy_recieved_signal[num_rx_antenas * t:num_rx_antenas * (t + 1), :] = np.sqrt(
			1 / ro) * signal
	return noisy_recieved_signal + noisy_channel_matrix


###############################
# Creating the Neural Network #
###############################

class AnalogNetwork(nn.Module):
	def __init__(self):
		super(AnalogNetwork, self).__init__()
		self.analog_filter = nn.Linear(num_rx_antenas * dense_sampling_L, num_of_adc_p * dense_sampling_L,
		                               bias=False).double()  # without bias to get only the matrix

	def forward(self, x):
		x = self.analog_filter(x)
		return x


class SamplingLayer(nn.Module):
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
						-(t - self.weight[k]) ** 2 / gaussian_sampling_std[k] ** 2), dim=1)
		return out


class HardSamplingLayer(nn.Module):
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

	def __init__(self):
		super(QuantizationLayer, self).__init__()
		self.a = torch.nn.Parameter(
			data=torch.from_numpy(
				np.ones(num_code_words - 1) * np.max(labels_train) / num_code_words
			), requires_grad=False)
		self.b = torch.nn.Parameter(
			data=torch.from_numpy(
				np.linspace(-1, 1, num_code_words - 1) * np.max(train_samples)),
			requires_grad=True)
		self.c = torch.nn.Parameter(
			data=torch.from_numpy(
				(15 / np.mean(np.diff(self.b.data.numpy()))) * np.ones(num_code_words - 1)
			), requires_grad=False)

	def forward(self, x):
		z = torch.zeros(num_code_words - 1, x.shape[0], x.shape[1]).double()
		for i in range(num_code_words - 1):
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


class HardQuantizationLayer(nn.Module):

	def __init__(self, a, b, c):
		super(HardQuantizationLayer, self).__init__()
		self.a = a
		self.b = b
		self.c = c

	def quant_in_b(self, x):
		z = torch.zeros(num_code_words - 1, len(x)).double()
		x = x.detach().numpy()
		b = self.b.detach().numpy()
		ind = np.digitize(x, b)
		x = (b[ind - 1] + b[ind]) / 2
		x = torch.from_numpy(x)
		for i in range(num_code_words - 1):
			z[i, :] = self.a[i] * torch.tanh(self.c[i] * (x - self.b[i]))
		return torch.sum(z, dim=0)

	# def quant_in_b(self, x):
	# 	z = torch.zeros(x.shape).double()
	# 	for i in range(len(x)):
	# 		for k in range(1, len(self.b)):
	# 			if self.b[k - 1] < x[i] and x[i] <= self.b[k]:
	# 				middleB = torch.mean(self.b[k - 1:k + 1])
	# 				z[i] = torch.sum(self.a * torch.tanh(self.c * (middleB - self.b)))
	# 				break
	# 	return z

	def forward(self, x):
		z = torch.zeros(x.shape).double()
		z[x <= self.b[0]] = -torch.sum(self.a)
		z[x > self.b[-1]] = torch.sum(self.a)
		z[(x > self.b[0]) & (x <= self.b[-1])] = self.quant_in_b(x[(x > self.b[0]) & (x <= self.b[-1])])
		return z


class DigitalNetwork(nn.Module):
	def __init__(self):
		super(DigitalNetwork, self).__init__()
		self.fc1 = torch.nn.Linear(num_of_adc_p * num_samples_L_tilde, 32, bias=False).double()
		self.fc2 = torch.nn.Linear(32, 16, bias=False).double()

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class ADCNet(nn.Module):
	def __init__(self):
		super(ADCNet, self).__init__()
		self.analog_network = AnalogNetwork()
		self.sampling_layer = SamplingLayer()
		self.quantization_layer = QuantizationLayer()
		self.digital_network = DigitalNetwork()

	def forward(self, x):
		x = self.analog_network(x)
		x = self.sampling_layer(x)
		x = self.quantization_layer(x)
		x = self.digital_network(x)
		return x


def train(detector,parameters):
	pass






#######################
#        main         #
#######################


# detector_list = [create_received_signal]
# plot_labels = ['Deep task based, perfect CSI']
detector_list = [create_received_signal, create_received_signal_uncertanity_error]
plot_labels = ['Deep task based, perfect CSI', 'Deep task based, CSI uncertainity']
for d, detector in tqdm(enumerate(detector_list)):
	BER = []
	for r, ro in enumerate(np.flip(snr_vals)):

		#################
		# Data Creation #
		#################

		train_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, train_size), p=[0.5, 0.5])
		validation_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, validation_size), p=[0.5, 0.5])
		test_data = np.random.choice(BPSK_symbols, (num_transmitted_symbols, test_size), p=[0.5, 0.5])
		labels_train = np.sum(0.5 * (train_data + 1).T * 2 ** np.flip(np.arange(4)), axis=1).astype(np.long)
		labels_validation = np.sum(0.5 * (validation_data + 1).T * 2 ** np.flip(np.arange(4)), axis=1).astype(np.long)
		labels_test = np.sum(0.5 * (test_data + 1).T * 2 ** np.flip(np.arange(4)), axis=1).astype(np.long)
		train_samples = detector(train_data, ro).T
		validation_samples = create_received_signal(validation_data, ro).T
		test_samples = create_received_signal(test_data, ro).T
		training_set = []
		validation_set = []
		test_set = []

		for i in range(len(train_samples)):
			training_set.append([train_samples[i], labels_train[i]])

		for i in range(len(validation_samples)):
			validation_set.append([validation_samples[i], labels_validation[i]])

		for i in range(len(test_samples)):
			test_set.append([test_samples[i], labels_test[i]])

		########################
		# Training the Network #
		########################

		net = ADCNet()
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(net.parameters(), lr=1e-2)
		training_loss_plot = []
		valid_loss_plot = []
		valid_loss_min = np.Inf

		for epoch in tqdm(range(epochs)):
			# keep track of training and validation loss
			train_loss = 0.0
			valid_loss = 0.0
			###################
			# train the model #
			###################
			net.train()
			for i, data in tqdm(enumerate(DataLoader(training_set, batch_size=batch_size))):
				inputs, labels = data
				# clear the gradients of all optimized variables
				optimizer.zero_grad()
				# forward pass: compute predicted outputs by passing inputs to the model
				outputs = net(inputs)
				# calculate the batch loss
				loss = criterion(outputs, labels.long())
				# backward pass: compute gradient of the loss with respect to model parameters
				loss.backward()
				# perform a single optimization step (parameter update)
				optimizer.step()
				# update training loss
				train_loss += loss.item() * batch_size
			######################
			# validate the model #
			######################
			net_val = deepcopy(net)
			net_val.eval()
			with torch.no_grad():
				net_val.sampling_layer = HardSamplingLayer(net.sampling_layer.weight)
				net_val.quantization_layer = HardQuantizationLayer(net.quantization_layer.a, net.quantization_layer.b,
				                                                   net.quantization_layer.c)
				for i, data in tqdm(enumerate(DataLoader(validation_set, batch_size=batch_size))):
					inputs, labels = data
					# forward pass: compute predicted outputs by passing inputs to the model
					outputs = net_val(inputs)
					# calculate the batch loss
					loss = criterion(outputs, labels.long())
					# update validation loss
					valid_loss += loss.item() * batch_size
				train_loss = train_loss / train_size
				valid_loss = valid_loss / validation_size
				training_loss_plot.append(train_loss)
				valid_loss_plot.append(valid_loss)
				print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
					epoch, train_loss, valid_loss))

				# save model if validation loss has decreased
				if valid_loss <= valid_loss_min:
					print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
						valid_loss_min,
						valid_loss))
					torch.save(net.state_dict(), 'model_cifar_' + str(r) + '.pt')
					valid_loss_min = valid_loss

		########################
		# Testing the Network #
		########################
		best_net = ADCNet()
		best_net.load_state_dict(torch.load('model_cifar_' + str(r) + '.pt'))
		net_test = deepcopy(best_net)
		net_test.eval()
		net_test.sampling_layer = HardSamplingLayer(best_net.sampling_layer.weight)
		net_test.quantization_layer = HardQuantizationLayer(best_net.quantization_layer.a,
		                                                    best_net.quantization_layer.b,
		                                                    best_net.quantization_layer.c)
		ber = None
		with torch.no_grad():
			for i, data in tqdm(enumerate(DataLoader(test_set, batch_size=len(test_samples)))):
				inputs, labels = data
				net_outputs = net_test(inputs)
				outputs = net_outputs.argmax(dim=1)
				power = 2 ** np.flip(np.arange(4))
				est_ber = np.floor((outputs.numpy()[:, None] % (2 * power)) / power).T
				est_ber_scale = 2 * est_ber - np.ones(test_data.shape)
				ber = np.mean(np.mean(est_ber_scale != test_data, axis=0))

			# wrong_bits = 0
			# for i, data in tqdm(enumerate(DataLoader(test_set, batch_size=batch_size))):
			# 	inputs, labels = data
			# 	net_outputs = net_test(inputs)
			# 	outputs = F.softmax(net_outputs).argmax(dim=1)
			# 	for label, output in zip(labels, outputs):
			# 		if label != output:
			# 			print('true is: ', label, "I got: ", output)
			# 		wrong_bits += bin(int(label) ^ int(output)).count('1') / 4
			# print('For ro = ', r, ' BER is: ', wrong_bits / test_size)
			# BER.append(wrong_bits / test_size)
			print('For ro = ', r, ' BER is: ', ber)
			BER.append(ber)
	plt.plot(np.arange(13), BER, marker='o', label=plot_labels[d])
########################
#   Plot the results   #
########################


plt.title('BER vs SNR Results')
plt.ylabel('BER')
plt.xlabel('SNR[db]')
plt.yscale('log')
plt.ylim((10 ** (-6), 10 ** (-1)))
plt.legend()
plt.show()
