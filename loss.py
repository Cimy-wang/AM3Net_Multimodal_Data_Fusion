import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
	def __init__(self, size_average=False, loss_lambda=0.5):
		"""
		Margin loss for digit existence
		L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

		Args:
			size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
			loss_lambda: parameter for down-weighting the loss for missing digits
		"""
		super(MarginLoss, self).__init__()
		self.size_average = size_average
		self.m_plus = 0.9
		self.m_minus = 0.1
		self.loss_lambda = loss_lambda

	def forward(self, inputs, labels):
		L_k = labels * F.relu(self.m_plus - inputs)**2 + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
		L_k = L_k.sum(dim=1)

		if self.size_average:
			return L_k.mean()
		else:
			return L_k.sum()
