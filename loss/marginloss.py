import torch.nn.functional as F
import torch


def MarginLoss(inputs, labels):
	"""
	Margin loss for digit existence
	Eq. (12): L_k = T_k * max(0, n+ - ||v_k||)^2 + mu * (1 - T_k) * max(0, ||v_k|| - n-)^2
	"""
	size_average = False
	n_plus, n_minus = 0.9, 0.1
	loss_mu = 0.025
	L_k = (labels * F.relu(n_plus - inputs)**2 + loss_mu * (1 - labels) * F.relu(inputs - n_minus)**2).sum(dim=1)
	if size_average:
		return L_k.mean()
	else:
		return L_k.sum()


if __name__ == "__main__":
	example = torch.load("example.pt")
	inputs = example['inputs']
	labels = example['labels']
	L_k = MarginLoss(inputs, labels)
	print('Inference Loss: %f' % L_k)
