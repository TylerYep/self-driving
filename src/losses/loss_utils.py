import torch
import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.sum((output - target)**2) / output.shape[0]


def compute_branch_masks(controls, num_targets):
	masks = []
	# if high_level_control = 0, branch index 0 is for Straight
	# if high_level_control = 1, branch index 1 is for Left
	# if high_level_control = 2, branch index 2 is for Right
	for i in range(3): # num branches
		mask = (controls == i)
		mask = mask.type(torch.float)
		mask = mask.reshape(-1, 1)
		mask = torch.cat([mask] * num_targets, dim=1)
		masks.append(mask)
	return masks

def branched_l2_loss(branches, targets, masks):
	loss_branches = []
	for i in range(3): # num branches
		loss_branches.append(((branches[i] - targets)**2) * masks[i])
	return torch.sum(torch.cat(loss_branches)) / branches[0].shape[0] # Take mean over batch size

# def l2_loss(outputs, targets):
# 	return torch.sum((outputs - targets)**2) / outputs.shape[0]
