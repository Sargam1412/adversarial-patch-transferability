import torch
import torch.nn as nn

class PatchLoss(nn.Module):
    def __init__(self, config):
        super(PatchLoss, self).__init__()
        self.config = config
        self.device = config.experiment.device
        self.ignore_label = config.train.ignore_label


    def compute_gradloss(self, model_output, label):
        """
        Compute the adaptive loss function
        """

        #print(model_output.shape,true_labels.shape)
        #print(model_output.argmax(dim=1).shape)
        # ce_loss = nn.CrossEntropyLoss(reduction="none",
        #                               ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        # loss_map = ce_loss(model_output, label.long())  # Compute loss for all pixels shape (N,H,W)
        # #print(f'loss map: {loss_map.shape}')
        # # Create mask for valid pixels
        # mask = (label.long() != self.config.train.ignore_label)

        # # Apply mask and compute mean over H and W for each image
        # masked_loss = loss_map * mask  # zeros out ignored pixels
        # sum_loss_per_image = masked_loss.view(model_output.size(0), -1).sum(dim=1)  # sum over H*W
        # valid_pixels_per_image = mask.view(model_output.size(0), -1).sum(dim=1)     # count of valid pixels per image

        # # Avoid division by zero
        # mean_loss_per_image = sum_loss_per_image / (valid_pixels_per_image + 1e-6)
      
        # # Get correctly classified and misclassified pixel sets
        # predict = torch.argmax(model_output, 1).float() + 1
        # target = label.float() + 1
        # target[target>=255] = 0
        # # print(predict.dtype,predict.shape,target.dtype,target.shape)
        # # temp1 = (predict == target).float()
        # # temp2 = (target>0).float()
        # # print(temp1.dtype,temp2.dtype)
        # correct_mask = (predict == target)*(target > 0)
        # incorrect_mask = (predict != target)*(target > 0)  # Opposite of correctly classified
        # #print(f'Correct mask: {correct_mask.shape}')  
        # # Compute separate loss terms
        # loss_correct = (loss_map * correct_mask).sum()/correct_mask.sum()  # Loss on correctly classified pixels
        # loss_incorrect = (loss_map * incorrect_mask).sum()/incorrect_mask.sum()  # Loss on already misclassified pixels

        # # Compute adaptive balancing factor
        # num_correct = correct_mask.sum()
        # num_total = (target != 0).sum()
        # gamma = num_correct / num_total  # Avoid division by zero

        # # Final adaptive loss
        # loss = gamma * loss_correct + (1 - gamma) * loss_incorrect
        # # print(f'Gamma:{gamma}')
        # # print(f'loss correct:{loss_correct}')
        # # print(f'loss incorrect: {loss_incorrect}')
        # #return loss
        # return loss_correct 
        #return mean_loss_per_image
        true_labels_unsq = label.unsqueeze(1).long()  # (B, 1, H, W)
        true_logits = torch.gather(model_output, 1, true_labels_unsq)  # (B, 1, H, W)
    
        B, C, H, W = model_output.shape
        mask = torch.ones_like(model_output, dtype=torch.bool)
        mask.scatter_(1, true_labels_unsq, False)
    
        other_logits = model_output.masked_fill(~mask, float('-inf'))
        max_other_logits, _ = other_logits.max(dim=1, keepdim=True)
    
        loss_per_pixel = max_other_logits - true_logits
        loss = loss_per_pixel.mean()
    
        return loss

    def compute_trainloss(self, F, omega = 1.2):
        """
        Compute the adaptive loss function
        """
        # Final adaptive loss
        loss = F.sum()
        return loss

    def compute_loss_direct(self, model_output, label):
        """
        Compute the adaptive loss function
        """
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss = ce_loss(model_output, label.long())  # Compute loss for all pixels
        return loss 
