import torch
import torch.nn as nn
import torch.nn.functional as F
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
        ignore_mask = (label != 255)  # shape: (B, H, W)

        safe_labels = label.clone()
        safe_labels[~ignore_mask] = 0  # Replace ignored pixels with 0 temporarily
        
        true_labels_unsq = safe_labels.unsqueeze(1).long()  # (B, 1, H, W)
        true_logits = torch.gather(model_output, 1, true_labels_unsq)  # (B, 1, H, W)
        
        B, C, H, W = model_output.shape
        mask = torch.ones_like(model_output, dtype=torch.bool)
        mask.scatter_(1, true_labels_unsq, False)
        other_logits = model_output.masked_fill(~mask, float('-inf'))
        
        max_other_logits, _ = other_logits.max(dim=1, keepdim=True)
        
        loss_per_pixel = max_other_logits - true_logits  # (B, 1, H, W)
        loss_per_pixel = loss_per_pixel.squeeze(1)      # (B, H, W)
        
        valid_loss = loss_per_pixel[ignore_mask]
        loss = valid_loss.mean()
        return loss

    def compute_trainloss(self, F, omega = 1.2):
        """
        Compute the adaptive loss function
        """
        # Final adaptive loss
        loss = F.sum()
        return loss

    def compute_target_trainloss(self, F,output, label):
        """
        Compute the adaptive loss function
        """
        # Final adaptive loss
        l2_norm = torch.sum(F**2)
        ce = nn.CrossEntropyLoss(ignore_index=self.config.train.ignore_label)
        ce_loss = ce(output, label.long())
        loss = 0.9*l2_norm + ce_loss
        return loss

    def compute_loss_direct(self, model_output, label):
        """
        Compute the adaptive loss function
        """
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss = ce_loss(model_output, label.long())  # Compute loss for all pixels
        return loss 

    def compute_cos_warmup_loss(self, adv_ft_map, rand_ft_map, pred, target):
        #Compute cosine similarity between adv and rand ft map
        # Flatten to vectors
        v1 = adv_ft_map.reshape(-1)
        v2 = rand_ft_map.reshape(-1)
    
        # Normalize
        v1_norm = v1 / (v1.norm(dim=0) + 1e-8)
        v2_norm = v2 / (v2.norm(dim=0) + 1e-8)
    
        # Cosine similarity
        cos_sim = torch.dot(v1_norm, v2_norm)
        if (cos_sim<0.8):
            return -cos_sim, cos_sim
        else:
            N, C, H, W = pred.shape
            pred_softmax = F.softmax(pred, dim=1)
            target_flat = target.view(-1)
            pred_label = pred_softmax.argmax(dim=1)
    
            # Flatten for per-pixel comparison
            pred_label_flat = pred_label.view(-1)
            correct_mask = (pred_label_flat == target_flat) & (target_flat != self.ignore_label)
            incorrect_mask = (pred_label_flat != target_flat) & (target_flat != self.ignore_label)
    
            loss = F.cross_entropy(pred, target.long(), ignore_index=self.ignore_label, reduction='none').view(-1)
    
            # total_pixels = float(correct_mask.sum() + incorrect_mask.sum() + 1e-8)
            loss_correct = loss[correct_mask]
            loss_incorrect = loss[incorrect_mask]
            
            # Avoid empty tensors
            if loss_correct.numel() > 0:
                loss_correct = (loss_correct - loss_correct.min()) / (loss_correct.max() - loss_correct.min() + 1e-8)
            else:
                loss_correct = torch.tensor(0., device=loss.device)
            
            if loss_incorrect.numel() > 0:
                loss_incorrect = (loss_incorrect - loss_incorrect.min()) / (loss_incorrect.max() - loss_incorrect.min() + 1e-8)
            else:
                loss_incorrect = torch.tensor(0., device=loss.device)
    
            loss_weighted = (0.3) * loss_correct.mean() + \
                            0.7 * loss_incorrect.mean()
            loss_cos = F.relu(cos_sim - 0.75)
            
            return loss_weighted, cos_sim # - 0.5*loss_cos

    def compute_entropy_loss(self, adv_ft_map, rand_ft_map):
        v1 = adv_ft_map.view(adv_ft_map.size(0), -1)
        p = F.softmax(v1, dim=-1)
        loss = - (p * p.log()).sum(dim=-1).mean()
        return loss
