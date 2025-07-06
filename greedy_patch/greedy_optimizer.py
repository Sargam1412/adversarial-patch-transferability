import sys
import pickle
# Save the original sys.path
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")
from dataset.cityscapes import Cityscapes

from pretrained_models.models import Models
from pretrained_models.PIDNet.model import PIDNet, get_pred_model

from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch

import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# Restore original sys.path to avoid conflicts or shadowing
sys.path = original_sys_path

class GreedyPatchOptimizer:
    def __init__(self, config, main_logger):
        """
        Initialize the greedy patch optimizer.
        
        Args:
            config: Configuration object with parameters
            main_logger: Logger for printing information
        """
        self.config = config
        self.device = config.experiment.device
        self.logger = main_logger
        self.patch_size = config.patch.size
        self.apply_patch = Patch(config).apply_patch
        
        # Dataset setup
        cityscape_train = Cityscapes(
            root=config.dataset.root,
            list_path=config.dataset.train,
            num_classes=config.dataset.num_classes,
            multi_scale=config.train.multi_scale,
            flip=config.train.flip,
            ignore_label=config.train.ignore_label,
            base_size=config.train.base_size,
            crop_size=(config.train.height, config.train.width),
            scale_factor=config.train.scale_factor
        )
        
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=cityscape_train,
            batch_size=1,  # We process one image at a time for greedy optimization
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            drop_last=config.train.drop_last
        )
        
        # Model setup - specifically load PIDNet-s
        self.model1 = Models(self.config)
        self.model1.get()
        self.model2 = Models(self.config)
        self.model2.get()
        
        # Loss function
        self.criterion = PatchLoss(self.config)
        
        # Metrics
        self.metric = SegmentationMetric(config)
        
        # Patch adjustment parameters
        self.delta = 2/255  # The adjustment step size (+2/255, 0, -2/255)
        
        self.H3 = torch.load('/kaggle/working/logs/H3_pidnet_l.pt', map_location = self.device)
        self.H2 = torch.load('/kaggle/working/logs/H2_pidnet_l.pt', map_location = self.device)
        self.H4 = torch.load('/kaggle/working/logs/H4_pidnet_l.pt', map_location = self.device)
        self.H3 /= torch.norm(self.H3, p=2, dim=(2,3), keepdim=True) + 1e-8 
        self.H2 /= torch.norm(self.H2, p=2, dim=(2,3), keepdim=True) + 1e-8
        self.H4 /= torch.norm(self.H4, p=2, dim=(2,3), keepdim=True) + 1e-8

        self.layer1_name = 'layer1.0.bn2' 
        self.feature_map1_shape = [64, 256, 256]
        self.layer2_name = 'layer3.2.bn2' 
        self.feature_map2_shape = [256,64,64]
        self.layer3_name = 'layer3_.1.bn2' 
        self.feature_map3_shape = [128, 128, 128]
        self.layer4_name = 'layer5.1.relu' 
        self.feature_map4_shape = [512, 16, 16]

        self.feature_maps_adv1 = None
        self.feature_maps_adv2 = None
        self.feature_maps_adv3 = None
        self.feature_maps_adv4 = None
        self.feature_maps_rand1 = None
        self.feature_maps_rand2 = None
        self.feature_maps_rand3 = None
        self.feature_maps_rand4 = None

    # Hook to store feature map
    def hook1(self, module, input, output):
      self.feature_maps_adv1 = output
      if output.requires_grad:
        output.retain_grad()

    def hook12(self, module, input, output):
      self.feature_maps_adv2 = output
      if output.requires_grad:
        output.retain_grad()

    def hook13(self, module, input, output):
      self.feature_maps_adv3 = output
      if output.requires_grad:
        output.retain_grad()

    def hook14(self, module, input, output):
      self.feature_maps_adv4 = output
      if output.requires_grad:
        output.retain_grad()

    def hook2(self, module, input, output):
      self.feature_maps_rand1 = output
      if output.requires_grad:
        output.retain_grad()

    def hook22(self, module, input, output):
      self.feature_maps_rand2 = output
      if output.requires_grad:
        output.retain_grad()

    def hook23(self, module, input, output):
      self.feature_maps_rand3 = output
      if output.requires_grad:
        output.retain_grad()

    def hook24(self, module, input, output):
      self.feature_maps_rand4 = output
      if output.requires_grad:
        output.retain_grad()

    def register_forward_hook1(self):
      for name, module in self.model1.model.named_modules():
        # if name == self.layer1_name:
        #   module.register_forward_hook(self.hook1)
        if name == self.layer2_name:
          module.register_forward_hook(self.hook12)
        if name == self.layer3_name:
          module.register_forward_hook(self.hook13)
        if name == self.layer4_name:
          module.register_forward_hook(self.hook14)

    def register_forward_hook2(self):
      for name, module in self.model2.model.named_modules():
        # if name == self.layer1_name:
        #   module.register_forward_hook(self.hook2)
        if name == self.layer2_name:
          module.register_forward_hook(self.hook22)
        if name == self.layer3_name:
          module.register_forward_hook(self.hook23)
        if name == self.layer4_name:
          module.register_forward_hook(self.hook24)

    def load_patch(self, patch_path=None):
            """
            Load an existing patch or create a new random one if path is None.
        
            Args:
                patch_path: Path to the existing patch file
        
            Returns:
                The loaded or newly created patch
            """
            if patch_path:
                # First try the exact path
                if os.path.exists(patch_path):
                    self.logger.info(f"Loading patch from {patch_path}")
                    try:
                        # Try torch.load
                        patch = torch.load(patch_path, map_location=self.device, weights_only=False)
                    except Exception as e:
                        self.logger.warning(f"torch.load failed: {e}, trying pickle.load...")
                        patch = pickle.load(open(patch_path, "rb"))[0]  # Your patches are pickled lists
                    return patch.to(self.device)
        
                # Try alternate extensions
                base_path, ext = os.path.splitext(patch_path)
                for try_ext in ['.pt', '.p']:
                    try_path = base_path + try_ext
                    if os.path.exists(try_path):
                        self.logger.info(f"Loading patch from {try_path}")
                        try:
                            patch = torch.load(try_path, map_location=self.device, weights_only=False)
                        except Exception as e:
                            self.logger.warning(f"torch.load failed: {e}, trying pickle.load...")
                            patch = pickle.load(open(try_path, "rb"))[0]
                        return patch.to(self.device)
        
                self.logger.info(f"Could not find patch at {patch_path} with .pt or .p")
                self.logger.info("Creating new random patch")
            else:
                self.logger.info("No patch path provided. Creating new random patch")
        
            # Create a new random patch if no file found or no path provided
            patch1 = torch.rand((3, 100, 100), device=self.device)
            patch2 = torch.rand((3, 100, 100), device=self.device)
            patch3 = torch.rand((3, 100, 100), device=self.device)
            patch4 = torch.rand((3, 100, 100), device=self.device)
            return patch1, patch2, patch3, patch4

    def compute_grad(self, image, patch1, patch2, patch3, patch4, true_label, idx):
        """
        Compute a pixel-level priority map based on gradient values.
        
        Args:
            image: The input image
            patch: The current patch
            true_label: The ground truth label
            
        Returns:
            A priority map with values for each pixel in the patch
        """
        # Create a copy of the patch that requires gradients
        patch_with_grad1 = patch1.clone().detach().requires_grad_(True)
        patch_with_grad2 = patch2.clone().detach().requires_grad_(True)
        patch_with_grad3 = patch3.clone().detach().requires_grad_(True)
        patch_with_grad4 = patch4.clone().detach().requires_grad_(True)
        rand_patch1 = torch.rand((3, 100, 100), device=self.device)
        rand_patch2 = torch.rand((3, 100, 100), device=self.device)
        rand_patch3 = torch.rand((3, 100, 100), device=self.device)
        rand_patch4 = torch.rand((3, 100, 100), device=self.device)
        self.register_forward_hook1()
        self.register_forward_hook2()
        
        # Apply the patch to the image
        patched_image_adv, patched_label = self.apply_patch(image, true_label, patch_with_grad1, patch_with_grad2, patch_with_grad3, patch_with_grad4)
        patched_image_rand, patched_label = self.apply_patch(image, true_label, rand_patch1, rand_patch2, rand_patch3, rand_patch4)
        
        # Ensure label is the correct dtype for loss computation
        patched_label = patched_label.long()

        # Forward pass
        output1 = self.model1.predict(patched_image_adv, patched_label.shape)
        output2 = self.model2.predict(patched_image_rand, patched_label.shape)
        
        # Compute loss
        # with torch.no_grad():
        #     clean_output = self.model.predict(image, patched_label.shape)
            
        # # Compute loss (using the same loss as in the reference implementation)
        # pred_labels = output.argmax(dim=1)
        # correct_pixels = (pred_labels == patched_label) & (patched_label != self.config.train.ignore_label)
        # num_correct = correct_pixels.sum().item()
        for i in range(image.shape[0]):
            F3 = ((self.feature_maps_adv3[i]-self.feature_maps_rand3[i])*self.H3[idx[i]]) + (self.H3[idx[i]])**2
            F2 = ((self.feature_maps_adv2[i]-self.feature_maps_rand2[i])*self.H2[idx[i]]) + (self.H2[idx[i]])**2
            F4 = ((self.feature_maps_adv4[i]-self.feature_maps_rand4[i])*self.H4[idx[i]]) + (self.H4[idx[i]])**2
        # if num_correct > 0:
        #     loss = self.criterion.compute_loss_transegpgd_stage1(output, patched_label, clean_output)
        # else:
        #     loss = self.criterion.compute_loss_transegpgd_stage2(output, patched_label, clean_output)
        
        #loss1 = self.criterion.compute_trainloss(F1)
        loss2 = self.criterion.compute_trainloss(F2)
        loss3 = self.criterion.compute_trainloss(F3)
        loss4 = self.criterion.compute_trainloss(F4)
        #loss = self.criterion.compute_loss_direct(output, patched_label)
        #total_loss += (loss1.item() + loss2.item() + loss4.item())# + loss3.item()
        grad1 = torch.autograd.grad(loss3, patch_with_grad1, retain_graph=True)[0]
        grad2 = torch.autograd.grad(loss2, patch_with_grad2, retain_graph=True)[0]
        grad3 = torch.autograd.grad(loss2, patch_with_grad3, retain_graph=True)[0]
        grad4 = torch.autograd.grad(loss4, patch_with_grad4, retain_graph=True)[0]
        # Compute gradients
        # loss.backward()
        
        # # Get the gradients
        # patch_grad = patch_with_grad.grad
        
        # Compute priority based on sum of absolute gradient values across channels
        # priority_map1 = torch.sum(torch.abs(grad1), dim=0)
        # priority_map2 = torch.sum(torch.abs(grad2), dim=0)
        # priority_map3 = torch.sum(torch.abs(grad3), dim=0)
        # priority_map4 = torch.sum(torch.abs(grad4), dim=0)
        
        return grad1, grad2, grad3, grad4
    
    def evaluate_patch(self, image, patch1, patch2, patch3, patch4, true_label):
        """
        Evaluate the effectiveness of a patch by measuring misclassification.
        
        Args:
            image: The input image
            patch: The patch to evaluate
            true_label: The ground truth label
            
        Returns:
            A score representing the effectiveness of the patch (higher is better)
        """
        with torch.no_grad():
            # Apply the patch to the image
            patched_image, patched_label = self.apply_patch(image, true_label, patch1, patch2, patch3, patch4)
            
            # Forward pass
            output = self.model1.predict(patched_image, patched_label.shape)
            
            # Compute metrics
            pred_labels = output.argmax(dim=1)
            correct_pixels = (pred_labels == patched_label) & (patched_label != self.config.train.ignore_label)
            num_correct = correct_pixels.sum().item()
            
            # Lower number of correct pixels means better patch
            return -num_correct
    
    def optimize_patch(self, initial_patch=None, num_images=10, save_path=None):
        """
        Perform greedy pixel-wise patch optimization.
        
        Args:
            initial_patch: Initial patch to start with (if None, a random one will be created)
            num_images: Number of images to use for optimization
            save_path: Path to save the optimized patch
            
        Returns:
            The optimized patch and IoU metrics
        """
        # Load or create initial patch
        momentum1 = torch.tensor(0, dtype=torch.float32).to(self.device)
        momentum2 = torch.tensor(0, dtype=torch.float32).to(self.device)
        momentum3 = torch.tensor(0, dtype=torch.float32).to(self.device)
        momentum4 = torch.tensor(0, dtype=torch.float32).to(self.device)
        adv_patch1, adv_patch2, adv_patch3, adv_patch4 = self.load_patch(initial_patch)
        
        start_time = time.time()
        self.logger.info("Starting greedy pixel-wise patch optimization")
        
        # Keep track of IoU metrics
        IoU = []
        
        # Process a subset of images
        for ep in range(30):
            for i_iter, batch in enumerate(self.train_dataloader):
                if i_iter >= num_images:
                    break
                    
                image, true_label, _, _, _, index = batch
                image, true_label = image.to(self.device), true_label.to(self.device)
                
                self.logger.info(f"Processing image {i_iter+1}/{num_images}")
                
                # Compute priority map for this image
                grad1, grad2, grad3, grad4 = self.compute_grad(image, adv_patch1, adv_patch2, adv_patch3, adv_patch4, true_label, index)
                with torch.no_grad():
                  momentum1 = (0.9*momentum1) + (grad1/ (torch.norm(grad1) + 1e-8))
                  self.adv_patch1 += self.epsilon * momentum1.sign()
                  self.adv_patch1.clamp_(0, 1)  # Keep pixel values in valid range
                  momentum2 = (0.9*momentum2) + (grad2/ (torch.norm(grad2) + 1e-8))
                  self.adv_patch2 += self.epsilon * momentum2.sign()
                  self.adv_patch2.clamp_(0, 1)  # Keep pixel values in valid range
                  momentum3 = (0.9*momentum3) + (grad3/ (torch.norm(grad3) + 1e-8))
                  self.adv_patch3 += self.epsilon * momentum3.sign()
                  self.adv_patch3.clamp_(0, 1)  # Keep pixel values in valid range
                  momentum4 = (0.9*momentum4) + (grad3/ (torch.norm(grad3) + 1e-8))
                  self.adv_patch4 += self.epsilon * momentum4.sign()
                  self.adv_patch4.clamp_(0, 1)  # Keep pixel values in valid range
                
                # Evaluate the patch after processing this image
                with torch.no_grad():
                    patched_image, patched_label = self.apply_patch(image, true_label, adv_patch1, adv_patch2, adv_patch3, adv_patch4)
                    output = self.model1.predict(patched_image, patched_label.shape)
                    self.metric.update(output, patched_label)
                    pixAcc, mIoU = self.metric.get()
                    
                    self.logger.info(f"Image {i_iter+1} complete - mIoU: {mIoU:.4f}, pixAcc: {pixAcc:.4f}")
                    IoU.append(self.metric.get(full=True))
                    self.metric.reset()
            
                # Save the optimized patch if a path is provided
                # if save_path:
                #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                #     torch.save(patch, save_path)
                #     self.logger.info(f"Saved optimized patch to {save_path}")
            safety1 = adv_patch1.clone()
            pickle.dump( safety1.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss_amap_greedy1"+".p", "wb" ) )
            safety2 = adv_patch2.clone()
            pickle.dump( safety2.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss_amap_greedy2"+".p", "wb" ) )
            safety3 = adv_patch3.clone()
            pickle.dump( safety3.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss_amap_greedy3"+".p", "wb" ) )
            safety4 = adv_patch4.clone()
            pickle.dump( safety4.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss_amap_greedy4"+".p", "wb" ) )
            self.logger.info(f"Epoch {ep} completed")
            
        total_time = time.time() - start_time
        self.logger.info(f"Greedy optimization completed in {str(datetime.timedelta(seconds=int(total_time)))}")
        
        return adv_patch1.detach(), adv_patch2.detach(), adv_patch3.detach(), adv_patch4.detach(), np.array(IoU)

# # For adv_patch1
#             # Flatten the priority map and get indices sorted by priority (highest to lowest)
#             flat_priorities = priority_map1.view(-1)
#             sorted_indices = torch.argsort(flat_priorities, descending=True)
            
#             # Convert flat indices to 2D coordinates
#             h, w = priority_map1.shape
#             y_coords = sorted_indices // w
#             x_coords = sorted_indices % w
            
#             # Process pixels in order of priority
#             for idx in range(50):
#                 y, x = y_coords[idx].item(), x_coords[idx].item()
                
#                 # Try different adjustments for each channel
#                 best_score = float('-inf')
#                 best_adjustment = torch.zeros(3, device=self.device)
                
#                 for c in range(3):  # For each color channel
#                     for adj in [-self.delta, 0, self.delta]:
#                         # Create a temporary patch with this adjustment
#                         temp_patch = adv_patch1.clone()
#                         temp_patch[c, y, x] += adj
#                         temp_patch.clamp_(0, 1)  # Keep in valid range
                        
#                         # Evaluate this adjustment
#                         score = self.evaluate_patch(image, temp_patch, adv_patch2, adv_patch3, adv_patch4, true_label)
                        
#                         if score > best_score:
#                             best_score = score
#                             best_adjustment[c] = adj
                
#                 # Apply the best adjustment to the patch
#                 for c in range(3):
#                     adv_patch1[c, y, x] += best_adjustment[c]
                
#                 # Ensure patch values stay in valid range
#                 adv_patch1.clamp_(0, 1)
                
#                 # Log progress periodically
#                 if idx % 10 == 0:
#                     self.logger.info(f"Processed {idx}/{len(sorted_indices)} pixels for patch1")

#             # For adv_patch2
#             # Flatten the priority map and get indices sorted by priority (highest to lowest)
#             flat_priorities = priority_map2.view(-1)
#             sorted_indices = torch.argsort(flat_priorities, descending=True)
            
#             # Convert flat indices to 2D coordinates
#             h, w = priority_map2.shape
#             y_coords = sorted_indices // w
#             x_coords = sorted_indices % w
            
#             # Process pixels in order of priority
#             for idx in range(50):
#                 y, x = y_coords[idx].item(), x_coords[idx].item()
                
#                 # Try different adjustments for each channel
#                 best_score = float('-inf')
#                 best_adjustment = torch.zeros(3, device=self.device)
                
#                 for c in range(3):  # For each color channel
#                     for adj in [-self.delta, 0, self.delta]:
#                         # Create a temporary patch with this adjustment
#                         temp_patch = adv_patch2.clone()
#                         temp_patch[c, y, x] += adj
#                         temp_patch.clamp_(0, 1)  # Keep in valid range
                        
#                         # Evaluate this adjustment
#                         score = self.evaluate_patch(image, adv_patch1, temp_patch, adv_patch3, adv_patch4, true_label)
                        
#                         if score > best_score:
#                             best_score = score
#                             best_adjustment[c] = adj
                
#                 # Apply the best adjustment to the patch
#                 for c in range(3):
#                     adv_patch2[c, y, x] += best_adjustment[c]
                
#                 # Ensure patch values stay in valid range
#                 adv_patch2.clamp_(0, 1)
                
#                 # Log progress periodically
#                 if idx % 10 == 0:
#                     self.logger.info(f"Processed {idx}/{len(sorted_indices)} pixels for patch2")

#             # For adv_patch3
#             # Flatten the priority map and get indices sorted by priority (highest to lowest)
#             flat_priorities = priority_map3.view(-1)
#             sorted_indices = torch.argsort(flat_priorities, descending=True)
            
#             # Convert flat indices to 2D coordinates
#             h, w = priority_map3.shape
#             y_coords = sorted_indices // w
#             x_coords = sorted_indices % w
            
#             # Process pixels in order of priority
#             for idx in range(50):
#                 y, x = y_coords[idx].item(), x_coords[idx].item()
                
#                 # Try different adjustments for each channel
#                 best_score = float('-inf')
#                 best_adjustment = torch.zeros(3, device=self.device)
                
#                 for c in range(3):  # For each color channel
#                     for adj in [-self.delta, 0, self.delta]:
#                         # Create a temporary patch with this adjustment
#                         temp_patch = adv_patch3.clone()
#                         temp_patch[c, y, x] += adj
#                         temp_patch.clamp_(0, 1)  # Keep in valid range
                        
#                         # Evaluate this adjustment
#                         score = self.evaluate_patch(image, adv_patch1, adv_patch2, temp_patch, adv_patch4, true_label)
                        
#                         if score > best_score:
#                             best_score = score
#                             best_adjustment[c] = adj
                
#                 # Apply the best adjustment to the patch
#                 for c in range(3):
#                     adv_patch3[c, y, x] += best_adjustment[c]
                
#                 # Ensure patch values stay in valid range
#                 adv_patch3.clamp_(0, 1)
                
#                 # Log progress periodically
#                 if idx % 10 == 0:
#                     self.logger.info(f"Processed {idx}/{len(sorted_indices)} pixels for patch3")

#             # For adv_patch4
#             # Flatten the priority map and get indices sorted by priority (highest to lowest)
#             flat_priorities = priority_map4.view(-1)
#             sorted_indices = torch.argsort(flat_priorities, descending=True)
            
#             # Convert flat indices to 2D coordinates
#             h, w = priority_map4.shape
#             y_coords = sorted_indices // w
#             x_coords = sorted_indices % w
            
#             # Process pixels in order of priority
#             for idx in range(50):
#                 y, x = y_coords[idx].item(), x_coords[idx].item()
                
#                 # Try different adjustments for each channel
#                 best_score = float('-inf')
#                 best_adjustment = torch.zeros(3, device=self.device)
                
#                 for c in range(3):  # For each color channel
#                     for adj in [-self.delta, 0, self.delta]:
#                         # Create a temporary patch with this adjustment
#                         temp_patch = adv_patch4.clone()
#                         temp_patch[c, y, x] += adj
#                         temp_patch.clamp_(0, 1)  # Keep in valid range
                        
#                         # Evaluate this adjustment
#                         score = self.evaluate_patch(image, adv_patch1, adv_patch2, adv_patch3, temp_patch, true_label)
                        
#                         if score > best_score:
#                             best_score = score
#                             best_adjustment[c] = adj
                
#                 # Apply the best adjustment to the patch
#                 for c in range(3):
#                     adv_patch4[c, y, x] += best_adjustment[c]
                
#                 # Ensure patch values stay in valid range
#                 adv_patch4.clamp_(0, 1)
                
#                 # Log progress periodically
#                 if idx % 10 == 0:
#                     self.logger.info(f"Processed {idx}/{len(sorted_indices)} pixels for patch4")
