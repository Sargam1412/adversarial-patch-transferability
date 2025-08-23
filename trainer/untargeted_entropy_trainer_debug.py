import sys
sys.path.append("/kaggle/working/adversarial-patch-transferability")
from dataset.cityscapes import Cityscapes

from pretrained_models.models import Models

from pretrained_models.ICNet.icnet import ICNet
from pretrained_models.BisNetV1.model import BiSeNetV1
from pretrained_models.BisNetV2.model import BiSeNetV2
from pretrained_models.PIDNet.model import PIDNet, get_pred_model

from metrics.performance import SegmentationMetric
from metrics.loss import PatchLoss
from patch.create import Patch
from torch.optim.lr_scheduler import ExponentialLR
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

class PatchTrainerDebug():
  def __init__(self,config,main_logger,model_name, coords, resume=False, patch=None):
      self.config = config
      self.start_epoch = 0
      self.end_epoch = 1
      self.epochs = self.end_epoch - self.start_epoch
      self.batch_train = config.train.batch_size
      self.batch_test = config.test.batch_size
      self.device = config.experiment.device
      self.logger = main_logger
      self.lr = config.optimizer.init_lr
      self.power = config.train.power
      self.lr_scheduler = config.optimizer.exponentiallr
      self.lr_scheduler_gamma = config.optimizer.exponentiallr_gamma
      self.log_per_iters = config.train.log_per_iters
      self.patch_size = config.patch.size
      self.apply_patch = Patch(config).apply_patch
      self.apply_patch_grad = Patch(config).apply_patch_grad
      self.apply_patch_rand = Patch(config).apply_patch_rand
      self.epsilon = config.optimizer.init_lr
      self.coords=coords
    
      cityscape_train = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.train,
          num_classes = config.dataset.num_classes,
          multi_scale = config.train.multi_scale,
          flip = config.train.flip,
          ignore_label = config.train.ignore_label,
          base_size = config.train.base_size,
          crop_size = (config.train.height,config.train.width),
          scale_factor = config.train.scale_factor
        )

      cityscape_test = Cityscapes(
          root = config.dataset.root,
          list_path = config.dataset.val,
          num_classes = config.dataset.num_classes,
          multi_scale = False,
          flip = False,
          ignore_label = config.train.ignore_label,
          base_size = config.test.base_size,
          crop_size = (config.test.height,config.test.width),
        )
      
      self.train_dataloader = torch.utils.data.DataLoader(dataset=cityscape_train,
                                              batch_size=self.batch_train,
                                              shuffle=False,
                                              num_workers=config.train.num_workers,
                                              pin_memory=config.train.pin_memory,
                                              drop_last=config.train.drop_last)
      self.test_dataloader = torch.utils.data.DataLoader(dataset=cityscape_test,
                                            batch_size=self.batch_test,
                                            shuffle=False,
                                            num_workers=config.test.num_workers,
                                            pin_memory=config.test.pin_memory,
                                            drop_last=config.test.drop_last)
      

      self.iters_per_epoch = len(self.train_dataloader)
      self.max_iters = self.end_epoch * self.iters_per_epoch

      ## Getting the model
      self.model1 = Models(self.config)
      self.model1.get()

      self.model2 = Models(self.config)
      self.model2.get()

      ## loss
      self.criterion = PatchLoss(self.config)
      
      # Initialize adversarial patch (random noise)
      if (resume==False):
        low, high = -2.1, 2.6
        self.adv_patch = (low + (high - low) * torch.rand((3, 200, 200), 
                                                  device=self.device,
                                                  requires_grad=True)).detach()
        self.adv_patch.requires_grad_()
      else:
        self.adv_patch = patch.clone().detach().to(self.device)
        self.adv_patch.requires_grad = True
      
      ## Initializing quantities
      self.metric = SegmentationMetric(config) 
      self.current_mIoU = 0.0
      self.best_mIoU = 0.0

      self.current_epoch = 0
      self.current_iteration = 0
      self.model_name=model_name

      # Register hook
      if 'pidnet_s' in self.model_name:
        self.layer_name = 'layer3.2.bn2' 
        self.feature_map_shape = [128,64,64]
      elif 'pidnet_m' in self.model_name:
        self.layer_name = 'layer3.2.bn2' 
        self.feature_map_shape = [256,64,64]
      elif 'pidnet_l' in self.model_name:
        self.layer_name = 'layer3.2.bn2' 
        self.feature_map_shape = [256,64,64]
      elif 'bisenet' in self.model_name: 
        self.layer_name = 'segment.S3.1.relu'
        self.feature_map_shape=[32,128,128]
      else:
        self.layer_name = 'pretrained.layer2.3.relu'
        self.feature_map_shape=[512,32,32]

      self.feature_maps_adv = None
      self.feature_maps_rand = None
      
      # Store gradient debugging information
      self.gradient_debug_info = {
          'loss_wrt_feature_maps': [],
          'feature_maps_wrt_patch': [],
          'patch_gradients': [],
          'loss_values': [],
          'feature_map_norms': [],
          'patch_updates': []
      }
    
  # Hook to store feature map
  def hook1(self, module, input, output):
      self.feature_maps_adv = output
      output.retain_grad()

  def hook2(self, module, input, output):
      self.feature_maps_rand = output
      output.retain_grad()

  def register_forward_hook1(self):
    for name, module in self.model1.model.named_modules():
        if name == self.layer_name:
          module.register_forward_hook(self.hook1)

  def register_forward_hook2(self):
    for name, module in self.model2.model.named_modules():
        if name == self.layer_name:
          module.register_forward_hook(self.hook2)
  
  def debug_gradients_before_backward(self, loss, feature_maps_adv, feature_maps_rand, iteration):
      """
      Debug gradient flow BEFORE backward pass - only check basic properties
      """
      debug_info = {}
      
      # 1. Check feature map statistics (no gradients yet)
      if feature_maps_adv is not None:
          debug_info['feature_maps_adv_norm'] = torch.norm(feature_maps_adv).item()
          debug_info['feature_maps_adv_mean'] = feature_maps_adv.mean().item()
          debug_info['feature_maps_adv_std'] = feature_maps_adv.std().item()
          debug_info['feature_maps_adv_min'] = feature_maps_adv.min().item()
          debug_info['feature_maps_adv_max'] = feature_maps_adv.max().item()
          debug_info['feature_maps_adv_shape'] = feature_maps_adv.shape
      else:
          debug_info['feature_maps_adv_norm'] = 0.0
          debug_info['feature_maps_adv_mean'] = 0.0
          debug_info['feature_maps_adv_std'] = 0.0
          debug_info['feature_maps_adv_min'] = 0.0
          debug_info['feature_maps_adv_max'] = 0.0
          debug_info['feature_maps_adv_shape'] = None
      
      # 2. Check patch properties (no gradients yet)
      debug_info['patch_shape'] = self.adv_patch.shape
      debug_info['patch_norm'] = torch.norm(self.adv_patch).item()
      debug_info['patch_requires_grad'] = self.adv_patch.requires_grad
      debug_info['patch_grad_exists'] = self.adv_patch.grad is not None
      
      # 3. Check loss properties (no gradients yet)
      debug_info['loss_value'] = loss.item()
      debug_info['loss_shape'] = loss.shape
      debug_info['loss_requires_grad'] = loss.requires_grad
      
      # 4. Check if setup is correct for gradients
      debug_info['setup_correct'] = (
          feature_maps_adv is not None and 
          self.adv_patch.requires_grad and 
          loss.requires_grad
      )
      
      # Store debug info
      self.gradient_debug_info['loss_wrt_feature_maps'].append(0.0)  # Will be updated after backward
      self.gradient_debug_info['feature_maps_wrt_patch'].append(0.0)  # Will be updated after backward
      self.gradient_debug_info['patch_gradients'].append(0.0)  # Will be updated after backward
      self.gradient_debug_info['loss_values'].append(debug_info['loss_value'])
      self.gradient_debug_info['feature_map_norms'].append(debug_info['feature_maps_adv_norm'])
      
      # Log detailed debug information
      self.logger.info(f"=== GRADIENT DEBUG ITERATION {iteration} (BEFORE BACKWARD) ===")
      self.logger.info(f"Feature Maps Adv - Shape: {debug_info['feature_maps_adv_shape']}")
      self.logger.info(f"Feature Maps Adv - Norm: {debug_info['feature_maps_adv_norm']:.6f}, Mean: {debug_info['feature_maps_adv_mean']:.6f}, Std: {debug_info['feature_maps_adv_std']:.6f}")
      self.logger.info(f"Patch - Shape: {debug_info['patch_shape']}, Norm: {debug_info['patch_norm']:.6f}, Requires Grad: {debug_info['patch_requires_grad']}")
      self.logger.info(f"Loss - Value: {debug_info['loss_value']:.6f}, Shape: {debug_info['loss_shape']}, Requires Grad: {debug_info['loss_requires_grad']}")
      self.logger.info(f"Setup Correct for Gradients: {debug_info['setup_correct']}")
      self.logger.info(f"==========================================")
      
      return debug_info
      
  def debug_gradients_after_backward(self, loss, feature_maps_adv, feature_maps_rand, iteration):
      """
      Debug gradient flow AFTER backward pass - now check actual gradients
      """
      debug_info = {}
      
      # 1. Check feature map gradients (after backward pass)
      if feature_maps_adv is not None and feature_maps_adv.grad is not None:
          debug_info['feature_maps_adv_grad_norm'] = torch.norm(feature_maps_adv.grad).item()
          debug_info['feature_maps_adv_grad_mean'] = feature_maps_adv.grad.mean().item()
          debug_info['feature_maps_adv_grad_std'] = feature_maps_adv.grad.std().item()
      else:
          debug_info['feature_maps_adv_grad_norm'] = 0.0
          debug_info['feature_maps_adv_grad_mean'] = 0.0
          debug_info['feature_maps_adv_grad_std'] = 0.0
      
      # 2. Check patch gradients (after backward pass)
      if self.adv_patch.grad is not None:
          debug_info['patch_grad_norm'] = torch.norm(self.adv_patch.grad).item()
          debug_info['patch_grad_mean'] = self.adv_patch.grad.mean().item()
          debug_info['patch_grad_std'] = self.adv_patch.grad.std().item()
      else:
          debug_info['patch_grad_norm'] = 0.0
          debug_info['patch_grad_mean'] = 0.0
          debug_info['patch_grad_std'] = 0.0
      
      # 3. Check if gradients are meaningful
      debug_info['gradients_meaningful'] = (
          debug_info['feature_maps_adv_grad_norm'] > 1e-8 and 
          debug_info['patch_grad_norm'] > 1e-8
      )
      
      # Update the stored debug info with actual gradient values
      if len(self.gradient_debug_info['loss_wrt_feature_maps']) > 0:
          self.gradient_debug_info['loss_wrt_feature_maps'][-1] = debug_info['feature_maps_adv_grad_norm']
          self.gradient_debug_info['feature_maps_wrt_patch'][-1] = debug_info['patch_grad_norm']
          self.gradient_debug_info['patch_gradients'][-1] = debug_info['patch_grad_norm']
      
      # Log detailed debug information
      self.logger.info(f"=== AFTER BACKWARD PASS - ITERATION {iteration} ===")
      self.logger.info(f"Feature Maps Grad - Norm: {debug_info['feature_maps_adv_grad_norm']:.6f}, Mean: {debug_info['feature_maps_adv_grad_mean']:.6f}, Std: {debug_info['feature_maps_adv_grad_std']:.6f}")
      self.logger.info(f"Patch Grad - Norm: {debug_info['patch_grad_norm']:.6f}, Mean: {debug_info['patch_grad_mean']:.6f}, Std: {debug_info['patch_grad_std']:.6f}")
      
      if debug_info['gradients_meaningful']:
          self.logger.info("✓ Gradients are meaningful (> 1e-8)")
      else:
          self.logger.warning("✗ Gradients are too small (< 1e-8) - this explains no updates!")
          
          if debug_info['feature_maps_adv_grad_norm'] < 1e-8:
              self.logger.warning("  - Feature map gradients are too small")
          if debug_info['patch_grad_norm'] < 1e-8:
              self.logger.warning("  - Patch gradients are too small")
      
      self.logger.info("================================================")
      
      return debug_info
  
  def visualize_gradients(self, save_path=None):
      """
      Create visualizations to debug gradient flow
      """
      if len(self.gradient_debug_info['loss_values']) == 0:
          self.logger.warning("No gradient debug information available")
          return
      
      fig, axes = plt.subplots(2, 3, figsize=(18, 12))
      fig.suptitle('Gradient Flow Debug Analysis', fontsize=16)
      
      iterations = range(len(self.gradient_debug_info['loss_values']))
      
      # Plot 1: Loss progression
      axes[0, 0].plot(iterations, self.gradient_debug_info['loss_values'])
      axes[0, 0].set_title('Loss Progression')
      axes[0, 0].set_xlabel('Iteration')
      axes[0, 0].set_ylabel('Loss Value')
      axes[0, 0].grid(True, alpha=0.3)
      
      # Plot 2: Feature map gradients
      axes[0, 1].plot(iterations, self.gradient_debug_info['loss_wrt_feature_maps'])
      axes[0, 1].set_title('Loss w.r.t Feature Maps Gradients')
      axes[0, 1].set_xlabel('Iteration')
      axes[0, 1].set_ylabel('Gradient Norm')
      axes[0, 1].grid(True, alpha=0.3)
      
      # Plot 3: Patch gradients
      axes[0, 2].plot(iterations, self.gradient_debug_info['feature_maps_wrt_patch'])
      axes[0, 2].set_title('Feature Maps w.r.t Patch Gradients')
      axes[0, 2].set_xlabel('Iteration')
      axes[0, 2].set_ylabel('Gradient Norm')
      axes[0, 2].grid(True, alpha=0.3)
      
      # Plot 4: Feature map norms
      axes[1, 0].plot(iterations, self.gradient_debug_info['feature_map_norms'])
      axes[1, 0].set_title('Feature Map Norms')
      axes[1, 0].set_xlabel('Iteration')
      axes[1, 0].set_ylabel('Norm Value')
      axes[1, 0].grid(True, alpha=0.3)
      
      # Plot 5: Gradient flow health
      healthy_flow = [1 if norm > 1e-8 else 0 for norm in self.gradient_debug_info['patch_gradients']]
      axes[1, 1].plot(iterations, healthy_flow, 'o-')
      axes[1, 1].set_title('Gradient Flow Health (1=Healthy, 0=Problem)')
      axes[1, 1].set_xlabel('Iteration')
      axes[1, 1].set_ylabel('Flow Status')
      axes[1, 1].grid(True, alpha=0.3)
      axes[1, 1].set_ylim(-0.1, 1.1)
      
      # Plot 6: Combined gradient analysis
      axes[1, 2].plot(iterations, self.gradient_debug_info['loss_wrt_feature_maps'], label='Loss→Feature Maps', alpha=0.7)
      axes[1, 2].plot(iterations, self.gradient_debug_info['feature_maps_wrt_patch'], label='Feature Maps→Patch', alpha=0.7)
      axes[1, 2].set_title('Combined Gradient Flow Analysis')
      axes[1, 2].set_xlabel('Iteration')
      axes[1, 2].set_ylabel('Gradient Norm')
      axes[1, 2].legend()
      axes[1, 2].grid(True, alpha=0.3)
      
      plt.tight_layout()
      
      if save_path:
          plt.savefig(save_path, dpi=300, bbox_inches='tight')
          self.logger.info(f"Gradient debug plots saved to {save_path}")
      
      plt.show()
      
      # Print summary statistics
      self.logger.info("=== GRADIENT DEBUG SUMMARY ===")
      self.logger.info(f"Total iterations analyzed: {len(iterations)}")
      self.logger.info(f"Healthy gradient flow iterations: {sum(healthy_flow)}")
      self.logger.info(f"Problematic iterations: {len(iterations) - sum(healthy_flow)}")
      
      if len(iterations) > 0:
          avg_loss = np.mean(self.gradient_debug_info['loss_values'])
          avg_feature_grad = np.mean(self.gradient_debug_info['loss_wrt_feature_maps'])
          avg_patch_grad = np.mean(self.gradient_debug_info['feature_maps_wrt_patch'])
          
          self.logger.info(f"Average loss: {avg_loss:.6f}")
          self.logger.info(f"Average feature map gradient norm: {avg_feature_grad:.6f}")
          self.logger.info(f"Average patch gradient norm: {avg_patch_grad:.6f}")
          
          if avg_patch_grad < 1e-8:
              self.logger.warning("CRITICAL: Patch gradients are extremely small - this explains why no updates are happening!")
          elif avg_feature_grad < 1e-8:
              self.logger.warning("CRITICAL: Feature map gradients are extremely small - loss is not flowing to feature maps!")
          else:
              self.logger.info("Gradient flow appears normal")
      
      self.logger.info("==============================")
  
  def train(self):
    epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters
    self.feature_maps_adv = None
    self.feature_maps_rand = None
    start_time = time.time()
    self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, 1000))
    IoU = []
    
    # Register hooks for gradient debugging
    self.register_forward_hook1()
    self.register_forward_hook2()
    
    for ep in range(self.start_epoch, self.end_epoch):
      self.current_epoch = ep
      self.metric.reset()
      total_loss = 0
      samplecnt = 0
      momentum = torch.zeros_like(self.adv_patch, device=self.device)
      for i_iter, batch in enumerate(self.train_dataloader, 0):
        if i_iter<1000:
          self.current_iteration += 1
          samplecnt += batch[0].shape[0]
          image, true_label,_, _, _, idx = batch
          self.coords[idx] = self.coords[idx].to(self.device)
          image, true_label = image.to(self.device), true_label.to(self.device)
          
          patched_image_rand, patched_label_rand = image,true_label
          if(len(self.coords[idx])!=0):
              x1, y1, x2, y2 = self.coords[idx]
              #self.logger.info(f"(x1,y1,x2,y2):{x1,y1,x2,y2}, Idx:{idx}, Iter: {i_iter}")
              image[:,:,y1:y2,x1:x2] = self.adv_patch
              patched_image_adv = image
              patched_label_adv = true_label
              
              # Forward pass through the model (and interpolation if needed)
              output1 = self.model1.predict(patched_image_adv,patched_label_adv.shape)
              output2 = self.model2.predict(patched_image_rand,patched_label_rand.shape)
              
              # Check if feature maps are captured
              if self.feature_maps_adv is None:
                  self.logger.warning(f"Iteration {i_iter}: Feature maps not captured! Check if hook is working.")
                  continue
              
              # Compute adaptive loss
              loss,cos_sim = self.criterion.compute_cos_warmup_loss(self.feature_maps_adv, self.feature_maps_rand, output1, patched_label_adv)
              total_loss += loss.item()
              
              # Debug gradients before backward pass
              debug_info = self.debug_gradients_before_backward(loss, self.feature_maps_adv, self.feature_maps_rand, i_iter)
              
              #break
    
              ## metrics
              self.metric.update(output1, patched_label_adv)
              pixAcc, mIoU = self.metric.get()
    
              # Backpropagation
              self.model1.model.zero_grad()
              self.model2.model.zero_grad()
              if self.adv_patch.grad is not None:
                self.adv_patch.grad.zero_()
              
              # Backward pass
              loss.backward()
              
              # Debug gradients after backward pass
              debug_info_after = self.debug_gradients_after_backward(loss, self.feature_maps_adv, self.feature_maps_rand, i_iter)
              
              with torch.no_grad():
                  # Check if gradients are meaningful
                  if self.adv_patch.grad is not None and torch.norm(self.adv_patch.grad) > 1e-8:
                      old_patch_norm = torch.norm(self.adv_patch).item()
                      self.adv_patch += 0.01 * self.adv_patch.grad.data.sign()
                      new_patch_norm = torch.norm(self.adv_patch).item()
                      patch_update_magnitude = abs(new_patch_norm - old_patch_norm)
                      
                      self.gradient_debug_info['patch_updates'].append(patch_update_magnitude)
                      self.logger.info(f"Iteration {i_iter}: Patch updated successfully, update magnitude: {patch_update_magnitude:.6f}")
                  else:
                      self.gradient_debug_info['patch_updates'].append(0.0)
                      self.logger.warning(f"Iteration {i_iter}: No meaningful gradients for patch update!")
                  
                  self.adv_patch.clamp_(-2.1, 2.6)  # Keep pixel values in valid range
    
              ## ETA
              eta_seconds = ((time.time() - start_time) / self.current_iteration) * (1000*epochs - self.current_iteration)
              eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    
              if i_iter % self.log_per_iters == 0:
                self.logger.info(
                  "Epochs: {:d}/{:d} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                      self.current_epoch, self.end_epoch,
                      samplecnt, 1000,
                      #self.optimizer.param_groups[0]['lr'],
                      self.epsilon,
                      loss.item(),
                      mIoU,
                      str(datetime.timedelta(seconds=int(time.time() - start_time))),
                      eta_string))
        else:
          break
          
      # Generate gradient debug visualization at the end of each epoch
      if ep % 5 == 0:  # Every 5 epochs
          self.visualize_gradients(save_path=f"gradient_debug_epoch_{ep}.png")

      average_pixAcc, average_mIoU = self.metric.get()
      average_loss = total_loss/1000
      self.logger.info('-------------------------------------------------------------------------------------------------')
      self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
        self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))
      IoU.append(self.metric.get(full=True))
      
      # Save patch, IoU data, and gradient debug information
      safety = self.adv_patch.clone().detach(), np.array(IoU), self.gradient_debug_info
      pickle.dump( safety, open(self.config.experiment.log_patch_address+self.config.model.name+"_cos_loss_sidewalk_gradient_debug"+".p", "wb" ) )
      
      #self.test() ## Doing 1 iteration of testing
      self.logger.info('-------------------------------------------------------------------------------------------------')
      #self.model.train() ## Setting the model back to train mode
      # if self.lr_scheduler:
      #     self.scheduler.step()

    # Final gradient debug visualization
    self.visualize_gradients(save_path="final_gradient_debug_analysis.png")
    
    return self.adv_patch.detach(),np.array(IoU)  # Return adversarial patch and IoUs over epochs
