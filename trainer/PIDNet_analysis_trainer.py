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

class PatchTrainer():
  def __init__(self,config,main_logger,model_name, coords, resume=False, patch=None):#,patch1,patch2,patch3,patch4
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
      ## optimizer
      # Initialize adversarial patch (random noise)
      if (resume==False):
        self.adv_patch = torch.rand((3, 200, 200), 
                                 requires_grad=True, 
                                 device=self.device)
      else:
        self.adv_patch = patch.clone().detach().to(self.device)
        self.adv_patch.requires_grad = True
      
      # # Define optimizer
      # self.optimizer = torch.optim.SGD(params = [self.patch],
      #                         lr=self.lr,
      #                         momentum=config.optimizer.momentum,
      #                         weight_decay=config.optimizer.weight_decay,
      #                         nesterov=config.optimizer.nesterov,
      # )
      # if self.lr_scheduler:
      #   self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_scheduler_gamma)


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
      
      # Store gradients for each branch
      self.branch_gradients = {'P': [], 'I': [], 'D': []}
      self.branch_losses = {'P': [], 'I': [], 'D': []}
    
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
  
  def compute_branch_gradients(self, outputs, labels):
      """
      Compute gradients of cross-entropy loss with respect to the patch for each PIDNet branch
      """
      if 'pidnet' not in self.model_name:
          return None, None
      
      try:
          # PIDNet returns [x_extra_p, x_, x_extra_d] when augment=True
          # P branch: outputs[0], I branch: outputs[1], D branch: outputs[2]
          branch_outputs = {
              'P': outputs[0],  # Precise branch
              'I': outputs[1],  # Integral branch  
              'D': outputs[2]   # Detail branch
          }
          
          branch_losses = {}
          branch_gradients = {}
          
          # Debug info
          self.logger.info(f"Labels shape: {labels.shape}")
          for branch_name, branch_output in branch_outputs.items():
              self.logger.info(f"{branch_name} branch output shape: {branch_output.shape}")
          
          for branch_name, branch_output in branch_outputs.items():
              # Interpolate branch output to match label dimensions
              # labels shape: [B, H, W], branch_output shape: [B, C, H', W']
              target_size = (labels.shape[-2], labels.shape[-1])  # (H, W)
              
              # Interpolate the branch output to match target spatial dimensions
              interpolated_output = torch.nn.functional.interpolate(
                  branch_output, 
                  size=target_size, 
                  mode='bilinear', 
                  align_corners=True
              )
              
              self.logger.info(f"{branch_name} interpolated output shape: {interpolated_output.shape}")
              
              # Handle D branch differently - it outputs 1 channel but we need 2 for binary classification
              if branch_name == 'D':
                  # Convert 1-channel output to 2-channel binary logits
                  # Channel 0: background/ignore, Channel 1: foreground/valid pixels
                  zeros_channel = torch.zeros_like(interpolated_output)
                  binary_output = torch.cat([zeros_channel, interpolated_output], dim=1)
                  
                  # Convert multi-class labels to binary labels
                  # 0: ignore label, 1: any valid semantic class
                  binary_labels = (labels != self.config.train.ignore_label).long()
                  
                  self.logger.info(f"D branch binary output shape: {binary_output.shape}")
                  self.logger.info(f"D branch binary labels shape: {binary_labels.shape}")
                  self.logger.info(f"D branch binary labels unique values: {torch.unique(binary_labels)}")
                  
                  # Compute binary cross-entropy loss for D branch
                  ce_loss = self.criterion.compute_loss_direct(binary_output, binary_labels)
              else:
                  # P and I branches have 19 channels, use original labels
                  ce_loss = self.criterion.compute_loss_direct(interpolated_output, labels)
              
              branch_losses[branch_name] = ce_loss.item()
              
              # Compute gradient with respect to the patch
              if self.adv_patch.grad is not None:
                  self.adv_patch.grad.zero_()
              
              grad = torch.autograd.grad(ce_loss, self.adv_patch, retain_graph=True)[0]
              branch_gradients[branch_name] = grad.clone().detach()
          
          return branch_losses, branch_gradients
          
      except Exception as e:
          self.logger.error(f"Error in compute_branch_gradients: {e}")
          self.logger.error(f"Outputs type: {type(outputs)}, length: {len(outputs) if isinstance(outputs, (list, tuple)) else 'N/A'}")
          self.logger.error(f"Labels shape: {labels.shape}")
          if isinstance(outputs, (list, tuple)):
              for i, output in enumerate(outputs):
                  self.logger.error(f"Output {i} shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
          return None, None
  
  def log_branch_analysis(self, branch_losses, branch_gradients, iteration):
      """
      Log and analyze the importance of each branch based on gradients
      """
      if branch_losses is None or branch_gradients is None:
          return
      
      # Store for later analysis
      for branch_name in ['P', 'I', 'D']:
          self.branch_losses[branch_name].append(branch_losses[branch_name])
          self.branch_gradients[branch_name].append(branch_gradients[branch_name])
      
      # Compute gradient magnitudes for comparison
      grad_magnitudes = {}
      for branch_name in ['P', 'I', 'D']:
          grad_magnitudes[branch_name] = torch.norm(branch_gradients[branch_name]).item()
      
      # Log the analysis
      self.logger.info(f"Iteration {iteration} - Branch Analysis:")
      self.logger.info(f"  P Branch - Loss: {branch_losses['P']:.4f}, Grad Norm: {grad_magnitudes['P']:.4f}")
      self.logger.info(f"  I Branch - Loss: {branch_losses['I']:.4f}, Grad Norm: {grad_magnitudes['I']:.4f}")
      self.logger.info(f"  D Branch - Loss: {branch_losses['D']:.4f}, Grad Norm: {grad_magnitudes['D']:.4f}")
      
      # Determine most crucial branch based on gradient magnitude
      crucial_branch = max(grad_magnitudes, key=grad_magnitudes.get)
      self.logger.info(f"  Most crucial branch: {crucial_branch} (highest gradient magnitude)")
      
      return crucial_branch
  
  def train(self):
    epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters
    self.feature_maps_adv = None
    self.feature_maps_rand = None
    start_time = time.time()
    self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, 1000))
    IoU = []
    for ep in range(self.start_epoch, self.end_epoch):
      self.current_epoch = ep
      self.metric.reset()
      total_loss = 0
      samplecnt = 0
      momentum = torch.tensor(0, dtype=torch.float32).to(self.device)
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
              
              # For PIDNet, get the raw outputs before interpolation for branch analysis
              if 'pidnet' in self.model_name:
                  raw_outputs1 = self.model1.model(patched_image_adv)
                  raw_outputs2 = self.model2.model(patched_image_rand)
                  
                  # Compute branch-specific gradients
                  branch_losses, branch_gradients = self.compute_branch_gradients(raw_outputs1, patched_label_adv)
                  
                  # Log branch analysis only if we have valid data
                  if branch_losses is not None and branch_gradients is not None:
                      if i_iter % self.log_per_iters == 0:
                          crucial_branch = self.log_branch_analysis(branch_losses, branch_gradients, i_iter)
                  else:
                      if i_iter % self.log_per_iters == 0:
                          self.logger.warning(f"Iteration {i_iter}: Branch analysis failed, skipping logging")
              
              # Compute adaptive loss
              loss = self.criterion.compute_entropy_loss(self.feature_maps_adv, self.feature_maps_rand)
              total_loss += loss.item()
              #break
    
              ## metrics
              self.metric.update(output1, patched_label_adv)
              pixAcc, mIoU = self.metric.get()
    
              # Backpropagation
              self.model1.model.zero_grad()
              self.model2.model.zero_grad()
              if self.adv_patch.grad is not None:
                self.adv_patch.grad.zero_()
              grad = torch.autograd.grad(loss, self.adv_patch, retain_graph=True)[0]
              with torch.no_grad():
                  #norm_grad1 = grad1/ (torch.norm(grad1) + 1e-8)
                  momentum = (0.9*momentum) + (grad/ (torch.norm(grad) + 1e-8))
                  self.adv_patch -= self.epsilon * momentum.sign()
                  #self.patch += self.epsilon * self.patch.grad.data.sign()
                  # self.adv_patch1.clamp_(0, 1)  # Keep pixel values in valid range
    
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
          

      average_pixAcc, average_mIoU = self.metric.get()
      average_loss = total_loss/1000
      self.logger.info('-------------------------------------------------------------------------------------------------')
      self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
        self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))
      IoU.append(self.metric.get(full=True))
      
      # Save patch and IoU data along with branch analysis
      # Check if we have valid branch analysis data
      if len(self.branch_losses['P']) > 0 and len(self.branch_gradients['P']) > 0:
          safety = self.adv_patch.clone().detach(), np.array(IoU), self.branch_losses, self.branch_gradients
          pickle.dump( safety, open(self.config.experiment.log_patch_address+self.config.model.name+"_cos_loss_sidewalk_branch_analysis"+".p", "wb" ) )
          self.logger.info("Saved training data with branch analysis")
      else:
          # Fallback to saving without branch analysis
          safety = self.adv_patch.clone().detach(), np.array(IoU)
          pickle.dump( safety, open(self.config.experiment.log_patch_address+self.config.model.name+"_cos_loss_sidewalk"+".p", "wb" ) )
          self.logger.warning("Branch analysis data not available, saved training data without branch analysis")
      
      #self.test() ## Doing 1 iteration of testing
      self.logger.info('-------------------------------------------------------------------------------------------------')
      #self.model.train() ## Setting the model back to train mode
      # if self.lr_scheduler:
      #     self.scheduler.step()

    return self.adv_patch.detach(),np.array(IoU)  # Return adversarial patch and IoUs over epochs

    
