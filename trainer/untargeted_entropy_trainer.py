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
      self.end_epoch = 3000
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
      self.epsilon = 0.001
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
        low, high = -2.1, 2.6
        self.adv_patch = (low + (high - low) * torch.rand((3, 200, 200), 
                                                  device=self.device,
                                                  requires_grad=True)).detach()
        self.adv_patch.requires_grad_()
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
        self.layer1_name = 'layer5.1.relu' 
        self.feature_map1_shape = [256,16,32]
        self.layer2_name = 'layer5_d.0.downsample.1' 
        self.feature_map2_shape = [128, 128, 256]
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

      self.feature_maps1_adv = None
      self.feature_maps1_rand = None
      self.feature_maps2_adv = None
      self.feature_maps2_rand = None
      self.patch_ce_ref = None
    
  # Hook to store feature map
  def hook11(self, module, input, output):
      self.feature_maps1_adv = output
      output.retain_grad()

  def hook12(self, module, input, output):
      self.feature_maps2_adv = output
      output.retain_grad()

  def hook21(self, module, input, output):
      self.feature_maps1_rand = output
      output.retain_grad()

  def hook22(self, module, input, output):
      self.feature_maps2_rand = output
      output.retain_grad()

  def register_forward_hook11(self):
    for name, module in self.model1.model.named_modules():
        if name == self.layer1_name:
          module.register_forward_hook(self.hook11)

  def register_forward_hook12(self):
    for name, module in self.model1.model.named_modules():
        if name == self.layer2_name:
          module.register_forward_hook(self.hook12)

  def register_forward_hook21(self):
    for name, module in self.model2.model.named_modules():
        if name == self.layer1_name:
          module.register_forward_hook(self.hook21)

  def register_forward_hook22(self):
    for name, module in self.model2.model.named_modules():
        if name == self.layer2_name:
          module.register_forward_hook(self.hook22)
  
  def train(self):
    epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters
    self.feature_maps1_adv = None
    self.feature_maps1_rand = None
    self.feature_maps2_adv = None
    self.feature_maps2_rand = None
    start_time = time.time()
    self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, 1000))
    IoU = []
    for ep in range(self.start_epoch, self.end_epoch):
      self.current_epoch = ep
      self.metric.reset()
      total_loss = 0
      samplecnt = 0
      momentum = torch.zeros_like(self.adv_patch, device=self.device)
      for i_iter, batch in enumerate(self.train_dataloader, 0):
        if i_iter<1:
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
              # Compute adaptive loss
              if (ep<self.end_epoch-50):
                loss = self.criterion.compute_ce_loss(output1, patched_label_adv)
              else:
                if(ep%5==0):
                  loss1=self.criterion.compute_D_loss(self.feature_maps2_adv)
                else:
                  loss1 = self.criterion.compute_hsic_loss_spatial(self.feature_maps1_adv, self.feature_maps1_rand, sigma=1.0)
                cos_loss = self.criterion.compute_cos_loss(self.adv_patch, self.patch_ce_ref)
                loss = loss1 + 0.2*cos_loss
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
              loss.backward()
              # grad = torch.autograd.grad(loss, self.adv_patch, retain_graph=True)[0]
              with torch.no_grad():
                  #norm_grad1 = grad1/ (torch.norm(grad1) + 1e-8)
                  # momentum = (0.9*momentum) + (grad/ (torch.norm(grad) + 1e-8))
                  # self.adv_patch += 0.01 * momentum / (momentum.norm() + 1e-8)
                  # self.logger.info(grad)
                  self.adv_patch -= self.epsilon * self.adv_patch.grad.data.sign()
                  # Compute gradient
                  # grad = self.adv_patch.grad.data
                  
                  # Normalize the gradient (L2 norm across all pixels)
                  # grad_norm = grad / (torch.norm(grad) + 1e-8)
                  
                  # Update in the normalized gradient direction
                  # self.adv_patch += 0.01 * (grad / (torch.norm(grad) + 1e-8))
                  self.adv_patch.clamp_(-2.1, 2.6)  # Keep pixel values in valid range

              if ep == self.end_epoch-51:  
                self.patch_ce_ref = self.adv_patch.clone().detach()  # save reference

              ## ETA
              eta_seconds = ((time.time() - start_time) / self.epochs) * (self.epochs - ep)
              eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    
              if i_iter % self.log_per_iters == 0:
                self.logger.info(
                  "Epochs: {:d}/{:d} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                      self.current_epoch, self.end_epoch,
                      samplecnt, 1,
                      #self.optimizer.param_groups[0]['lr'],
                      self.epsilon,
                      loss.item(),
                      mIoU,
                      str(datetime.timedelta(seconds=int(time.time() - start_time))),
                      eta_string))
        else:
          break
          

      average_pixAcc, average_mIoU = self.metric.get()
      average_loss = total_loss
      self.logger.info('-------------------------------------------------------------------------------------------------')
      self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
        self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))
      IoU.append(self.metric.get(full=True))
      safety = self.adv_patch.clone().detach(), np.array(IoU)
      pickle.dump( safety, open(self.config.experiment.log_patch_address+self.config.model.name+"_2stage_hsic_loss_sidewalk_pbranch_sign"+".p", "wb" ) )
      
      #self.test() ## Doing 1 iteration of testing
      self.logger.info('-------------------------------------------------------------------------------------------------')
      #self.model.train() ## Setting the model back to train mode
      # if self.lr_scheduler:
      #     self.scheduler.step()

    return self.adv_patch.detach(),np.array(IoU)  # Return adversarial patch and IoUs over epochs

    
