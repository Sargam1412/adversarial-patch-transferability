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
  def __init__(self,config,main_logger,model_name,target_ft, patch_coords, ft_map_coords):#,patch1,patch2,patch3,patch4
      self.config = config
      self.start_epoch = 0
      self.end_epoch = 30
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

      # self.model2 = Models(self.config)
      # self.model2.get()

      ## loss
      self.criterion = PatchLoss(self.config)
      ## optimizer
      # Initialize adversarial patch (random noise)
      self.adv_patch = torch.rand((3, 200, 200), 
                               requires_grad=True, 
                               device=self.device)
      # self.adv_patch1 = patch1.clone().detach().to(self.device)
      # self.adv_patch1.requires_grad = True#torch.rand((3, 100, 100), 
                              # requires_grad=True, 
                              # device=self.device)
      # self.adv_patch1 = torch.rand((3, 100, 100), 
      #                         requires_grad=True, 
      #                         device=self.device)
      # self.adv_patch2 = patch2.clone().detach().to(self.device)
      # self.adv_patch2.requires_grad = True#torch.rand((3, 100, 100), 
                              # requires_grad=True, 
                              # device=self.device)
      # self.adv_patch2 = torch.rand((3, 100, 100), 
      #                         requires_grad=True, 
      #                         device=self.device)
      # self.adv_patch3 = patch3.clone().detach().to(self.device)
      # self.adv_patch3.requires_grad = True#torch.rand((3, 100, 100), 
                              # requires_grad=True, 
                              # device=self.device)
      # self.adv_patch3 = torch.rand((3, 100, 100), 
      #                         requires_grad=True, 
      #                         device=self.device)
      # self.adv_patch4 = patch4.clone().detach().to(self.device)
      # self.adv_patch4.requires_grad = True#torch.rand((3, 100, 100), 
                              # requires_grad=True, 
                              # device=self.device)
      # self.adv_patch4 = torch.rand((3, 100, 100), 
      #                         requires_grad=True, 
      #                         device=self.device)
      
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

      # self.H1 = torch.load('/kaggle/working/logs/H1_pidnet_s.pt', map_location = self.device)
      # self.H2 = torch.load('/kaggle/working/logs/H2_pidnet_s.pt', map_location = self.device)
      # self.H4 = torch.load('/kaggle/working/logs/H4_pidnet_s.pt', map_location = self.device)
      # self.H1 /= torch.norm(self.H1, p=2, dim=(2,3), keepdim=True) + 1e-8 
      # self.H2 /= torch.norm(self.H2, p=2, dim=(2,3), keepdim=True) + 1e-8
      # self.H4 /= torch.norm(self.H4, p=2, dim=(2,3), keepdim=True) + 1e-8

      # Register hook
      if 'pidnet_s' in self.model_name:
        self.layer1_name = 'layer1.0.bn2' 
        self.feature_map1_shape = [32, 256, 256]
        self.layer2_name = 'layer3.2.bn2' 
        self.feature_map2_shape = [128,64,64]
        self.layer3_name = 'layer3_.1.bn2' 
        self.feature_map3_shape = [64, 128, 128]
        self.layer4_name = 'layer5.1.relu' 
        self.feature_map4_shape = [256, 16, 16]
      elif 'pidnet_m' in self.model_name:
        self.layer1_name = 'layer1.0.bn2' 
        self.feature_map1_shape = [64, 256, 256]
        self.layer2_name = 'layer3.2.bn2' 
        self.feature_map2_shape = [256,64,64]
        self.layer3_name = 'layer3_.1.bn2' 
        self.feature_map3_shape = [128, 128, 128]
        self.layer4_name = 'layer5.1.relu' 
        self.feature_map4_shape = [512, 16, 16]
      elif 'pidnet_l' in self.model_name:
        self.layer1_name = 'layer1.0.bn2' 
        self.feature_map1_shape = [64, 256, 256]
        self.layer2_name = 'layer3.2.bn2' 
        self.feature_map2_shape = [256,64,64]
        self.layer3_name = 'layer3_.1.bn2' 
        self.feature_map3_shape = [128, 128, 128]
        self.layer4_name = 'layer5.1.relu' 
        self.feature_map4_shape = [512, 16, 16]
      elif 'bisenet' in self.model_name: 
        self.layer_name = 'segment.S1S2.fuse'
        self.feature_map_shape=[16, 256, 256]
        self.layer_name = 'segment.S3.1.relu'
        self.feature_map_shape=[32,128,128]
        self.layer_name = 'segment.S4.1.relu'
        self.feature_map_shape=[64, 64, 64]
        self.layer_name = 'segment.S5_4.3.relu'
        self.feature_map_shape=[128, 32, 32]
      else:
        self.layer_name = 'pretrained.layer2.3.relu'
        self.feature_map_shape=[512,32,32]

      self.feature_maps_adv1 = None
      self.feature_maps_adv2 = None
      self.feature_maps_adv3 = None
      self.feature_maps_adv4 = None
      self.feature_maps_rand1 = None
      self.feature_maps_rand2 = None
      self.feature_maps_rand3 = None
      self.feature_maps_rand4 = None
      self.target_ft = target_ft.to(self.device)
      self.patch_coords = patch_coords
      self.ft_map_coords = ft_map_coords
    
  # Hook to store feature map
  def hook1(self, module, input, output):
      self.feature_maps_adv1 = output
      output.retain_grad()

  def hook12(self, module, input, output):
      self.feature_maps_adv2 = output
      output.retain_grad()

  def hook13(self, module, input, output):
      self.feature_maps_adv3 = output
      output.retain_grad()

  def hook14(self, module, input, output):
      self.feature_maps_adv4 = output
      output.retain_grad()

  # def hook2(self, module, input, output):
  #     self.feature_maps_rand1 = output
  #     output.retain_grad()

  # def hook22(self, module, input, output):
  #     self.feature_maps_rand2 = output
  #     output.retain_grad()

  # def hook23(self, module, input, output):
  #     self.feature_maps_rand3 = output
  #     output.retain_grad()

  # def hook24(self, module, input, output):
  #     self.feature_maps_rand4 = output
  #     output.retain_grad()

  def register_forward_hook1(self):
    for name, module in self.model1.model.named_modules():
        # if name == self.layer1_name:
          # module.register_forward_hook(self.hook1)
        if name == self.layer2_name:
          module.register_forward_hook(self.hook12)
        # if name == self.layer3_name:
        #   module.register_forward_hook(self.hook13)
        # if name == self.layer4_name:
        #   module.register_forward_hook(self.hook14)

  # def register_forward_hook2(self):
  #   for name, module in self.model2.model.named_modules():
  #       if name == self.layer1_name:
  #         module.register_forward_hook(self.hook2)
  #       if name == self.layer2_name:
  #         module.register_forward_hook(self.hook22)
  #       # if name == self.layer3_name:
  #       #   module.register_forward_hook(self.hook23)
  #       if name == self.layer4_name:
  #         module.register_forward_hook(self.hook24)
          
  def train(self):#, H1, H2, H3, H4
    epochs, iters_per_epoch, max_iters = self.epochs, self.iters_per_epoch, self.max_iters
    # self.feature_maps_adv1 = None
    # self.feature_maps_rand1 = None
    self.feature_maps_adv2 = None
    # self.feature_maps_rand2 = None
    # self.feature_maps_adv3 = None
    # self.feature_maps_rand3 = None
    # self.feature_maps_adv4 = None
    # self.feature_maps_rand4 = None
    # H1 /= torch.norm(H1, p=2, dim=(2,3), keepdim=True) + 1e-8 
    # H2 /= torch.norm(H2, p=2, dim=(2,3), keepdim=True) + 1e-8
    # H3 /= torch.norm(H3, p=2, dim=(2,3), keepdim=True) + 1e-8
    # H4 /= torch.norm(H4, p=2, dim=(2,3), keepdim=True) + 1e-8
    # self.logger.info(f'H1.shape: {H1.shape}')
    start_time = time.time()
    self.logger.info('Start training, Total Epochs: {:d} = Iterations per epoch {:d}'.format(epochs, 1000))
    IoU = []
    for ep in range(self.start_epoch, self.end_epoch):
      samplecnt = 0
      self.current_epoch = ep
      self.metric.reset()
      momentum1 = torch.tensor(0, dtype=torch.float32).to(self.device)
      # momentum2 = torch.tensor(0, dtype=torch.float32).to(self.device)
      # momentum3 = torch.tensor(0, dtype=torch.float32).to(self.device)
      # momentum4 = torch.tensor(0, dtype=torch.float32).to(self.device)
      for i_iter, batch in enumerate(self.train_dataloader, 0):
          if i_iter>=1000:
            break
          else:
            self.current_iteration += 1
            samplecnt += batch[0].shape[0]
            image, true_label,_, _, _, idx = batch
            self.patch_coords[idx] = self.patch_coords[idx].to(self.device)
            self.ft_map_coords[idx] = self.ft_map_coords[idx].to(self.device)
            image, true_label = image.to(self.device), true_label.to(self.device)
            
            # Randomly place patch in image and label(put ignore index)
            if(len(self.patch_coords[idx])!=0):
              x1, y1, x2, y2 = self.patch_coords[idx]
              #self.logger.info(f"(x1,y1,x2,y2):{x1,y1,x2,y2}, Idx:{idx}, Iter: {i_iter}")
              image[:,:,y1:y2,x1:x2] = self.adv_patch
              patched_image_adv = image
              true_label[:,y1:y2,x1:x2] = 10
              patched_label_adv = true_label
              #patched_image_adv, patched_label_adv = self.apply_patch_grad(image,true_label,self.adv_patch)
              #patched_image_rand, patched_label_rand = self.apply_patch_rand(image,true_label)
              # fig = plt.figure()
              # ax = fig.add_subplot(1,2,1)
              # ax.imshow(patched_image[0].permute(1,2,0).cpu().detach().numpy())
              # ax = fig.add_subplot(1,2,2)
              # ax.imshow(patched_label[0].cpu().detach().numpy())
              # plt.show()
    
              # Forward pass through the model (and interpolation if needed)
              output1 = self.model1.predict(patched_image_adv,patched_label_adv.shape)
              #output2 = self.model2.predict(patched_image_rand,patched_label_rand.shape)
              #F = torch.zeros(( self.feature_map_shape[1], self.feature_map_shape[2]), device=self.device)
              #self.logger.info(f"feature_map shape:{self.feature_maps_adv2.shape}")
              fx1, fy1, fx2, fy2 = self.ft_map_coords[idx]
              adv_ft = self.feature_maps_adv2.squeeze(0)[:,fy1:fy2,fx1:fx2]
              # F1 = ((self.feature_maps_adv1[i]-self.feature_maps_rand1[i])*H1[idx[i]]) + (H1[idx[i]])**2
              F2 = adv_ft-self.target_ft
              # F3 = ((self.feature_maps_adv3[i]-self.feature_maps_rand3[i])*self.H3[idx[i]]) + (self.H3[idx[i]])**2
              # F4 = ((self.feature_maps_adv4[i]-self.feature_maps_rand4[i])*self.H4[idx[i]]) + (self.H4[idx[i]])**2
              #plt.imshow(output.argmax(dim =1)[0].cpu().detach().numpy())
              #plt.show()
              #break
    
              # Compute adaptive loss
              # loss1 = self.criterion.compute_trainloss(F1)
              loss2 = self.criterion.compute_target_trainloss(F2, output1, true_label)
              # loss3 = self.criterion.compute_trainloss(F3)
              # loss4 = self.criterion.compute_trainloss(F4)
              #loss = self.criterion.compute_loss_direct(output, patched_label)
              # total_loss += (loss1.item() + loss2.item() + loss4.item())# + loss3.item()
              #break
    
              ## metrics
              self.metric.update(output1, patched_label_adv)
              pixAcc, mIoU = self.metric.get()
    
              # Backpropagation
              self.model1.model.zero_grad()
              if self.adv_patch.grad is not None:
                self.adv_patch.grad.zero_()
              # self.model2.model.zero_grad()
              # if self.adv_patch1.grad is not None:
              #   self.adv_patch1.grad.zero_()
              # if self.adv_patch2.grad is not None:
              #   self.adv_patch2.grad.zero_()
              # if self.adv_patch3.grad is not None:
              #   self.adv_patch3.grad.zero_()
              # if self.adv_patch4.grad is not None:
              #   self.adv_patch4.grad.zero_()
              # loss1.backward()
              # loss2.backward()
              # loss3.backward()
              # loss4.backward()
              # grad1 = torch.autograd.grad(loss1, self.adv_patch1, retain_graph=True)[0]
              grad = torch.autograd.grad(loss2, self.adv_patch, retain_graph=True)[0]
              # grad3 = torch.autograd.grad(loss2, self.adv_patch3, retain_graph=True)[0]
              # grad4 = torch.autograd.grad(loss4, self.adv_patch4, retain_graph=True)[0]
              with torch.no_grad():
                  #self.patch += self.epsilon * self.patch.grad.sign()  # Update patch using FGSM-style ascent
                  #norm_grad1 = grad1/ (torch.norm(grad1) + 1e-8)
                  momentum1 = (0.9*momentum1) + (grad/ (torch.norm(grad) + 1e-8))
                  self.adv_patch -= self.epsilon * momentum1.sign()
                  #self.patch += self.epsilon * self.patch.grad.data.sign()
                  self.adv_patch.clamp_(0, 1)  # Keep pixel values in valid range
                  #norm_grad2 = grad2/ (torch.norm(grad2) + 1e-8)
                  # momentum2 = (0.9*momentum2) + (grad2/ (torch.norm(grad2) + 1e-8))
                  # self.adv_patch2 += self.epsilon * momentum2.sign()
                  # self.patch += self.epsilon * self.patch.grad.data.sign()
                  # self.adv_patch2.clamp_(0, 1)  # Keep pixel values in valid range
                  # #norm_grad3 = grad3/ (torch.norm(grad3) + 1e-8)
                  # momentum3 = (0.9*momentum3) + (grad3/ (torch.norm(grad3) + 1e-8))
                  # self.adv_patch3 += self.epsilon * momentum3.sign()
                  # #self.patch += self.epsilon * self.patch.grad.data.sign()
                  # self.adv_patch3.clamp_(0, 1)  # Keep pixel values in valid range
                  # #norm_grad4 = grad4/ (torch.norm(grad4) + 1e-8)
                  # momentum4 = (0.9*momentum4) + (grad3/ (torch.norm(grad3) + 1e-8))
                  # self.adv_patch4 += self.epsilon * momentum4.sign()
                  # #self.patch += self.epsilon * self.patch.grad.data.sign()
                  # self.adv_patch4.clamp_(0, 1)  # Keep pixel values in valid range
    
              ## ETA
              eta_seconds = ((time.time() - start_time) / self.current_iteration) * (1000*epochs - self.current_iteration)
              eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    
              if ep % self.log_per_iters == 0:
                self.logger.info(
                  "Epochs: {:d}/{:d} || Samples: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                      self.current_epoch, self.end_epoch,
                      samplecnt, 1000,
                      #self.optimizer.param_groups[0]['lr'],
                      self.epsilon,
                      loss2.item(),
                      mIoU,
                      str(datetime.timedelta(seconds=int(time.time() - start_time))),
                      eta_string))
                      

      average_pixAcc, average_mIoU = self.metric.get()
      average_loss = loss2.item()
      self.logger.info('-------------------------------------------------------------------------------------------------')
      self.logger.info("Epochs: {:d}/{:d}, Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(
        self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))
      safety1 = self.adv_patch.clone()
      pickle.dump( safety1.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_targeted_sky_multiimg"+".p", "wb" ) )
      # safety2 = self.adv_patch2.clone()
      # pickle.dump( safety2.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss_amap2"+".p", "wb" ) )
      # safety3 = self.adv_patch3.clone()
      # pickle.dump( safety3.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss_amap3"+".p", "wb" ) )
      # safety4 = self.adv_patch4.clone()
      # pickle.dump( safety4.detach(), open(self.config.experiment.log_patch_address+self.config.model.name+"_bbfa_modifiedloss_amap4"+".p", "wb" ) )
      
      #self.test() ## Doing 1 iteration of testing
      self.logger.info('-------------------------------------------------------------------------------------------------')
      #self.model.train() ## Setting the model back to train mode
      # if self.lr_scheduler:
      #     self.scheduler.step()

      IoU.append(self.metric.get(full=True))

    return self.adv_patch.detach(),np.array(IoU)  # Return adversarial patch and IoUs over epochs

    
