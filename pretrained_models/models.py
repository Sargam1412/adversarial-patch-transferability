import sys
import torch
sys.path.append("/kaggle/working/adversarial-patch-transferability")
from pretrained_models.ICNet.icnet import ICNet
from pretrained_models.BisNetV1.model import BiSeNetV1
from pretrained_models.BisNetV2.model import BiSeNetV2
from pretrained_models.PIDNet.model import PIDNet, get_pred_model
from pretrained_models.PSPNet.pspnet import PSPNet
from pretrained_models.Deeplab.deeplabv3 import DeepLabV3
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch.nn.functional as F
import torch.nn as nn

class Models():
  def __init__(self,config):
    self.config = config
    self.name = config.model.name
    self.device = config.experiment.device
    self.model = None

  def get(self):
    if 'pidnet' in self.config.model.name:
      if '_s' in self.config.model.name:
        model = torch.load('/kaggle/working/adversarial-patch-transferability/pretrained_models/PIDNet/PIDNet_S_Cityscapes_test.pt',map_location=self.device)
      if '_m' in self.config.model.name:
        model = torch.load('/kaggle/working/adversarial-patch-transferability/pretrained_models/PIDNet/PIDNet_M_Cityscapes_test.pt',map_location=self.device)
      if '_l' in self.config.model.name:
        model = torch.load('/kaggle/working/adversarial-patch-transferability/pretrained_models/PIDNet/PIDNet_L_Cityscapes_test.pt',map_location=self.device)
      
  
      pidnet = get_pred_model(name = self.config.model.name, num_classes = 19).to(self.device)
      if 'state_dict' in model:
          model = model['state_dict']
      model_dict = pidnet.state_dict()
      model = {k[6:]: v for k, v in model.items() # k[6:] to start after model. in key names
                          if k[6:] in model_dict.keys()}

      pidnet.load_state_dict(model)
      self.model = pidnet
      self.model.eval()
      

    if 'bisenet' in self.config.model.name:
      if '_v1' in self.config.model.name:
        model = torch.load('/kaggle/working/adversarial-patch-transferability/pretrained_models/BisNetV1/bisnetv1.pth',map_location=self.device)
        bisenet = BiSeNetV1(19,aux_mode = 'eval').to(self.device)
        bisenet.load_state_dict(model, strict=False)
      if '_v2' in self.config.model.name:
        model = torch.load('/kaggle/working/adversarial-patch-transferability/pretrained_models/BisNetV2/bisnetv2.pth',map_location=self.device)
        bisenet = BiSeNetV2(19,aux_mode = 'eval').to(self.device)
        bisenet.load_state_dict(model, strict=False)
      self.model = bisenet
      self.model.eval()


    if 'icnet' in self.config.model.name:
      model = torch.load('/kaggle/working/adversarial-patch-transferability/pretrained_models/ICNet/Copy of resnet50_2024-12-22 08:52:50 EST-0500_176_0.661.pth.tar',map_location=self.device)
      icnet = ICNet(nclass = 19).to(self.device)
      icnet.load_state_dict(model['model_state_dict'])
      self.model = icnet
      self.model.eval()

    if 'segformer' in self.config.model.name:
      feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
      segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(self.device)
      self.model = segformer
      self.model.eval()

    if 'pspnet' in self.config.model.name:
      # Load pretrained PSPNet weights
      if '101' in self.config.model.name:
          pspnet = PSPNet(layers=101).to(self.device)

          # Load Cityscapes pretrained PSPNet weights
          state_dict = torch.load(
              '/kaggle/working/adversarial-patch-transferability/pretrained_models/PSPNet/pspnet101_cityscapes.pth',
              map_location=self.device
          )
          pspnet.load_state_dict(state_dict, strict=False)

      elif '50' in self.config.model.name:
          pspnet = PSPNet(layers=50).to(self.device)

          state_dict = torch.load(
              '/kaggle/working/adversarial-patch-transferability/pretrained_models/PSPNet/pspnet50_cityscapes.pth',
              map_location=self.device
          )
          pspnet.load_state_dict(state_dict, strict=False)

      self.model = pspnet
      self.model.eval()

    if 'deeplab' in self.config.model.name:
      if '50' in self.config.model.name:
          deeplab = DeepLabV3(model_id="deeplabv3_resnet50", 
                              project_dir="/kaggle/working/").to(self.device)

          # Load pretrained weights (Cityscapes)
          state_dict = torch.load(
              '/kaggle/working/adversarial-patch-transferability/pretrained_models/DeepLabV3/deeplabv3_cityscapes.pth',
              map_location=self.device
          )
          deeplab.load_state_dict(state_dict, strict=False)
          # ---- Drop the 20th neuron ----
          old_cls = deeplab.aspp.conv_1x1_4  # final classifier Conv2d(256 -> 20)
          new_cls = nn.Conv2d(
              old_cls.in_channels, 
              19, 
              kernel_size=old_cls.kernel_size, 
              stride=old_cls.stride, 
              padding=old_cls.padding, 
              bias=True
          ).to(self.device)

          # Copy weights for first 19 classes
          with torch.no_grad():
              new_cls.weight.copy_(old_cls.weight[:19, :, :, :])
              if old_cls.bias is not None:
                  new_cls.bias.copy_(old_cls.bias[:19])

          # Replace final layer in ASPP
          deeplab.aspp.conv_1x1_4 = new_cls


      self.model = deeplab
      self.model.eval()



  def predict(self,image_standard,size):
    image_standard = image_standard.to(self.device)
    outputs = self.model(image_standard)
    if 'pidnet' in self.config.model.name:
      output = F.interpolate(
                    outputs[self.config.test.output_index_pidnet], size[-2:],
                    mode='bilinear', align_corners=True
                )

    if 'segformer' in self.config.model.name:
      output = F.interpolate(
                    outputs.logits, size[-2:],
                    mode='bilinear', align_corners=True
                )

    if 'icnet' in self.config.model.name:
      output = outputs[self.config.test.output_index_icnet]

    if 'bisenet' in self.config.model.name:
      ## Images needs to be unnormalized and then normalized as:
      ## mean=[0.3257, 0.3690, 0.3223], std=[0.2112, 0.2148, 0.2115]
      ## The it will give 75% miou instead of 71 and to keep things simple keeping it as it
      output = outputs[self.config.test.output_index_bisenet]

    if 'pspnet' in self.config.model.name:
      output = outputs  # already logits
      output = F.interpolate(output, size[-2:], mode='bilinear', align_corners=True)

    if 'deeplab' in self.config.model.name:
      output = outputs  # already logits
      output = F.interpolate(output, size[-2:], mode='bilinear', align_corners=True)

    return output


    





