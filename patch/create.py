import torchvision.transforms as transforms
import random
import torch

class Patch:
    def __init__(self,config):
        self.config = config
        self.patch_size = config.patch.size
        # Transformation list for EOT (scaling, rotation, translation)
        self.eot_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=(-20, 20)),  # Random rotation
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random translation
            transforms.RandomResizedCrop(size=self.patch_size, scale=(0.8, 1.2)),  # Random scaling
        ])

    def apply_patch_grad(self, image, label, patch):
        patched_image = image.clone()
        patched_label = label.clone()
        _,c, h, w = image.shape
        location = self.config.patch.loc
        # Get patch starting coordinates
        if location == "random":
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
        elif location == "center":
            x = (w - self.patch_size) // 2
            y = (h - self.patch_size) // 2
        elif location == "corner":
            x = 0
            y = 0
        elif isinstance(location, tuple):
            x, y = location
        else:
            raise ValueError("Invalid location for patch.")

        x_end, y_end = x + self.patch_size, y + self.patch_size
        transformed_patch = patch
        patched_image[:,:, y:y_end, x:x_end] = transformed_patch
        patched_label[:, y:y_end, x:x_end] = 10
        return patched_image, patched_label

    def apply_patch(self, image, label, patch1, patch2, patch3, patch4):#
        """
        Overlay the adversarial patch on the image at a given position.
        """
        patched_image = image.clone()
        patched_label = label.clone()
        _,c, h, w = image.shape
        location = self.config.patch.loc

        # Get patch starting coordinates
        if location == "random":
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
        elif location == "center":
            x = (w - self.patch_size) // 2
            y = (h - self.patch_size) // 2
        elif location == "corner":
            x = 0
            y = 0
        elif isinstance(location, tuple):
            x, y = location
        else:
            raise ValueError("Invalid location for patch.")

        x_mid, y_mid = x + 100, y + 100
        x_end, y_end = x + self.patch_size, y + self.patch_size

        # Apply transformation to patch (EOT)
        #transformed_patch = self.eot_transforms(patch)
        #transformed_patch = patch

        # Overlay patch onto the image and accordingly edit the label
        patched_image[:,:, y:y_mid, x:x_mid] = patch1
        patched_image[:,:, y:y_mid, x_mid:x_end] = patch2
        patched_image[:,:, y_mid:y_end, x:x_mid] = patch3
        patched_image[:,:, y_mid:y_end, x_mid:x_end] = patch4
        #patched_image[:,:, y:y_end, x:x_end] = transformed_patch
        patched_label[:, y:y_end, x:x_end] = self.config.train.ignore_label
        #print(patched_label[:, y:y_end, x:x_end])
        return patched_image, patched_label

    def apply_patch_rand(self, image, label):#
        """
        Overlay the adversarial patch on the image at a given position.
        """
        patched_image = image.clone()
        patched_label = label.clone()
        _,c, h, w = image.shape
        location = self.config.patch.loc

        # Get patch starting coordinates
        if location == "random":
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
        elif location == "center":
            x = (w - self.patch_size) // 2
            y = (h - self.patch_size) // 2
        elif location == "corner":
            x = 0
            y = 0
        elif isinstance(location, tuple):
            x, y = location
        else:
            raise ValueError("Invalid location for patch.")

        x_end, y_end = x + self.patch_size, y + self.patch_size

        # Apply transformation to patch (EOT)
        #transformed_patch = self.eot_transforms(patch)
        #transformed_patch = patch

        # Overlay patch onto the image and accordingly edit the label
        patched_image[:,:, y:y_end, x:x_end] = torch.rand((3, 200, 200))
        #patched_image[:,:, y:y_end, x:x_end] = transformed_patch
        patched_label[:, y:y_end, x:x_end] = self.config.train.ignore_label
        #print(patched_label[:, y:y_end, x:x_end])
        return patched_image, patched_label
