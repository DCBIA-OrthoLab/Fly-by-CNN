from pytorch_lightning.callbacks import Callback
import torchvision
import torch


class TeethNetImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
        if batch_idx % self.log_steps == 0:

                V, F, YF, CN = batch

                batch_size = V.shape[0]
                num_images = min(batch_size, self.num_images)

                V = V.to(pl_module.device, non_blocking=True)
                F = F.to(pl_module.device, non_blocking=True)
                YF = YF.to(pl_module.device, non_blocking=True)
                CN = CN.to(pl_module.device, non_blocking=True).to(torch.float32)

                with torch.no_grad():

                    x, X, PF = pl_module((V[0:1], F[0:1], CN[0:1]))

                    y = torch.take(YF, PF).to(torch.int64)*(PF >= 0) # YF=input, pix_to_face=index. shape of y = shape of pix_to_face
                    x = torch.argmax(x, dim=2, keepdim=True)
                    
                    grid_X = torchvision.utils.make_grid(X[0:num_images, 0, 0:3, :, :])#Grab the first image, RGB channels only, X, Y. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_normals', grid_X, 0)

                    grid_X = torchvision.utils.make_grid(X[0:num_images, 0, 3:, :, :])#Grab the depth map. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_depth', grid_X, 0)
                    
                    grid_x = torchvision.utils.make_grid(x[0:num_images, 0, 0:1, :, :]/pl_module.out_channels)# The time dimension is on dim 1 grab only the first one
                    trainer.logger.experiment.add_image('x', grid_x, 0)

                    grid_y = torchvision.utils.make_grid(y[0:num_images, 0, :, :, :]/pl_module.out_channels)# The time dimension here is swapped after the permute and is on dim=2. It will grab the first image
                    trainer.logger.experiment.add_image('Y', grid_y, 0)