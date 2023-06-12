from pytorch_lightning.callbacks import Callback
import torchvision
import torch


class CleftImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

                V, F, CN, Y = batch

                batch_size = V.shape[0]
                num_images = min(batch_size, self.num_images)

                V = V.to(pl_module.device, non_blocking=True)
                F = F.to(pl_module.device, non_blocking=True)                
                CN = CN.to(pl_module.device, non_blocking=True).to(torch.float32)

                with torch.no_grad():

                    x, X, PF = pl_module((V[0:1], F[0:1], CN[0:1]))
                    
                    grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 0:3, :, :])#Grab the first image, RGB channels only, X, Y. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_normals', grid_X, pl_module.global_step)

                    grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 3:, :, :])#Grab the depth map. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_depth', grid_X, pl_module.global_step)