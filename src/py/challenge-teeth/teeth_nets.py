import math
import numpy as np 

import torch
from torch import Tensor, nn

import torchvision
from torchvision import models
from torchvision import transforms
import torchmetrics

import utils

import monai
from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
from pytorch3d.structures import Meshes

import pytorch_lightning as pl

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class MonaiUNet(pl.LightningModule):
    def __init__(self, args = None, out_channels=3, class_weights=None, image_size=320, radius=1.35, subdivision_level=1, train_sphere_samples=4):

        super(MonaiUNet, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args
        
        self.out_channels = out_channels
        self.class_weights = None
        if(class_weights is not None):
            self.class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy()
        
        unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
            out_channels=out_channels, 
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.model = TimeDistributed(unet)

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)
        self.register_buffer("ico_verts", ico_verts)

        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0, 
            faces_per_pixel=1,
            max_faces_per_bin=100000
        )        
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        lights = AmbientLights()
        self.renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=HardPhongShader(cameras=cameras, lights=lights)
        )

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def to(self, device=None):
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, x):

        V, F, CN = x
        
        X = []
        PF = []

        for camera_position in self.ico_verts:
            images, pix_to_face = self.render(V, F, CN, camera_position.unsqueeze(0))
            X.append(images.unsqueeze(0))
            PF.append(pix_to_face.unsqueeze(0))

        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)

        x = self.model(X)
        
        return x*(PF >= 0), X, PF

    def render(self, V, F, CN, camera_position):

        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)        

        R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
        
        images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)
        
        fragments = self.renderer.rasterizer(meshes.clone())
        pix_to_face = fragments.pix_to_face
        zbuf = fragments.zbuf

        images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
        images = images.permute(0,3,1,2)

        pix_to_face = pix_to_face.permute(0,3,1,2)

        return images, pix_to_face

    def training_step(self, train_batch, batch_idx):

        V, F, YF, CN = train_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        YF = YF.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)

        opt = self.optimizers()
        
        batch_size = V.shape[0]

        for i in range(self.hparams.train_sphere_samples):
            camera_position = torch.normal(mean=0.0, std=0.1, size=(3,), device=self.device)
            camera_position *= self.hparams.radius/torch.linalg.norm(camera_position)            
            camera_position = torch.unsqueeze(camera_position, dim=0)

            X, PF = self.render(V, F, CN, camera_position)

            y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        
            opt.zero_grad()

            x = self.model.module(X)*(PF >= 0)
            
            loss = self.loss(x, y)
            self.manual_backward(loss)
            opt.step()
        
            self.log('train_loss', loss, batch_size=batch_size)

    def validation_step(self, val_batch, batch_idx):
        V, F, YF, CN = val_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        YF = YF.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)

        batch_size = V.shape[0]

        val_loss = 0
        val_accuracy = 0

        for i in range(self.hparams.train_sphere_samples):
            camera_position = torch.normal(mean=0.0, std=0.1, size=(3,), device=self.device)
            camera_position *= self.hparams.radius/torch.linalg.norm(camera_position)            
            camera_position = torch.unsqueeze(camera_position, dim=0)

            X, PF = self.render(V, F, CN, camera_position)

            y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)

            x = self.model.module(X)*(PF >= 0)
            
            val_loss += self.loss(x, y)

            val_accuracy += self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
        
        self.log("val_acc", val_accuracy/self.hparams.train_sphere_samples, batch_size=batch_size, sync_dist=True)
        self.log('val_loss', val_loss/self.hparams.train_sphere_samples, batch_size=batch_size, sync_dist=True)