import math
import numpy as np 

import torch
from torch import Tensor, nn

import torchvision
from torchvision import models
from torchvision import transforms
import torchmetrics

import monai
from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
from pytorch3d.structures import Meshes

import pytorch_lightning as pl
import utils

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
    def __init__(self, args = None, out_channels=3, class_weights=None, image_size=224):
        super(MonaiUNet, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args
        
        self.class_weights = None
        if(class_weights is not None):
            self.class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True, ce_weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy()

        unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
            out_channels=out_channels, 
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.model = nn.Sequential(TimeDistributed(unet))
        
        self.ico_verts, self.ico_faces, self.ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=1.1, sl=1))
        self.ico_verts = self.ico_verts.to(torch.float32)

        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            bin_size=0,
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):

        V, F, CN = x

        batch_size = V.shape[0]
        textures = TexturesVertex(verts_features=CN.to(torch.float32))
        meshes = Meshes(verts=V.to(torch.float32), faces=F, textures=textures)
        
        X, PF = self.render(meshes, batch_size)

        x = self.model(X)
        
        return x, X, PF

    def render(self, meshes, batch_size=1):

        X = []
        PF = []

        sphere_centers = torch.zeros([batch_size, 3]).to(torch.float32).to(self.device)
        meshes = meshes.to(self.device)
        renderer = self.renderer.to(self.device)

        for camera_position in self.ico_verts:
            
            current_cam_pos = sphere_centers + camera_position.to(self.device)            

            R = look_at_rotation(current_cam_pos, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:,:,None])[:, :, 0]   # (1, 3)
            T = T.to(self.device)

            images = renderer(meshes_world=meshes.clone(), R=R, T=T)            
            pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes.clone())

            pix_to_face = pix_to_face.permute(0, 3, 1, 2)
            images = images.permute(0, 3, 1, 2)            
            zbuf = zbuf.permute(0, 3, 1, 2)

            images = images[:,:-1,:,:] #grab RGB components only
            images = torch.cat([images, zbuf], dim=1) #append the zbuf as a channel

            X.append(images.unsqueeze(dim=1))
            PF.append(pix_to_face.unsqueeze(dim=1))
            
        return torch.cat(X, dim=1), torch.cat(PF, dim=1)


    def training_step(self, train_batch, batch_idx):
        V, F, Y, YF, CN = train_batch        

        batch_size = V.shape[0]
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)
        
        X, PF = self.render(meshes, batch_size)
        y = torch.take(YF, PF)*(PF >= 0) # YF=input, pix_to_face=index. shape of y = shape of pix_to_face

        x = self.model(X)

        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4).to(torch.int64)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        x = torch.argmax(x, dim=1, keepdim=True)
        self.accuracy(torch.argmax(x, dim=1).reshape(-1, 1), y.reshape(-1, 1))
        self.log("train_acc", self.accuracy)
        

        grid_X = torchvision.utils.make_grid(X[:6, 0, 0:3, :, :])
        self.logger.experiment.add_image('X', grid_X, 0)
        
        grid_x = torchvision.utils.make_grid(x[:6, :, 0, :, :])
        self.logger.experiment.add_image('x', grid_x, 0)

        grid_y = torchvision.utils.make_grid(y[:6, :, 0, :, :])
        self.logger.experiment.add_image('Y', grid_y, 0)

        return loss

    def validation_step(self, val_batch, batch_idx):
        V, F, Y, YF, CN = val_batch
        
        batch_size = V.shape[0]
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)
        
        X, PF = self.render(meshes, batch_size)
        y = torch.take(YF, PF)*(PF >= 0)

        x = self.model(X)

        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4).to(torch.int64)

        loss = self.loss(x, y)
        
        self.log('val_loss', loss)

        x = torch.argmax(x, dim=1, keepdim=True)
        self.accuracy(torch.argmax(x, dim=1).reshape(-1, 1), y.reshape(-1, 1))
        self.log("val_acc", self.accuracy)