import numpy   as np
import nibabel as nib
from fsl.data import gifti
from icecream import ic
import sys
sys.path.insert(0,'../..')
import utils
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# rendering components
from pytorchtools import EarlyStopping
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
# datastructures
from pytorch3d.structures import Meshes

import monai
from monai.inferers import (sliding_window_inference,SimpleInferer)
from monai.transforms import ToTensor
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
)
from torch.utils.tensorboard import SummaryWriter

class BrainDataset(Dataset):
    def __init__(self,np_split,triangles):
        self.np_split = np_split
        self.triangles = triangles  
        self.nb_triangles = len(triangles)

    
    def __len__(self):
        return(len(self.np_split))

    def __getitem__(self,idx):
        data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Native_Space'
        #data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Segmentation/Template_Space'
        item = self.np_split[idx][0]

        # for now just try with Left
        #path_features = f'{data_dir}/segmentation_template_space_features/{item}_L.shape.gii'
        #path_labels = f'{data_dir}/segmentation_template_space_labels/{item}_L.label.gii'
        
        path_features = f"{data_dir}/segmentation_native_space_features/{item}_L.shape.gii"
        path_labels = f"{data_dir}/segmentation_native_space_labels/{item}_L.label.gii"
        
        vertex_features = gifti.loadGiftiVertexData(path_features)[1] # vertex features
                
        vertex_labels = gifti.loadGiftiVertexData(path_labels)[1] # vertex labels
        faces_pid0 = self.triangles[:,0:1]

        # vertex_features_0 = vertex_features[:,0]
        # vertex_features_1 = vertex_features[:,1]
        # vertex_features_2 = vertex_features[:,2]
        # vertex_features_3 = vertex_features[:,3]  
        
        # face_features_0 = np.take(vertex_features_0,faces_pid0)
        # face_features_1 = np.take(vertex_features_1,faces_pid0)
        # face_features_2 = np.take(vertex_features_2,faces_pid0)
        # face_features_3 = np.take(vertex_features_3,faces_pid0)
        # l_face_features = [face_features_0,face_features_1,face_features_2,face_features_3]
        
        # face_features = np.stack(l_face_features,axis=-1)
        # face_features = np.squeeze(face_features)        
        
        
        face_labels = np.take(vertex_labels,faces_pid0) # face labels (taking first vertex for each face)        
        
        #offset = np.arange(self.nb_triangles*4).reshape((self.nb_triangles,4))
        offset = np.zeros((self.nb_triangles,4), dtype=int) + np.array([0,1,2,3])
        faces_pid0_offset = offset + np.multiply(faces_pid0,4)        
        
        face_features = np.take(vertex_features,faces_pid0_offset)
        
        
        """
        ic(vertex_features.shape)
        ic(face_features.shape)
        ic(vertex_labels.shape)
        ic(face_labels.shape)
        ic(faces_pid0.shape)
        ic(faces_pid0_offset.shape)
        ic(faces_pid0)
        ic(faces_pid0_offset)
        ic(vertex_features)
        ic(face_features)
        ic(face_features_test)
        """

        
        return vertex_features,face_features, face_labels

def main():

    ###
    ### TRAINING PARAMETERS
    ###

    path_ico = '/CMF/data/geometric-deep-learning-benchmarking/Icospheres/ico-6.surf.gii'
    image_size = 512
    dist_cam = 2
    batch_size = 4
    nb_epochs = 1_000
    nb_loops = 12
    nb_val = 0
    val_interval = 2    
    train_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_train_TEA.npy'
    val_split_path = '/CMF/data/geometric-deep-learning-benchmarking/Train_Val_Test_Splits/Segmentation/M-CRIB-S_val_TEA.npy'

    num_classes = 39 #37
    model_name= "checkpoints/05-03.pt"
    patience = 500
    early_stopping = EarlyStopping(patience=patience, verbose=True,path=model_name)
    write_image_interval = 1

    ###
    ###
    ###


    # Set the cuda device 
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # Initialize a perspective camera.
    cameras = FoVPerspectiveCameras(device=device)


    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0, 
        faces_per_pixel=1, 
    )
    # We can add a point light in front of the object. 

    # lights = AmbientLights(device=device)
    lights = PointLights(device=device)
    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    phong_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, cameras=cameras)
    )


    # Setup Neural Network model

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes, num_classes=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, num_classes=num_classes)

    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
        out_channels=num_classes, 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # model.load_state_dict(torch.load("early_stopping/checkpoint_1.pt"))

    loss_function = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()

    
    # load icosahedron
    ico_surf = nib.load(path_ico)

    # extract points and faces
    coords = ico_surf.agg_data('pointset')
    triangles = ico_surf.agg_data('triangle')
    nb_faces = len(triangles)
    connectivity = triangles.reshape(nb_faces*3,1) # 3 points per triangle
    connectivity = np.int64(connectivity)	
    offsets = [3*i for i in range (nb_faces)]
    offsets.append(nb_faces*3) #  The last value is always the length of the Connectivity array.
    offsets = np.array(offsets)

    # rescale icosphere [0,1]
    coords = np.multiply(coords,0.01)

    # convert to vtk
    vtk_coords = vtk.vtkPoints()
    vtk_coords.SetData(numpy_to_vtk(coords))
    vtk_triangles = vtk.vtkCellArray()
    vtk_offsets = numpy_to_vtkIdTypeArray(offsets)
    vtk_connectivity = numpy_to_vtkIdTypeArray(connectivity)
    vtk_triangles.SetData(vtk_offsets,vtk_connectivity)

    """
    # Create icosahedron as a VTK polydata 
    ico_polydata = vtk.vtkPolyData() # initialize polydata
    ico_polydata.SetPoints(vtk_coords) # add points
    ico_polydata.SetPolys(vtk_triangles) # add polys
    """

    # convert ico verts / faces to tensor
    ico_verts = torch.from_numpy(coords).unsqueeze(0).to(device)
    ico_faces = torch.from_numpy(triangles).unsqueeze(0).to(device)


    # load train / test splits

    train_split = np.load(train_split_path)
    val_split = np.load(val_split_path)
    train_dataset = BrainDataset(train_split,triangles)
    val_dataset = BrainDataset(val_split,triangles)

    # Match icosphere vertices and faces tensor with batch size
    l_ico_verts = []
    l_ico_faces = []
    for i in range(batch_size):
        l_ico_verts.append(ico_verts)
        l_ico_faces.append(ico_faces)    
    batched_ico_verts = torch.cat(l_ico_verts,dim=0)
    batched_ico_faces = torch.cat(l_ico_faces,dim=0)

        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=True)


    for epoch in range(nb_epochs):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{nb_epochs}")
        epoch_loss = 0
        step = 0

        ##
        ## TRAINING
        ##

        for batch, (vertex_features, face_features, face_labels) in enumerate(train_dataloader):
            
            vertex_features = vertex_features.to(device)
            vertex_features = vertex_features[:,:,0:3]
            face_labels = torch.squeeze(face_labels,0)
            face_labels = face_labels.to(device)
            face_features = face_features.to(device)

            step += 1
            for s in range(nb_loops):
                textures = TexturesVertex(verts_features=vertex_features)
                try:
                    meshes = Meshes(
                        verts=batched_ico_verts,   
                        faces=batched_ico_faces, 
                        textures=textures
                    )
                except ValueError:
                    reduced_batch_size = vertex_features.shape[0]
                    l_ico_verts = []
                    l_ico_faces = []
                    for i in range(reduced_batch_size):
                        l_ico_verts.append(ico_verts)
                        l_ico_faces.append(ico_faces)  
                    batched_ico_verts,batched_ico_faces  = torch.cat(l_ico_verts,dim=0), torch.cat(l_ico_faces,dim=0)
                    meshes = Meshes(
                        verts=batched_ico_verts,   
                        faces=batched_ico_faces, 
                        textures=textures
                    )
                array_coord = np.random.normal(0, 1, 3)
                array_coord *= dist_cam/(np.linalg.norm(array_coord))
                camera_position = ToTensor(dtype=torch.float32, device=device)([array_coord.tolist()])
                R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
                T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

                images = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)    
                pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone()) 
                labels = torch.take(face_labels, pix_to_face)*(pix_to_face >= 0) 
                l_inputs = []
                for index in range(4):
                    l_inputs.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0)) # take each feature     
                inputs = torch.cat(l_inputs,dim=3)            
                inputs, labels  = inputs.permute(0,3,1,2), labels.permute(0,3,1,2)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs,labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # epoch_len = int(np.ceil(len(train_dataloader) / train_dataloader.batch_size))
                epoch_len = int(np.ceil(len(train_dataloader)))
                
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= (step*nb_loops)
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")        
        writer.add_scalar("training_loss", epoch_loss, epoch + 1)

                

        # VALIDATION

        if (epoch) % val_interval == 0: # every two epochs : validation
            nb_val += 1 
            model.eval()
            with torch.no_grad():
                val_inputs = None
                val_labels = None
                val_outputs = None
                for batch, (vertex_features, face_features, face_labels) in enumerate(train_dataloader):
                    
                    vertex_features = vertex_features.to(device)
                    vertex_features = vertex_features[:,:,0:3]
                    face_labels = torch.squeeze(face_labels,0)
                    face_labels = face_labels.to(device)
                    face_features = face_features.to(device)

                    textures = TexturesVertex(verts_features=vertex_features)
                    try:
                        meshes = Meshes(
                            verts=batched_ico_verts,   
                            faces=batched_ico_faces, 
                            textures=textures
                        )
                    except ValueError:
                        reduced_batch_size = vertex_features.shape[0]
                        l_ico_verts = []
                        l_ico_faces = []
                        for i in range(reduced_batch_size):
                            l_ico_verts.append(ico_verts)
                            l_ico_faces.append(ico_faces)  
                        batched_ico_verts,batched_ico_faces  = torch.cat(l_ico_verts,dim=0), torch.cat(l_ico_faces,dim=0)
                        meshes = Meshes(
                            verts=batched_ico_verts,   
                            faces=batched_ico_faces, 
                            textures=textures
                        )
                    array_coord = np.random.normal(0, 1, 3)
                    array_coord *= dist_cam/(np.linalg.norm(array_coord))
                    camera_position = ToTensor(dtype=torch.float32, device=device)([array_coord.tolist()])
                    R = look_at_rotation(camera_position, device=device)  # (1, 3, 3)
                    T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

                    images = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)    
                    pix_to_face, zbuf, bary_coords, dists = phong_renderer.rasterizer(meshes.clone()) 
                    labels = torch.take(face_labels, pix_to_face)*(pix_to_face >= 0) 
                    l_inputs = []
                    for index in range(4):
                        # take each feature
                        l_inputs.append(torch.take(face_features[:,:,index],pix_to_face)*(pix_to_face >= 0))      
                    inputs = torch.cat(l_inputs,dim=3)            
                    val_inputs, val_labels  = inputs.permute(0,3,1,2), labels.permute(0,3,1,2)


                    roi_size = (image_size, image_size)
                    sw_batch_size = batch_size
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)               


                    val_labels_list = decollate_batch(val_labels)                
                    val_labels_convert = [
                        post_label(val_label_tensor) for val_label_tensor in val_labels_list
                    ]
                    
                    val_outputs_list = decollate_batch(val_outputs)
                    val_outputs_convert = [
                        post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                    ]
                    
                    dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                    
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)

                if nb_val % 4 == 0: # save every 4 validations
                    torch.save(model.state_dict(), model_name)
                    print(f'saving model: {model_name}')

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "trash.pth")
                    print("saved new best metric model")
                    print(model_name)
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(1-metric, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break 

                writer.add_scalar("validation_mean_dice", metric, epoch + 1)
                imgs_output = torch.argmax(val_outputs, dim=1).detach().cpu()
                imgs_output = imgs_output.unsqueeze(1)  # insert dim of size 1 at pos. 1
                imgs_normals = val_inputs[:,0:3,:,:]
                val_rgb = torch.cat((255*(1-2*val_labels/33),255*(2*val_labels/33-1),val_labels),dim=1) 
                out_rgb = torch.cat((255*(1-2*imgs_output/33),255*(2*imgs_output/33-1),imgs_output),dim=1) 
                
                val_rgb[:,2,...] = 255 - val_rgb[:,1,...] - val_rgb[:,0,...]
                out_rgb[:,2,...] = 255 - out_rgb[:,1,...] - out_rgb[:,0,...]

                norm_rgb = imgs_normals

                if nb_val %  write_image_interval == 0:       
                    writer.add_images("labels",val_rgb,epoch)
                    writer.add_images("output", out_rgb,epoch)
                    writer.add_images("normals",norm_rgb,epoch)
                
                
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == '__main__':
    main()

