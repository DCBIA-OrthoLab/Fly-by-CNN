import vtk
import os
import glob
from utils import PolyDataToTensors
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from utils_class import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

def dataset(data):
    model_lst = []
    landmarks_lst = []
    datalist = []
    normpath = os.path.normpath("/".join([data, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
            model_lst.append(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".json"]]: 
            landmarks_lst.append(img_fn)
           
    # for i in model_lst:
    #     print("model_lst :",i)
    # for i in landmarks_lst:
    #     print("landmarks_lst :",i)
    
    if len(model_lst) != len(landmarks_lst):
        print("ERROR : Not the same number of models and landmarks file")
        return
    
    for file_id in range(0,len(model_lst)):
        data = {"model" : model_lst[file_id], "landmarks" : landmarks_lst[file_id]}
        datalist.append(data)
    
    # for i in datalist:
    #     print("datalist :",i)
    
    return datalist

def generate_sphere_mesh(center,radius,device):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(center[0],center[1],center[2])
    sphereSource.SetRadius(radius)

    # Make the surface smooth.
    sphereSource.SetPhiResolution(10)
    sphereSource.SetThetaResolution(10)
    sphereSource.Update()
    vtk_landmarks = vtk.vtkAppendPolyData()
    vtk_landmarks.AddInputData(sphereSource.GetOutput())
    vtk_landmarks.Update()

    verts_teeth,faces_teeth = PolyDataToTensors(vtk_landmarks.GetOutput())

    verts_rgb = torch.ones_like(verts_teeth)[None]  # (1, V, 3)
    # color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts_teeth], 
        faces=[faces_teeth],
        textures=textures)
    
    return mesh


def training( epoch, move_net, train_dataloader, phong_renderer, loss_function, optimizer, epoch_loss, writer):
    move_net.train()
    for batch, (V, F, Y, F0, CN, IP) in enumerate(train_dataloader):
            
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )

            camera_net = CameraNet(meshes, phong_renderer)
            NSteps = 10
            step_loss = 0
            img_batch = torch.empty((0))

            for i in range(NSteps):
                optimizer.zero_grad()   # prepare the gradients for this step's back propagation  
                images = camera_net.forward()  #[batchsize,3,224,224]
                x = move_net(images)  # [batchsize,3]  return the deplacment 
                img_batch = torch.cat((img_batch,images),dim=0)
                x += camera_net.camera_position
                loss = loss_function(x, IP)
   
                loss.backward()   # backward propagation
                optimizer.step()   # tell the optimizer to update the weights according to the gradients and its internal optimisation strategy
                
                step_loss += loss.item()
                camera_net.camera_position = x.detach().clone()

            step_loss /= NSteps
            print("Step loss:", step_loss)
            epoch_loss += step_loss
            writer.add_images('image',img_batch,epoch)


def validation(epoch,move_net,test_dataloader,phong_renderer,loss_function,list_distance,best_deplacment,best_deplacment_epoch,out):
    move_net.eval() 
    with torch.no_grad():
        for batch, (V, F, Y, F0, CN, IP) in enumerate(test_dataloader):
            
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            
            camera_net = CameraNet(meshes, phong_renderer)
            NSteps = 2
            img_batch = torch.empty((0))
            for r in range(2):
                print(r)
                camera_net.set_random_position()
                for i in range(NSteps):
                    print("step :", i)
                    images = camera_net.forward()  #[batchsize,3,224,224]
                    x = move_net(images)  # [batchsize,3]  return the deplacment 
                    x += camera_net.camera_position
                    camera_net.camera_position = x.detach().clone()
                    img_batch = torch.cat((img_batch,images),dim=0)
                
                distance = loss_function(x, IP)
                list_distance.append(distance)
            
            mean_distance = torch.sum(distance)/2

            if mean_distance<best_deplacment:
                best_deplacment=mean_distance
                best_deplacment_epoch = epoch + 1
                output_dir = os.path.join(out, "best_move_net")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(move_net, os.path.join(output_dir, "best_move_net.pth"))
                print("saved new best metric network")
                print(f"Model Was Saved ! Current Best Avg. Dice: {best_deplacment} at epoch: {best_deplacment_epoch}")

def pad_verts_faces(batch):
    verts = [v for v, f, rid, fpid0, cn, ip in batch]
    faces = [f for v, f, rid, fpid0, cn, ip in batch]
    region_ids = [rid for v, f, rid, fpid0, cn, ip in batch]
    faces_pid0s = [fpid0 for v, f, rid, fpid0, cn, ip in batch]
    color_normals = [cn for v, f, rid, fpid0, cn, ip in batch]
    ideal_position = [ip for v, f, rid, fpid0, cn, ip in batch]

    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(region_ids, batch_first=True, padding_value=0), pad_sequence(faces_pid0s, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.),pad_sequence(ideal_position, batch_first=True, padding_value=0.0)
