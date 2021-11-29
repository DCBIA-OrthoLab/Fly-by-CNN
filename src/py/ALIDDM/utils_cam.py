from torch._C import device
import vtk
import os
import glob
from utils import PolyDataToTensors
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from utils_class import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,
)
import csv

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
    
    # if len(model_lst) != len(landmarks_lst):
    #     print("ERROR : Not the same number of models and landmarks file")
    #     return
    
    # for file_id in range(0,len(model_lst)):
    #     data = {"model" : model_lst[file_id], "landmarks" : landmarks_lst[file_id]}
    #     datalist.append(data)
    
    # # for i in datalist:
    # #     print("datalist :",i)
    # # print(datalist)
    # return datalist
    
    
    outfile = os.path.join(os.path.dirname(data),'data_O.csv')
    fieldnames = ['surf', 'landmarks', 'number_of_landmarks']
    data_list = []
    for idx,path in enumerate(landmarks_lst):
        data = json.load(open(path))
        markups = data['markups']
        landmarks_dict = markups[0]['controlPoints']
        number_of_landmarks = len(landmarks_dict)

        rows = {'surf':model_lst[idx],
                'landmarks':path,
                'number_of_landmarks':number_of_landmarks }
        data_list.append(rows)
    
    with open(outfile, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)

    return outfile

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

def training(agents, agents_ids, train_dataloader, loss_function, optimizer, epoch_loss, device):
    for batch, (V, F, CN, LP) in enumerate(train_dataloader):
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )
        
        # list_pictures = agent.shot(meshes)
        # agent.affichage(list_pictures)
        img_batch = torch.empty((0)).to(device)

        for aid in agents_ids: #aid == idlandmark_id
            print('---------- agents id :', aid,'----------')

            NSteps = 10
            step_loss = 0
        
            agents[aid].trainable(True)
            agents[aid].train()

            for i in range(NSteps):
                print('---------- step :', i,'----------')

                optimizer.zero_grad()   # prepare the gradients for this step's back propagation

                x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]
                
                x += agents[aid].sphere_centers
                # print('coord sphere center :', agent.sphere_center)
                
                lm_pos = torch.empty((0)).to(device)
                for lst in LP:
                    lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)
                # print(lm_pos)
                
                loss = loss_function(x, lm_pos)

                loss.backward()   # backward propagation
                optimizer.step()   # tell the optimizer to update the weights according to the gradients and its internal optimisation strategy
                
                l = loss.item()
                step_loss += l
                print("Step loss:",l)
                agents[aid].sphere_centers = x.detach().clone()
            
            step_loss /= NSteps
            agents[aid].trainable(False)

            print("Step loss:", step_loss)
            epoch_loss += step_loss

def validation(epoch,agents,agents_ids,test_dataloader,num_step,num_agents,loss_function,best_deplacment,best_deplacment_epoch,out,device):
    list_distance = []
    with torch.no_grad():
        for batch, (V, F, CN, LP) in enumerate(test_dataloader):

            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            
            for aid in agents_ids: #aid == idlandmark_id
                print('---------- agents id :', aid,'----------')

                NSteps =  num_step
                aid_loss = 0
                agents[aid].eval() 

                for i in range(NSteps):
                    print('---------- step :', i,'----------')

                    x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]

                    x += agents[aid].sphere_centers
                    # print('coord sphere center :', agent.sphere_center)
                    
                    lm_pos = torch.empty((0)).to(device)
                    for lst in LP:
                        lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)
                    
                    loss = loss_function(x, lm_pos)

                    l = loss.item()
                    aid_loss += l
                    print("Step loss:",l)
                    agents[aid].sphere_centers = x.detach().clone()
                
                aid_loss /= NSteps
                print("Step loss:", aid_loss)
                list_distance.append(aid_loss)
        
        mean_distance = torch.sum(list_distance)/num_agents

        if mean_distance<best_deplacment:
            best_deplacment=mean_distance
            best_deplacment_epoch = epoch + 1
            output_dir = os.path.join(out, "best_move_net")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(agents, os.path.join(output_dir, "best_move_net.pth"))
            print("saved new best metric network")
            print(f"Model Was Saved ! Current Best Avg. Dice: {best_deplacment} at epoch: {best_deplacment_epoch}")

# def validation(epoch,move_net,test_dataloader,phong_renderer,loss_function,list_distance,best_deplacment,best_deplacment_epoch,out,device):
#     move_net.eval() 
#     with torch.no_grad():
#         for batch, (V, F, Y, F0, CN, IP,IL) in enumerate(test_dataloader):
            
#             textures = TexturesVertex(verts_features=CN)
#             meshes = Meshes(
#                 verts=V,   
#                 faces=F, 
#                 textures=textures
#             )
            
#             camera_net = CameraNet(meshes, phong_renderer)
#             NSteps = 10
#             NRandomPosition = 2
#             # img_batch = torch.empty((0)).to(device)
#             for r in range(NRandomPosition):
#                 print(r)
#                 camera_net.set_random_position()
#                 for i in range(NSteps):
#                     print("step :", i)
#                     images = camera_net.shot().to(device) #[batchsize,3,224,224]
#                     x = move_net(images)  # [batchsize,3]  return the deplacment 
#                     x += torch.cat((camera_net.camera_position,camera_net.focal_pos),dim=1)
#                     camera_net.move(x.detach().clone())
#                     camera_net.move_focal(x.detach().clone())
#                     # img_batch = torch.cat((img_batch,images),dim=0)
                
#                 distance = loss_function(x, torch.cat((IP,IL),dim=1))
#                 list_distance.append(distance)
            
#             mean_distance = torch.sum(distance)/2

#             if mean_distance<best_deplacment:
#                 best_deplacment=mean_distance
#                 best_deplacment_epoch = epoch + 1
#                 output_dir = os.path.join(out, "best_move_net")
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#                 torch.save(move_net, os.path.join(output_dir, "best_move_net.pth"))
#                 print("saved new best metric network")
#                 print(f"Model Was Saved ! Current Best Avg. Dice: {best_deplacment} at epoch: {best_deplacment_epoch}")

def affichage(data_loader,phong_renderer):
    for batch, (V, F, Y, F0, CN, IP,IL) in enumerate(data_loader):
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )
        
        agent = Agent(meshes,phong_renderer)
        list_pictures = agent.shot().to(device)
        for pictures in list_pictures:
            plt.imshow(pictures)
            plt.show()

def pad_verts_faces(batch):
    verts = [v for v, f, cn, lp in batch]
    faces = [f for v, f, cn, lp in batch]
    # region_ids = [rid for v, f, rid, fpid0, cn, ip, lp in batch]
    # faces_pid0s = [fpid0 for v, f, fpid0, cn, ip, lp in batch]
    color_normals = [cn for v, f, cn, lp in batch]
    # ideal_position = [ip for v, f, fpid0, cn, ip, lp in batch]
    landmark_position = [lp for v, f, cn, lp in batch]
    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), landmark_position

# def SavePrediction(data, outpath):
#     print("Saving prediction to : ", outpath)
#     img = data.numpy()
#     output = sitk.GetImageFromArray(img)
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(outpath)
#     writer.Execute(output)

# def Accuracy(agents,df_val,phong_renderer,min_variance,loss_function,writer,device):
#     list_distance = ({'obj' : [], 'distance':[]})
#     agents.eval()
#     with torch.no_grad():
#         for batch, (V,F,CN,LP) in enumerate(df_val):
#             textures = TexturesVertex(verts_features=CN)
#             meshes = Meshes(
#                 verts=V,   
#                 faces=F, 
#                 textures=textures
#             )
#             agent = Agent(phong_renderer,device)

#             center_pos = agent.search(move_net,min_variance,writer,device)
            
#             # distance = loss_function(torch.cat((camera_net.camera_position,camera_net.focal_pos),dim=1), torch.cat((IP,IL),dim=1))
#             # print(camera_net.camera_position)
#             # print(IP.shape)
#             # print(distance)
#             landmarks_position = LP.cpu().numpy()
#             distance_land = np.linalg.norm(center_pos-landmarks_position)
#             list_distance['obj'].append('agent')
#             list_distance['distance'].append(distance_land)

#             print(list_distance)
                
#         sns.violinplot(x='obj',y='distance',data=list_distance)
#         plt.show()

# def Accuracy(move_net,df_val,phong_renderer,min_variance,loss_function,writer,device):
#     list_distance = ({'obj' : [], 'distance':[]})
#     move_net.eval()
#     with torch.no_grad():
#         for batch, (V, F, Y, F0, CN, IP,IL) in enumerate(df_val):
#             textures = TexturesVertex(verts_features=CN)
#             meshes = Meshes(
#                 verts=V,   
#                 faces=F, 
#                 textures=textures
#             )
#             camera_net = CameraNet(meshes, phong_renderer)
#             camera_net.search(move_net,min_variance,writer,device)
            
#             # distance = loss_function(torch.cat((camera_net.camera_position,camera_net.focal_pos),dim=1), torch.cat((IP,IL),dim=1))
#             # print(camera_net.camera_position)
#             # print(IP.shape)
#             # print(distance)
#             cam_pos = camera_net.camera_position.cpu().numpy()
#             new_IP = IP.cpu().numpy()
#             foc_pos = camera_net.focal_pos.cpu().numpy()
#             new_IL = IL.cpu().numpy()
#             for index,element in enumerate(cam_pos):
#                 distance_cam = np.linalg.norm(element-new_IP[index])
#                 distance_land = np.linalg.norm(foc_pos[index]-new_IL[index])
#                 # print(distance)
#                 list_distance['obj'].append('cam')
#                 list_distance['distance'].append(distance_cam)
#                 list_distance['obj'].append('land')
#                 list_distance['distance'].append(distance_land)

#                 print(list_distance)
                

#         violin_plot = sns.violinplot(x='obj',y='distance',data=list_distance)
#         plt.show()
        

    # std_error = sem(list_distance)
    # print('std error :' , std_error )
    # print('mean error :' , distance)



# def Prediction(move_net,load_model):
#     move_net.eval()
#     move_net.load_state_dict(torch.load(load_model,map_location=device))

#     with torch.no_grad():

#     print("Loading data from :", args.dir)
#             for image in img_model:
#                 new_image = torch.from_numpy(image).permute(2,0,1) # convertion in tensor (7,258,258)
#                 img_output = net(new_image)
#                 # print(torch.from_numpy(img_output).size())
#                 output = torch.cat(img_output,0)
#         output = torch.cat(img_output,0)
#         distance = loss_function(img_output, IP)
#         print('difference between exact and predict position :', distance)
#         list_distance.append(distance)
        
    # SavePrediction(output, output_path)


def GenControlePoint(groupe_data):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data["x"], data["y"], data["z"]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "preview"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(lm_lst,out_path):
    false = False
    true = True
    file = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
    "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": lm_lst,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close