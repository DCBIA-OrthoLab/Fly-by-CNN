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
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene


def dataset(data):
    model_lst = []
    landmarks_lst = []
    datalist = []
    normpath = os.path.normpath("/".join([data, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
            if True in ['Lower' in img_fn]:
                model_lst.append(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".json"]]:
            if True in ['Lower' in img_fn]:
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

def generate_sphere_mesh(center,radius,device,col):
    sphereSource = vtk.vtkSphereSource()
    # print(center)
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

    verts_rgb = torch.ones_like(verts_teeth)[None]+col  # (1, V, 3)
    # color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts_teeth], 
        faces=[faces_teeth],
        textures=textures).to(device)
    
    return mesh,verts_teeth,faces_teeth,verts_rgb

def merge_meshes(agents,V,F,CN,device):
    center_vert = torch.empty((0)).to(device)
    center_faces = torch.empty((0)).to(device)
    center_text = torch.empty((0)).to(device)

    for image in range(V.shape[0]):
        # print(agents[aid].sphere_centers[image])
        # print(agents[aid].sphere_centers[...,0])
        center_mesh,agent_verts,agent_faces,textures= generate_sphere_mesh(agents.sphere_centers[image],0.02,device,0.9)
        text = torch.ones_like(agent_verts)
        center_text = torch.cat((center_text,text.unsqueeze(0)),dim=0)
        center_vert = torch.cat((center_vert,agent_verts.unsqueeze(0)),dim=0)
        center_faces = torch.cat((center_faces,agent_faces.unsqueeze(0)),dim=0)
    

    # print(center_vert.shape)
    # print(center_faces.shape)
    # print(V.shape)
    # print(F.shape)
    # print(CN.shape)
    # print(center_text.shape)
    # print(center_vert.shape)
    # print(center_faces.shape)
    # print(center_faces[-1])
    

    verts = torch.cat([center_vert,V], dim=1)
    faces = torch.cat([center_faces,F+center_vert.shape[1]], dim=1)
    text = torch.cat([center_text,CN], dim=1)
    
    # verts = center_vert
    # faces = center_faces
    # text = center_text
    # print(verts.shape)
    # print(faces.shape)
    # print(text.shape)


    textures = TexturesVertex(verts_features=text)
        
    meshes =  Meshes(
        verts=verts,   
        faces=faces, 
        textures=textures
    )
    return meshes

def Training(epoch, agents, agents_ids,num_step, train_dataloader, loss_function, optimizer, device):
    # for batch, (V, F, CN, LP, MR, SF) in enumerate(train_dataloader):
        
    torch.autograd.set_detect_anomaly(True)
    #Gravitational law F = G * (m_1*m_2/r^2)

    # G = torch.tensor(6.67408)#e-11 #gravitational constant
    # m_1 = torch.tensor(1.98)#e30 #kg mass of the sun
    # m_2 = torch.tensor(0.000005972)#e30 #kg mass of the earth
    epsilon = torch.tensor(1e-10)
    discount_factor = torch.tensor(0.8)
    

    for batch, (V, F, CN, LP, MR, SF) in enumerate(train_dataloader):
        # textures = TexturesVertex(verts_features=CN)
        # print(CN.shape)
        # meshes = Meshes(
        #     verts=V,   
        #     faces=F, 
        #     textures=textures
        # )
        # batchsize
        
        # batch_loss = 0

        batch_g_force = 0

        optimizer.zero_grad()

        for aid in agents_ids: #aid == idlandmark_id
            agents[aid].reset_sphere_center(V.shape[0], random=True)

            print('---------- agents id :', aid,'----------')

            lm_pos = torch.empty((0)).to(device)
            for lst in LP:
                lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)

            NSteps = num_step
            A_i_gforce = 0
        
            # agents[aid].trainable(True)
            # agents[aid].train()

            for i in range(NSteps):
                print('---------- step :', i,'----------')
                # print(agents[aid].sphere_centers)
                
                meshes = merge_meshes(agents[aid],V,F,CN,device)
                # dic = {"teeth_mesh": meshes}
                # plot_fig(dic)
                x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]

                x = x + agents[aid].sphere_centers
                #f_i = G*m_1*m_2/(loss_function(x, lm_pos) + epsilon) 
                f_i = 1.0/(loss_function(x, lm_pos) + epsilon) 
                A_i_gforce = A_i_gforce + f_i*torch.pow(discount_factor, i)
                
                agents[aid].sphere_centers = x

            
            print(f"agent {aid} force:", A_i_gforce.item())
            batch_g_force = batch_g_force + A_i_gforce
            agents[aid].writer.add_scalar('force', batch_g_force, epoch)

        batch_g_force = -1*batch_g_force # maximize
        batch_g_force.backward()   # backward propagation
        optimizer.step()   # tell the optimizer to update the weights according to the gradients and its internal optimisation strategy 

        #     batch_loss += aid_loss
        
        # batch_loss /= len(agents_ids)
        # writer.add_scalar('distance',batch_loss)
        
def Validation(epoch,agents,agents_ids,test_dataloader,num_step,loss_function,output_dir,early_stopping,device):
    with torch.no_grad():

        running_loss = 0

        for batch, (V, F, CN, LP, MR, SF) in enumerate(test_dataloader):
            # textures = TexturesVertex(verts_features=CN)
            # meshes = Meshes(
            #     verts=V,   
            #     faces=F, 
            #     textures=textures
            # )
            batch_loss = 0

            for aid in agents_ids: #aid == idlandmark_id
                agents[aid].reset_sphere_center(V.shape[0])

                print('---------- agents id :', aid,'----------')

                lm_pos = torch.empty((0)).to(device)
                for lst in LP:
                    lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)

                NSteps = num_step

                for i in range(NSteps):
                    print('---------- step :', i,'----------')
                    meshes = merge_meshes(agents[aid],V,F,CN,device)

                    x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]
                    
                    # delta_pos =  x[...,0:3]
                    
                    # delta_pos += agents[aid].sphere_centers
                    
                    # # agents[aid].set_radius(x[...,3:4].clone().detach())

                    # agents[aid].sphere_centers = delta_pos.detach().clone()
                    x = x + agents[aid].sphere_centers
                    agents[aid].sphere_centers = x.clone().detach()

                aid_loss = loss_function(x, lm_pos)
                
                batch_loss += aid_loss

                print(f"agent {aid} loss:", aid_loss.item())
                
                agents[aid].writer.add_scalar('Validation',aid_loss,epoch)

            running_loss += batch_loss

            # early_stopping(running_loss, agents)
            
            # return early_stopping.early_stop
        early_stopping(running_loss, agents)

        return early_stopping.early_stop
                
            # if aid_loss<agents[aid].best_loss:
            #     agents[aid].best_loss = aid_loss
            #     agents[aid].best_epoch_loss = epoch
            #     torch.save(agents[aid].attention, os.path.join(output_dir, f"best_attention_net_{aid}.pth"))
            #     torch.save(agents[aid].delta_move, os.path.join(output_dir, f"best_delta_move_net_{aid}.pth"))
            #     print("saved new best metric network")
            #     print(f"Model Was Saved ! Current Best Avg. Dice: {agents[aid].best_loss} at epoch: {agents[aid].best_epoch_loss}")

            #     batch_loss += aid_loss
            
            # batch_loss /= len(agents_ids)
            # writer.add_scalar('distance',batch_loss)


        # for batch, (V, F, CN, LP) in enumerate(test_dataloader):

        #     textures = TexturesVertex(verts_features=CN)
        #     meshes = Meshes(
        #         verts=V,   
        #         faces=F, 
        #         textures=textures
        #     )
            
        #     for aid in agents_ids: #aid == idlandmark_id
        #         print('---------- agents id :', aid,'----------')
        #         agents[aid].reset_sphere_center(V.shape[0])

        #         NSteps =  num_step
        #         aid_loss = 0
        #         epoch_loss = 0
        #         agents[aid].eval() 

        #         for i in range(NSteps):
        #             print('---------- step :', i,'----------')

        #             x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]

        #             x += agents[aid].sphere_centers
        #             # print('coord sphere center :', agent.sphere_center)
                    
        #             lm_pos = torch.empty((0)).to(device)
        #             for lst in LP:
        #                 lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)
                    
        #             loss = torch.sqrt(loss_function(x, lm_pos))
        #             print('agent position : ', x)
        #             print('landmark position :', lm_pos)

        #             l = loss.item()
        #             aid_loss += l
        #             print("Step loss:",l)
        #             agents[aid].sphere_centers = x.detach().clone()
                    
                    
        #         aid_loss /= NSteps
        #         print("Step loss:", aid_loss)
        #         epoch_loss += aid_loss

        #         if aid_loss<agents[aid].best_loss:
        #             agents[aid].best_loss = aid_loss
        #             agents[aid].best_epoch_loss = epoch
        #             torch.save(agents[aid].attention, os.path.join(output_dir, f"best_attention_net_{aid}.pth"))
        #             torch.save(agents[aid].delta_move, os.path.join(output_dir, f"best_delta_move_net_{aid}.pth"))
        #             print("saved new best metric network")
        #             print(f"Model Was Saved ! Current Best Avg. Dice: {agents[aid].best_loss} at epoch: {agents[aid].best_epoch_loss}")
            
                
              

        #     epoch_loss /= len(agents_ids)
            
        #     early_stopping(epoch_loss)

        #     if epoch_loss<agents[aid].best_epoch_loss:
        #         torch.save(agents[0].features_net, os.path.join(output_dir, "best_feature_net.pth"))

        #     return early_stopping.early_stop

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
    verts = [v for v, f, cn, lp, sc, ma  in batch]
    faces = [f for v, f, cn, lp, sc, ma  in batch]
    color_normals = [cn for v, f, cn, lp, sc, ma, in batch]
    landmark_position = [lp for v, f, cn, lp, sc, ma in batch]
    scale_factor = [sc for v, f, cn, lp , sc, ma  in batch]
    mean_arr = [ma for v, f, cn,lp, sc, ma   in batch]

    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), landmark_position, mean_arr, scale_factor


def SavePrediction(data, outpath):
    print("Saving prediction to : ", outpath)
    img = data.numpy()
    output = sitk.GetImageFromArray(img)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)

def pad_verts_faces_prediction(batch):
    verts = [v for v, f, cn, ma , sc, ps in batch]
    faces = [f for v, f, cn, ma , sc, ps in batch]
    color_normals = [cn for v, f, cn, ma , sc, ps in batch]
    mean_arr = [ma for v, f, cn, ma , sc, ps  in batch]
    scale_factor = [sc for v, f, cn, ma , sc, ps in batch]
    path_surf = [ps for v, f, cn, ma , sc,ps in batch]

    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), mean_arr, scale_factor, path_surf

def plot_fig(dic):
        radius_sphere = 0.1
        # R, T = look_at_view_transform(self.distance, self.elevation, self.azimuth, device=self.device,degrees=False)
        # cam_pos=torch.matmul(T,R)
        # print(cam_pos)
        # cam_pos = cam_pos.numpy()[0][0]
        # cam_mesh = generate_sphere_mesh(cam_pos,radius_sphere,self.device)
        # center_mesh = generate_sphere_mesh([0,0,0],radius_sphere,self.device)

        # dic = {"teeth_mesh": teeth_mesh, 'center':center_mesh}
        # for n,lm_mesh in enumerate(self.list_meshe_landmarks):
        #     dic[str(n)] = lm_mesh
        fig = plot_scene({"subplot1": dic},     
            xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
            yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
            zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
            axis_args=AxisArgs(showgrid=True))
            
        fig.show()

def Accuracy(agents,test_dataloader,agents_ids,min_variance,loss_function,device):
    list_distance = ({ 'obj' : [], 'distance' : [] })

    with torch.no_grad():
        for batch, (V, F, CN, LP, MR, SF) in enumerate(test_dataloader):
            groupe_data = {}
            radius = 0.02
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
       
            for aid in agents_ids: #aid == idlandmark_id
                print('---------- agents id :', aid,'----------')
                agents[aid].reset_sphere_center(V.shape[0])
                
                agents[aid].eval() 
                pos_center = agents[aid].search(meshes,min_variance) #[batchsize,3]
                # plot_fig(meshes,center_mesh)

                lm_pos = torch.empty((0)).to(device)
                for lst in LP:
                    lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)  #[batchsize,3]
                                # center_mesh = generate_sphere_mesh(pos_center[0],radius,device,0.9)
                
                perfect_pos = generate_sphere_mesh(lm_pos[0],radius,device,0.0)
                dic = {"teeth_mesh": meshes, 'landmark':perfect_pos}
                for index,step in enumerate(agents[aid].position_center_memory):
                    center_mesh = generate_sphere_mesh(step[0],radius,device,0.9)
                    dic[str(index)]=center_mesh

                plot_fig(dic)
                
                
                for i in range(V.shape[0]):
                    # loss = torch.sqrt(loss_function(pos_center[i], lm_pos[i]))
                    list_distance['obj'].append(str(aid))
                    # list_distance['distance'].append(float(loss.item()))
                    scale_surf = SF[i]
                    # print('scale_surf :', scale_surf)
                    mean_arr = MR[i]
                    # print('mean_arr :', mean_arr)
                    agent_pos = pos_center[i]
                    # print('landmark_pos before rescaling :', agent_pos)
                    new_pos_center = Upscale(agent_pos,mean_arr,scale_surf)#(landmark_pos/scale_surf) + mean_arr
                    # print('pos_center after rescaling :', new_pos_center)
                    landmark_pos = Upscale(lm_pos[i],scale_surf,mean_arr)
                    # print('d',LP[i][aid])
                    # new_landmark_pos = Upscale(LP[i][aid],mean_arr,scale_surf)
                    # print('m',mean_arr)
                    # print('s',scale_surf)
                    # print('u',new_landmark_pos)
                    new_pos_center=new_pos_center.cpu()
                    new_landmark_pos=new_landmark_pos.cpu()
                    distance = np.linalg.norm(new_pos_center-landmark_pos)
                    # print('distance between prediction and real landmark :',distance)
                    list_distance['distance'].append(distance)
                    coord_dic = {"x":new_landmark_pos[0],"y":new_landmark_pos[1],"z":new_landmark_pos[2]}
                    # print(coord_dic)
                    groupe_data[f'Lower_O-{aid+1}']=coord_dic
                    # print(groupe_data)
                    # print(PS[i])
                    # dic_patients[PS[i]]=groupe_data
                # writer.add_scalar('distance',loss)

            # print(list_distance)
        
        sns.violinplot(x='obj',y='distance',data=list_distance)
        plt.show()

def Prediction(agents,dataloader,agents_ids,min_variance,dic_patients):
    # list_distance = ({ 'obj' : [], 'distance' : [] })

    with torch.no_grad():
        for batch, (V, F, CN, MR, SF,PS) in enumerate(dataloader):
            groupe_data = {}
            print(PS)
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            
            for aid in agents_ids: #aid == idlandmark_id
                coord_dic = {}

                print('---------- agents id :', aid,'----------')
                agents[aid].reset_sphere_center(V.shape[0])

                agents[aid].eval() 
                
                pos_center = agents[aid].search(meshes,min_variance) #[batchsize,3]
                print('pos_center',pos_center)
                # lm_pos = torch.empty((0)).to(device)
                # for lst in LP:
                #     lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)  #[batchsize,3]
                
                # loss = loss_function(pos_center, lm_pos)

                # list_distance['obj'].append(str(aid))
                # list_distance['distance'].append(float(loss.item()))
                for i in range(V.shape[0]):
                    # print(pos_center[i],SF[i],MR[i])
                    scale_surf = SF[i]
                    print('scale_surf :', scale_surf)
                    mean_arr = MR[i]
                    print('mean_arr :', mean_arr)
                    landmark_pos = pos_center[i]
                    print('landmark_pos :', landmark_pos)
                    new_pos_center = (landmark_pos/scale_surf) + mean_arr
                    print('pos_center :', new_pos_center)
                    new_pos_center = new_pos_center.cpu().numpy()
                    # print(pos_center)
                    coord_dic = {"x":new_pos_center[0],"y":new_pos_center[1],"z":new_pos_center[2]}
                    groupe_data[f'Lower_O-{aid+1}']=coord_dic
                    print(PS[i])
                    dic_patients[PS[i]]=groupe_data

            # print(list_distance)
        
        print("all the landmarks :" , dic_patients)
    
    return dic_patients

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
            "position": [float(data["x"]), float(data["y"]), float(data["z"])],
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
        print(file)
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close


