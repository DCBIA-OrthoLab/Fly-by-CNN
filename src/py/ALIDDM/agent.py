import math as m
import random
from utils import (
    PolyDataToTensors,
    ReadSurf,
    ScaleSurf,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (    
    look_at_view_transform,
    look_at_rotation,
    TexturesVertex,
    FoVPerspectiveCameras, 
    RasterizationSettings,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
)
import torch
import fly_by_features as fbf
# from vtk.util.numpy_support import vtk_to_numpy
import json 
import numpy as np
import vtk
from utils_cam import *
from pytorch3d.vis.plotly_vis import AxisArgs,plot_scene

class environment:
    def __init__(
            self,
            distance=2,
            res=224
            ) -> None:
        
        self.distance=distance
        self.resolution = res
        self.azimuth = -m.pi/2
        self.elevation = m.pi/2
        self.device = torch.device('cpu')
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.camera = FoVPerspectiveCameras(device=self.device)
        self.create_renderer()
        self.list_meshe_landmarks =[]


    def set_random_azimuth(self):
        self.azimut = 2*m.pi*random.random()

    def set_random_elevation(self):
        self.elevation = m.pi*random.random()

    def set_random_rotation(self):
        self.set_random_azimuth()
        self.set_random_elevation()
    
    def create_renderer(self):
        raster_settings = RasterizationSettings(
            image_size=self.resolution, 
            blur_radius=0, 
            faces_per_pixel=1 
        )
    
        lights = PointLights(device=self.device)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=self.camera, 
            raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=self.camera,
                lights=lights
            )
        )
    def set_meshes(self,surf_path):
        surf = ReadSurf(surf_path)
        unit_surf, self.mean_arr, self.scale_factor = ScaleSurf(surf)
        verts_teeth,faces_teeth = PolyDataToTensors(unit_surf)
        verts_rgb = torch.ones_like(verts_teeth)[None]  # (1, V, 3)
        # color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        self.mesh = Meshes(
            verts=[verts_teeth], 
            faces=[faces_teeth],
            textures=textures
        )
        
    def set_landmarks(self,landmarks_path):

        data = json.load(open(landmarks_path))
        markups = data['markups']
        control_point = markups[0]['controlPoints']
        number_landmarks = len(control_point)
        dic_landmarks = {}
        for i in range(number_landmarks):
            label = control_point[i]["label"]
            position = control_point[i]["position"]
            landmark_position = (position - self.mean_arr)*self.scale_factor
            dic_landmarks[label] = landmark_position
        
        self.dic_landmarks=dic_landmarks
        return(self.dic_landmarks)

    def get_view(self):
        R = look_at_rotation(,)
        # R, T = look_at_view_transform(self.distance, self.elevation, self.azimuth, device=self.device, degrees=False)
        # print(R,T)
        images = self.renderer(meshes_world=self.mesh,R=R,T=T)
        images = images.numpy()
        return images
    
    def get_landmarks_angle(self,landmark_id):

        landmark_pos = self.dic_landmarks[landmark_id]

        proj_xy = landmark_pos*[0,1,1]
        vector_landmark_xy = proj_xy / np.linalg.norm(proj_xy)
        vector_z = [0,0,-1] 
        dot_product = np.dot(vector_landmark_xy, vector_z)
        angle_elevation = np.arccos(dot_product)
        if angle_elevation>m.pi:
            print(landmark_id)
            # angle_elevation=2*m.pi-angle_elevation

        proj_xz = landmark_pos*[1,1,0]
        vector_landmark_xz = proj_xz / np.linalg.norm(proj_xz)
        vector_y = [0,1,0] 
        dot_product = np.dot(vector_landmark_xz, vector_y)
        angle_azimuth = np.arccos(dot_product)
        self.azimuth=self.azimuth + m.pi/2
        self.elevation=3*m.pi/4
        return(angle_elevation,angle_azimuth)


    def mov_azimuth(self,delta_azimuth):
        self.azimuth = self.azimuth + delta_azimuth
        return self.azimuth

    def mov_elevation(self,delta_elevation):
        self.elevation = self.elevation + delta_elevation
        return self.elevation


    def get_random_sample(self,nbr_view,landmark):
        list_sample = []
        for i in range(nbr_view):
            self.set_random_rotation()
            images = self.get_view()
            target = self.get_best_mouv()
            list_sample.append({'image':images})
        return list_sample

    def plot_fig(self):
        radius_sphere = 0.1
        R, T = look_at_view_transform(self.distance, self.elevation, self.azimuth, device=self.device,degrees=False)
        cam_pos=torch.matmul(T,R)
        print(cam_pos)
        cam_pos = cam_pos.numpy()[0][0]
        cam_mesh = generate_sphere_mesh(cam_pos,radius_sphere,self.device)
        center_mesh = generate_sphere_mesh([0,0,0],radius_sphere,self.device)

        dic = {"teeth_mesh": self.mesh,'cam_mesh':cam_mesh,'center':center_mesh}
        for n,lm_mesh in enumerate(self.list_meshe_landmarks):
            dic[str(n)] = lm_mesh
        fig = plot_scene({"subplot1": dic},     
            xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
            yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
            zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
            axis_args=AxisArgs(showgrid=True))
            
        fig.show()
    
    def  generate_landmark_meshes(self,radius_sphere):
        # Create a sphere
        list_meshe_landmarks = []
        for key,value in self.dic_landmarks.items():
            landmark_mesh = generate_sphere_mesh(value,radius_sphere,self.device)
            list_meshe_landmarks.append(landmark_mesh)
        
        self.list_meshe_landmarks = list_meshe_landmarks

    # def get_best_mouv(self,landmarks_id):
    #     angle_elevation,angle_azimuth = self.get_landmarks_angle(landmarks_id)

