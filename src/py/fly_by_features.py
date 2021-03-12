import os
import re
import numpy as np
import itk
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import argparse
import time
import glob
import pandas
import uuid 

import LinearSubdivisionFilter as lsf
from utils import * 
from scaling import *

# class ShapeAlignment(tf.keras.Model):

# 	def __init__(self, dimension=2):
# 		super(ShapeAlignment, self).__init__()

# 		self.transform = tf.Variable(tf.eye(4), dtype=tf.float32)
# 		self.loss = tf.keras.losses.MeanAbsoluteError()

# 		# lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 100, 0.96, 1)
# 		self.optimizer = tf.keras.optimizers.SGD(1e-3)

# 	# @tf.function
# 	def train_step(self, logits, target):

# 		with tf.GradientTape() as tape:
			
# 			loss = self.loss(logits, target)
			
# 			var_list = self.trainable_variables

# 			gradients = tape.gradient(loss, var_list)
# 			self.optimizer.apply_gradients(zip(gradients, var_list))

# 			return loss

# 	def get_transform(self):
# 		return self.transform.get_weights()
	
# from orientation import *

class FlyByGenerator():
	def __init__(self, sphere, resolution, visualize=False, use_z=False, split_z=False):

		renderer = vtk.vtkRenderer()
		renderWindow = vtk.vtkRenderWindow()
		renderWindow.AddRenderer(renderer)		
		renderWindow.SetSize(resolution, resolution)
		renderWindow.SetMultiSamples(0)
		renderWindow.OffScreenRenderingOn()

		self.renderer = renderer
		self.renderWindow = renderWindow
		self.sphere = sphere
		self.visualize = visualize
		self.resolution = resolution
		self.use_z = use_z
		self.split_z = split_z

		#sphere_actor = GetActor(sphere)
		# print(sphere_actor)
		#sphere_actor.GetProperty().SetRepresentationToWireFrame()
		#self.renderer.AddActor(sphere_actor)

	def removeActor(self, actor):
		self.renderer.RemoveActor(actor)

	def removeActors(self):
		actors = self.renderer.GetActors()
		actors.InitTraversal()
		for i in range(actors.GetNumberOfItems()):
			self.renderer.RemoveActor(actors.GetNextActor())

	def addActor(self, actor):
		self.renderer.AddActor(actor)

	def getFlyBy(self):

		sphere_points = self.sphere.GetPoints()
		print(sphere_points.GetNumberOfPoints())
		camera = self.renderer.GetActiveCamera()

		if self.visualize:
			self.renderWindow.OffScreenRenderingOff()
			interactor = vtk.vtkRenderWindowInteractor()
			interactor.SetRenderWindow(self.renderWindow)
			interactor.Initialize()
			interactor.Start()

		img_seq = []

		for i in range(sphere_points.GetNumberOfPoints()):

			sphere_point = sphere_points.GetPoint(i)
			sphere_point_v = normalize_vector(sphere_point)

			if(abs(sphere_point_v[2]) != 1):
				camera.SetViewUp(0, 0, -1)
			elif(sphere_point_v[2] == 1):
				camera.SetViewUp(1, 0, 0)
			elif(sphere_point_v[2] == -1):
				camera.SetViewUp(-1, 0, 0)

			camera.SetPosition(sphere_point[0], sphere_point[1], sphere_point[2])
			camera.SetFocalPoint(0, 0, 0)

			self.renderer.ResetCameraClippingRange()

			windowToImageN = vtk.vtkWindowToImageFilter()
			windowToImageN.SetInputBufferTypeToRGB()
			windowToImageN.SetInput(self.renderWindow)
			windowToImageN.Update()

			img_o = windowToImageN.GetOutput()

			img_o_np = vtk_to_numpy(img_o.GetPointData().GetScalars())
			num_components = img_o.GetNumberOfScalarComponents()

			if self.use_z:
				windowFilterZ = vtk.vtkWindowToImageFilter()
				windowFilterZ.SetInputBufferTypeToZBuffer()
				windowFilterZ.SetInput(self.renderWindow)
				windowFilterZ.SetScale(1)

				windowFilterZ.Update()				
				img_z = windowFilterZ.GetOutput()
				
				img_z_np = vtk_to_numpy(img_z.GetPointData().GetScalars())
				img_z_np = img_z_np.reshape([-1, 1])

				z_near, z_far = camera.GetClippingRange()

				img_z_np = 2.0*z_far*z_near / (z_far + z_near - (z_far - z_near)*(2.0*img_z_np - 1.0))
				img_z_np[img_z_np > (z_far - 0.1)] = 0

				if(self.split_z):
					img_np = np.concatenate([img_o_np, img_z_np], axis=-1)
					num_components += 1
				else:
					img_z_np /= z_far
					img_np = np.multiply(img_o_np, img_z_np)
			else:
				img_np = img_o_np

			img_seq.append(img_np.reshape([d for d in img_o.GetDimensions() if d != 1] + [num_components]))

		return np.array(img_seq)

def main(args):

	filenames = []

	if(args.surf):
		fobj = {}
		fobj["surf"] = args.surf
		fobj["out"] = args.out
		filenames.append(fobj)
		print(filenames)
			
	else:
		surf_filenames = []

		if(args.dir):
			replace_dir_name = args.dir
			normpath = os.path.normpath("/".join([args.dir, '**', '*']))
			for surf in glob.iglob(normpath, recursive=True):
				if os.path.isfile(surf) and True in [ext in surf for ext in [".vtk", ".obj", ".stl"]]:
					surf_filenames.append(os.path.realpath(surf))

		elif(args.csv):
			replace_dir_name = args.csv_root_path
			with open(args.csv) as csvfile:
				df = pandas.read_csv(csvfile)

			for index, row in df.iterrows():
				surf_filenames.append(row["surf"])

		for surf in surf_filenames:
			fobj = {}
			fobj["surf"] = surf

			# don't want n output for n files
			# if(not args.fiberBundle):
			dir_filename = os.path.splitext(surf.replace(replace_dir_name, ''))[0] +  ".nrrd"
			fobj["out"] = os.path.normpath("/".join([args.out, dir_filename]))
	
			if(args.uuid):
				fobj["out"] = fobj["out"].replace(".nrrd", "-" + str(uuid.uuid1()).split('-')[0] + ".nrrd")

			if not os.path.exists(os.path.dirname(fobj["out"])):
				os.makedirs(os.path.dirname(fobj["out"]))
			
			if args.ow or not os.path.exists(fobj["out"]):
				filenames.append(fobj)


	if(args.subdivision):
		sphere = CreateIcosahedron(args.radius, args.subdivision)
	else:
		sphere = CreateSpiral(args.radius, args.spiral, args.turns)

	# model = None
	# if args.model is not None:
		# model = tf.keras.models.load_model(args.model, custom_objects={'tf': tf})
		# model.summary()

	flyby = FlyByGenerator(sphere, args.resolution, args.visualize, use_z=args.use_z, split_z=args.split_z)
	
	if args.point_features:
		flyby_features = FlyByGenerator(sphere, args.resolution, visualize=args.visualize)




	for fobj in filenames:

		surf = ReadSurf(fobj["surf"])
		
		surf = GetUnitSurf(surf, args.scaling, args.translate, args.shape)
		
		if args.fiberBundle:
			nbCell = surf.GetNumberOfCells()
			nbFiber = int(nbCell*args.nbFiber)
			print("number of fibers:", nbCell, "number of exctracted fiber:", nbFiber)
			list_random_id = np.random.default_rng().choice(nbCell, size=nbFiber, replace=False)
			for i_cell in list_random_id:
				fiber_surf = ExtractFiber(surf, i_cell)

				surf_actor = GetNormalsActor(fiber_surf)

				if surf_actor:

					flyby.addActor(surf_actor)
					
					out_np = flyby.getFlyBy()

					# if model is not None:
					# 	out_np = model.predict(out_np)

					out_img = GetImage(out_np)

					fiberName = "fiber_" + str(i_cell) + ".nrrd"
					path_file = os.path.normpath("/".join([args.out, fiberName]))

					print("Writing:", path_file)
					writer = itk.ImageFileWriter.New(FileName=path_file, Input=out_img)
					writer.UseCompressionOn()
					writer.Update()

					flyby.removeActor(surf_actor)
		else:


			if args.random_rotation:
				surf = RandomRotation(surf)

			if args.fiber :
				surf = GetTubeFilter(surf)

			if args.save_label:
				# surf = OrientLabel_vector(surf, args.save_label)
				surf = OrientLabel(surf, flyby.sphere, args.save_label, args.save_AA)

			if args.property:
				surf_actor = GetPropertyActor(surf, args.property)
			else:
				surf_actor = GetNormalsActor(surf)
		
#Split GetUnitActor function into 3 functions to make it more streamline: Read Surface, Rotate Surface, GetColorIdMap(apply property), GetNormalsActor(normal vector displayh)
		
			if surf_actor is not None:
				flyby.addActor(surf_actor)
				
			out_np = flyby.getFlyBy()

			# if model is not None:
			# 	out_np = model.predict(out_np)
		

			if ( not args.concatenate ):
				for i in range(out_np.shape[0]):
					out_img = GetImage(out_np[i])

	                #use os.path.splitext(fobj["out"])[0] <- parse the file name
					p = ".*(?=\.)"
					prefix = re.findall(p, fobj["out"])
					
					s ="([^\.]+$)" 
					suffix = re.findall(s, fobj["out"])

					filename=prefix[0]+"_"+str(i)+"."+suffix[0]
					print("Writing:", filename)
			
					writer = itk.ImageFileWriter.New(FileName=filename, Input=out_img)
					writer.UseCompressionOn()
					writer.Update()
			else:
				out_img = GetImage(out_np)

				print("Writing:", fobj["out"])
				writer = itk.ImageFileWriter.New(FileName=fobj["out"], Input=out_img)
				writer.UseCompressionOn()
				writer.Update()

			if args.point_features:
				surf_actor = GetPointIdMapActor(surf)
				flyby_features.addActor(surf_actor)
				out_pointids_rgb_np = flyby_features.getFlyBy(False)

				for point_features_name in args.point_features:
					print("Extracting:", point_features_name)
					out_features_np = ExtractPointFeatures(surf, out_pointids_rgb_np, point_features_name)
					out_features_name = os.path.splitext(fobj["out"])
					out_features_name = out_features_name[0] + "_" + point_features_name + out_features_name[1]
					print("Writing:", out_features_name)
					out_features = GetImage(out_features_np)
					writer = itk.ImageFileWriter.New(FileName=out_features_name, Input=out_features)
					writer.UseCompressionOn()
					writer.Update()

			flyby.removeActors()


if __name__ == '__main__':
	start_time = time.time()

	parser = argparse.ArgumentParser(description='Predict an input with a trained neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_group = parser.add_argument_group('Input parameters')
	input_params = input_group.add_mutually_exclusive_group(required=True)
	input_params.add_argument('--surf', type=str, help='Target surface/mesh')
	input_params.add_argument('--dir', type=str, help='Input directory with 3D models')
	input_params.add_argument('--csv', type=str, help='Input csv with column "surf"')

	input_group.add_argument('--csv_root_path', type=str, help='CSV rooth path for replacement', default="")
	input_group.add_argument('--model', type=str, help='Directory with saved model', default=None)
	input_group.add_argument('--random_rotation', type=bool, help='Apply a random rotation', default=False)

	input_group.add_argument('--fiberBundle', type=bool, help='If input directory is a fiber tract', default=False)
	input_group.add_argument('--fiber', type=bool, help='If input surface is a fiber', default=False)
	input_group.add_argument('--nbFiber', type=float, help='extract a percentage of fibers per cluster (value between 0 and 1)', default=0.05)
	input_group.add_argument('--scaling', type=float, help='Scaling factor computed with scaling.py', default=None)
	input_group.add_argument('--translate', nargs="+", type=float, help='Translation vector computed with scaling.py', default=None)
	input_group.add_argument('--shape', nargs="+", type=int, help='Shape of the matrix computed with scaling.py', default=None)

	# input_group.add_argument('--norm_shader', type=int, help='1 to color surface with normal shader, 0 to color with look up table',default = 1)
	# input_group.add_argument('--use_z', type=int, help='1 to scale normals by z buffer',default = 1)
	# input_group.add_argument('--property', type=str, help='Input property file with same number of points as "surf"', default=None)
	# input_group.add_argument('--point_features', nargs='+', type=str, help='Name of array in point data to extract features', default=None)
	# input_group.add_argument('--scale_factor', type=float, help='Scale the surface by this vale', default= -1)


	features_group = parser.add_argument_group('Shape features/property to extract')
	features_group.add_argument('--norm_shader', type=int, help='1 to color surface with normal shader, 0 to color with look up table', default = 1)
	features_group.add_argument('--split_z', type=int, help='1 to split the z buffer as a separate channel. Otherwise, the normals are scaled by z buffer to create an rgb image.', default = 0)
	features_group.add_argument('--use_z', type=int, help='1 to to use the z_buffer and compute the depth buffer (distance of camera to shape at every location).', default = 1)

	features_group.add_argument('--property', type=str, help='Input property file with same number of points as "surf"', default=None)
	features_group.add_argument('--point_features', nargs='+', type=str, help='Name of array in point data to extract features', default=None)
	features_group.add_argument('--zero', type=float, help="Default zero value when extracting properties. This is used when there is no 'collision' with the surface", default=0)

	sphere_params = parser.add_argument_group('Sampling parameters')
	sphere_params_sampling = sphere_params.add_mutually_exclusive_group(required=True)
	sphere_params_sampling.add_argument('--subdivision', type=int, help='Number of subdivisions for icosahedron')
	sphere_params_sampling.add_argument('--spiral', type=int, help='Number of samples along the spherical spiral')

	sphere_params.add_argument('--turns', type=int, default=4, help='Number of spiral turns')
	sphere_params.add_argument('--resolution', type=int, help='Image resolution', default=256)
	sphere_params.add_argument('--radius', type=float, help='Radius of the sphere for the view points', default=4)

	training_orientation = parser.add_argument_group('training orientation')
	training_orientation.add_argument('--save_label', type=str, help='save the label', default="")
	training_orientation.add_argument('--save_AA', type=str, help='save the label', default="")

	visu_params = parser.add_argument_group('Visualize')
	visu_params.add_argument('--visualize', type=int, default=0, help='Visualize the sampling')

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--out', type=str, help='Output filename or directory', default="out.nrrd")
	output_params.add_argument('--uuid', type=bool, help='Use uuid to name the outputs', default=False)
	output_params.add_argument('--ow', type=int, help='Overwrite outputs', default=1)
	output_params.add_argument('--concatenate', type=int, help='0 for multiple output files, 1 for single output file', default=1)

	args = parser.parse_args()

	main(args)
