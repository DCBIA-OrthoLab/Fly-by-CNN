import os
import re
import tensorflow as tf
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
	

class FlyByGenerator():
	def __init__(self, sphere, resolution, visualize=False):

		renderer = vtk.vtkRenderer()
		renderWindow = vtk.vtkRenderWindow()
		renderWindow.AddRenderer(renderer)
		renderWindow.SetSize(resolution, resolution)
		renderWindow.OffScreenRenderingOn()

		self.renderer = renderer
		self.renderWindow = renderWindow
		self.sphere = sphere
		self.visualize = visualize

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
			# self.renderWindow.Render()

			windowToImageN = vtk.vtkWindowToImageFilter()
			windowToImageN.SetInputBufferTypeToRGB()
			windowToImageN.SetInput(self.renderWindow)
			windowToImageN.Update()

			windowFilterZ = vtk.vtkWindowToImageFilter()
			windowFilterZ.SetInputBufferTypeToZBuffer()
			windowFilterZ.SetInput(self.renderWindow)

			scalez = vtk.vtkImageShiftScale()
			scalez.SetOutputScalarTypeToDouble();
			scalez.SetInputConnection(windowFilterZ.GetOutputPort());
			scalez.SetShift(0);
			scalez.SetScale(-1);
			scalez.Update()
			
			img_o = windowToImageN.GetOutput()
			img_z = scalez.GetOutput()
			
			img_o_np = vtk_to_numpy(img_o.GetPointData().GetScalars())
			scalez_np = vtk_to_numpy(img_z.GetPointData().GetScalars())
			scalez_np = np.abs(scalez_np).reshape([-1, 1])

			img_np = np.multiply(img_o_np, scalez_np)

			img_seq.append(img_np.reshape([d for d in img_o.GetDimensions() if d != 1] + [img_o.GetNumberOfScalarComponents()]))

		return np.array(img_seq)

def main(args):

	filenames = []

	if(args.surf):
		fobj = {}
		fobj["surf"] = args.surf
		fobj["out"] = args.out
		fobj["norm_shader"] = args.norm_shader
		filenames.append(fobj)
			
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

	model = None
	if args.model is not None:
		model = tf.keras.models.load_model(args.model, custom_objects={'tf': tf})
		model.summary()

	flyby = FlyByGenerator(sphere, args.resolution, args.visualize)

	for fobj in filenames:

		surf_actor = GetUnitActor(fobj["surf"], args.property, args.random_rotation, fobj["norm_shader"])
		
		if surf_actor is not None:
			flyby.addActor(surf_actor)
			
			out_np = flyby.getFlyBy()

			if model is not None:
				out_np = model.predict(out_np)
			
			if ( not args.concatenate ):
				for i in range(out_np.shape[0]):
					out_img = GetImage(out_np[i])

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
	input_group.add_argument('--norm_shader', type=int, help='1 to color surface with normal shader, 0 to color with look up table',default = 1)
	input_group.add_argument('--property', type=str, help='Input property file with same number of points as "surf"')

	sphere_params = parser.add_argument_group('Sampling parameters')
	sphere_params_sampling = sphere_params.add_mutually_exclusive_group(required=True)
	sphere_params_sampling.add_argument('--subdivision', type=int, help='Number of subdivisions for icosahedron')
	sphere_params_sampling.add_argument('--spiral', type=int, help='Number of samples along the spherical spiral')

	sphere_params.add_argument('--turns', type=int, default=4, help='Number of spiral turns')
	sphere_params.add_argument('--resolution', type=int, help='Image resolution', default=256)
	sphere_params.add_argument('--radius', type=float, help='Radius of the sphere for the view points', default=4)

	visu_params = parser.add_argument_group('Visualize')
	visu_params.add_argument('--visualize', type=int, default=0, help='Visualize the sampling')

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--out', type=str, help='Output filename or directory', default="out.nrrd")
	output_params.add_argument('--uuid', type=bool, help='Use uuid to name the outputs', default=False)
	output_params.add_argument('--ow', type=int, help='Overwrite outputs', default=1)
	output_params.add_argument('--concatenate', type=int, help='0 for multiple output files, 1 for single output file', default=1)


	args = parser.parse_args()

	main(args)
