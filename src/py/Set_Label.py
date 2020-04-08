import subdivision
import vtk
import numpy as np
import nrrd
import time
import LinearSubdivisionFilter as lsf
import itk

def Label_Set(vtkdata, model_output, IDlist, label_array) :
	j = -1
	for i in range (len(IDlist)) :
		j = j + 1
		if j == len(IDlist)-1 :
			break
		# print('before : ', int(IDlist[i]),'\n')
		while(int(IDlist[j]) == -1):
			j = j + 1
			if j ==len(IDlist)-1 :
				break
		if j == len(IDlist)-1 :
				break
		mesh_id = int(IDlist[j])
		current_array_label = label_array.GetTuple(mesh_id)

		if current_array_label == (-1, -1, -1) :
			new_mesh_label = tuple(model_output[j]) #mesh_id
			
		else :	
			new_mesh_label = tuple(np.array(current_array_label) + model_output[j]) #mesh_id

		label_array.SetTuple(mesh_id, new_mesh_label)

	return label_array


def Set_Real_Label(label_array) :
	real_label = vtk.vtkIntArray()
	real_label.SetNumberOfComponents(1)
	real_label.SetNumberOfTuples(label_array.GetNumberOfTuples())
	real_label.Fill(-1)
	for i in range (label_array.GetNumberOfTuples()) :
		current_label = np.array(label_array.GetTuple(i))
		if tuple(current_label) != (-1, -1, -1) :
			argmax_label = np.argmax(current_label)
			# print(current_label)
			# Background or unknow label	
			if argmax_label ==  0 :
				new_label = -1
			# Gum
			if argmax_label ==  1 :
				new_label = 0
			#Teeth
			if argmax_label ==  2 :
				new_label = 1
				# print('yo les saucissons')
		# print('new_label : ', new_label)
			real_label.SetTuple(i, (new_label,))
	return real_label







