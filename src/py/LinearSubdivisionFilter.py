import vtk
import numpy as np

class LinearSubdivisionFilter:

	InputData = None
	Output = None
	NumberOfSubdivisions = 1

	def SetInputData(self, polydata):
		self.InputData = polydata

	def GetOutput(self):
		return self.Output

	def SetNumberOfSubdivisions (self, subdivisions):
		self.NumberOfSubdivisions = subdivisions

	def Update(self):

		self.GenerateData()

	def GenerateData(self):

		if self.InputData:

			inputpolydata = self.InputData
			subdivisionlevel = self.NumberOfSubdivisions
			inputpolydata_points = inputpolydata.GetPoints()
			appendpoly = vtk.vtkAppendPolyData()

			# Iterate over the cells in the polydata
			# The idea is to linearly divide every cell according to the subdivision level
			for cellid in range(inputpolydata.GetNumberOfCells()):
				idlist = vtk.vtkIdList()
				inputpolydata.GetCellPoints(cellid, idlist)
				
				# For every cell we create a new poly data, i.e, bigger triangle with the interpolated triangles inside
				subdiv_poly = vtk.vtkPolyData()
				subdiv_points = vtk.vtkPoints()
				subdiv_cellarray = vtk.vtkCellArray()
				
				if(idlist.GetNumberOfIds() != 3):
					raise Exception("Only triangle meshes are supported. Convert your mesh to triangles!", idlist.GetNumberOfIds())

				# Get the triangle points from the current cell
				p1 = np.array(inputpolydata_points.GetPoint(idlist.GetId(0)))
				p2 = np.array(inputpolydata_points.GetPoint(idlist.GetId(1)))
				p3 = np.array(inputpolydata_points.GetPoint(idlist.GetId(2)))

				# Calculate the derivatives according to the level
				dp12 = (p2 - p1)/subdivisionlevel
				dp13 = (p3 - p1)/subdivisionlevel
				
				# Interpolate the points
				for s13 in range(0, subdivisionlevel + 1):
					for s12 in range(0, subdivisionlevel + 1 - s13):
						interp = p1 + s12*dp12 + s13*dp13
						subdiv_points.InsertNextPoint(interp[0], interp[1], interp[2])

				# Using the interpolated points, create the cells, i.e., triangles
				id1 = -1
				for s13 in range(0, subdivisionlevel):
					id1 += 1
					for s12 in range(0, subdivisionlevel - s13):
						
						id2 = id1 + 1
						id3 = id1 + subdivisionlevel + 1 - s13
						id4 = id3 + 1

						triangle = vtk.vtkTriangle()
						triangle.GetPointIds().SetId(0, id1);
						triangle.GetPointIds().SetId(1, id2);
						triangle.GetPointIds().SetId(2, id3);

						subdiv_cellarray.InsertNextCell(triangle)

						if s12 < subdivisionlevel - s13 - 1:
							triangle = vtk.vtkTriangle()
							triangle.GetPointIds().SetId(0, id2);
							triangle.GetPointIds().SetId(1, id4);
							triangle.GetPointIds().SetId(2, id3);
							subdiv_cellarray.InsertNextCell(triangle)

						id1 += 1
				
				#Set all the interpolated points and generated cells to the polydata
				subdiv_poly.SetPoints(subdiv_points)
				subdiv_poly.SetPolys(subdiv_cellarray)
				# Append the current interpolated triangle to the 'appendPolyDataFilter'
				appendpoly.AddInputData(subdiv_poly)

			# All interpolated triangles now from a single polydata
			appendpoly.Update()

			# Remove duplicate points (if you were paying attention, you know there are a lot of repetitions in every triangle edge)
			cleanpoly = vtk.vtkCleanPolyData()
			cleanpoly.SetInputData(appendpoly.GetOutput())
			cleanpoly.Update()

			# Return the subdivied polydata
			self.Output = cleanpoly.GetOutput()