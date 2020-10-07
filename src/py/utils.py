import vtk
import LinearSubdivisionFilter as lsf
import numpy as np

def Normalization(vtkdata):
	polypoints = vtkdata.GetPoints()
	
	nppoints = []
	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = polypoints.GetPoint(pid)
		nppoints.append(spoint)

	npmean = np.mean(np.array(nppoints), axis=0)
	nppoints -= npmean
	npscale = np.max([np.linalg.norm(p) for p in nppoints])
	nppoints /= npscale

	for pid in range(polypoints.GetNumberOfPoints()):
		vtkdata.GetPoints().SetPoint(pid, nppoints[pid])

	return vtkdata, npmean, npscale

def normalize_points(poly, radius):
	polypoints = poly.GetPoints()
	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = polypoints.GetPoint(pid)
		spoint = np.array(spoint)
		norm = np.linalg.norm(spoint)
		spoint = spoint/norm * radius
		polypoints.SetPoint(pid, spoint)
	poly.SetPoints(polypoints)
	return poly

def normalize_vector(x):
	return x/np.linalg.norm(x)

def CreateIcosahedron(radius, sl):
	icosahedronsource = vtk.vtkPlatonicSolidSource()
	icosahedronsource.SetSolidTypeToIcosahedron()
	icosahedronsource.Update()
	icosahedron = icosahedronsource.GetOutput()
	
	subdivfilter = lsf.LinearSubdivisionFilter()
	subdivfilter.SetInputData(icosahedron)
	subdivfilter.SetNumberOfSubdivisions(sl)
	subdivfilter.Update()

	icosahedron = subdivfilter.GetOutput()
	icosahedron = normalize_points(icosahedron, radius)

	return icosahedron

def CreatePlane(Origin,Point1,Point2,Resolution):
	plane = vtk.vtkPlaneSource()
	
	plane.SetOrigin(Origin)
	plane.SetPoint1(Point1)
	plane.SetPoint2(Point2)
	plane.SetXResolution(Resolution)
	plane.SetYResolution(Resolution)
	plane.Update()
	return plane.GetOutput()