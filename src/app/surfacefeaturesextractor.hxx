#include "surfacefeaturesextractor.h"
#include <vtkDoubleArray.h>
#include <sstream>
#include <vtkObjectFactory.h>

#if !defined(M_PI)
#define M_PI 3.14159265358979323846264338327950288   /* pi */
#endif


vtkStandardNewMacro(SurfaceFeaturesExtractor);

/**
* Constructor SurfaceFeaturesExtractor()
*/
SurfaceFeaturesExtractor::SurfaceFeaturesExtractor(){
	this->inputSurface = vtkSmartPointer<vtkPolyData>::New();
	this->outputSurface = vtkSmartPointer<vtkPolyData>::New();

	this->intermediateSurface = vtkSmartPointer<vtkPolyData>::New();
}

/**
* Destructor SurfaceFeaturesExtractor()
*/
SurfaceFeaturesExtractor::~SurfaceFeaturesExtractor(){}

/**
* Function SetInput() for SurfaceFeaturesExtractor
*/
void SurfaceFeaturesExtractor::SetInput(vtkSmartPointer<vtkPolyData> input, std::vector< vtkSmartPointer<vtkPolyData> > list, std::vector<std::string> landmarkFile)
{
	this->meanShapesList = list;
	this->inputSurface = input;
	this->landmarkFile = landmarkFile;
}

/**
 * Function init_output() for SurfaceFeaturesExtractor
 */
void SurfaceFeaturesExtractor::init_output()
{
	this->intermediateSurface = this->inputSurface;
	this->outputSurface = this->inputSurface;
}

/** 
 * Function compute_normals()
 */
void SurfaceFeaturesExtractor::compute_normals()
{
	vtkSmartPointer<vtkPolyDataNormals> NormalFilter = vtkSmartPointer<vtkPolyDataNormals>::New();
	NormalFilter->SetInputData(this->intermediateSurface);

	NormalFilter->ComputePointNormalsOn();
    NormalFilter->ComputeCellNormalsOff();
    NormalFilter->SetFlipNormals(0);
    NormalFilter->SplittingOff();
	NormalFilter->FlipNormalsOff();
	NormalFilter->ConsistencyOff();

	NormalFilter->Update();
	this->intermediateSurface = NormalFilter->GetOutput();
}

/** 
 * Function compute_positions()
 */
void SurfaceFeaturesExtractor::compute_positions()
{
    std::string name = "Position";
    int nbPoints = this->intermediateSurface->GetNumberOfPoints();

	vtkSmartPointer<vtkFloatArray> position = vtkSmartPointer<vtkFloatArray>::New();
	position->SetNumberOfComponents(3);
	position->SetName(name.c_str());

	for(int i=0; i<nbPoints; i++)
	{
		double* p = new double[3];
		p = this->intermediateSurface->GetPoint(i);

		position->InsertNextTuple3(p[0],p[1],p[2]);

	}
	
	this->intermediateSurface->GetPointData()->SetActiveVectors(name.c_str());
	this->intermediateSurface->GetPointData()->SetVectors(position);

}

/** 
 * Function compute_distances()
 */
void SurfaceFeaturesExtractor::compute_distances()
{
	int nbPoints = this->intermediateSurface->GetNumberOfPoints();

	// Load each mean groupe shape & create labels
	std::vector<std::string> meanDistLabels;
	for (int k=0; k<this->meanShapesList.size(); k++) 
	{
		std::ostringstream k_temp;
    	k_temp << k;
    	std::string k_char = k_temp.str();
		// std::string k_char = static_cast<std::ostringstream*>( &( std::ostringstream() << k) )->str();
		meanDistLabels.push_back("distanceGroup"+k_char);
	}
	//for (int k=0; k<this->meanShapesList.size(); k++) 
	//	std::cout<<meanDistLabels[k]<<std::endl;

	for(int k=0; k<meanShapesList.size(); k++)
	{
		vtkSmartPointer<vtkFloatArray> meanDistance = vtkSmartPointer<vtkFloatArray>::New() ;
		meanDistance->SetName(meanDistLabels[k].c_str());

		for (int i=0; i<nbPoints; i++)
		{
			double* p1 = new double[3];
			p1 = this->intermediateSurface->GetPoint(i);
			
			double* p2 = new double[3];
			p2 = meanShapesList[k]->GetPoint(i);

			double dist = sqrt( (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]) );

			meanDistance->InsertNextTuple1(dist);

			this->intermediateSurface->GetPointData()->SetActiveScalars(meanDistLabels[k].c_str());
			this->intermediateSurface->GetPointData()->SetScalars(meanDistance);
			
		}
	}
}


void SurfaceFeaturesExtractor::compute_maxcurvatures()		// Kappa2
{
	vtkSmartPointer<vtkCurvatures> curvaturesFilter = vtkSmartPointer<vtkCurvatures>::New();

	curvaturesFilter->SetInputDataObject(this->intermediateSurface);
	curvaturesFilter->SetCurvatureTypeToMaximum();
	curvaturesFilter->Update();

	this->intermediateSurface = curvaturesFilter->GetOutput();
}
void SurfaceFeaturesExtractor::compute_mincurvatures()		// Kappa1
{
	vtkSmartPointer<vtkCurvatures> curvaturesFilter = vtkSmartPointer<vtkCurvatures>::New();

	curvaturesFilter->SetInputDataObject(this->intermediateSurface);
	curvaturesFilter->SetCurvatureTypeToMinimum();
	curvaturesFilter->Update();

	this->intermediateSurface = curvaturesFilter->GetOutput();	
}
void SurfaceFeaturesExtractor::compute_gaussiancurvatures()	// G
{
	vtkSmartPointer<vtkCurvatures> curvaturesFilter = vtkSmartPointer<vtkCurvatures>::New();

	curvaturesFilter->SetInputDataObject(this->intermediateSurface);
	curvaturesFilter->SetCurvatureTypeToGaussian();
	curvaturesFilter->Update();

	this->intermediateSurface = curvaturesFilter->GetOutput();
}
void SurfaceFeaturesExtractor::compute_meancurvatures()		// H
{
	vtkSmartPointer<vtkCurvatures> curvaturesFilter = vtkSmartPointer<vtkCurvatures>::New();

	curvaturesFilter->SetInputDataObject(this->intermediateSurface);
	curvaturesFilter->SetCurvatureTypeToMean();
	curvaturesFilter->Update();

	this->intermediateSurface = curvaturesFilter->GetOutput();
}

void SurfaceFeaturesExtractor::compute_shapeindex()			// S
{
	// std::cout<<" :: Function compute_shapeindex"<<std::endl;

	int nbPoints = this->intermediateSurface->GetNumberOfPoints();

	vtkSmartPointer<vtkFloatArray> shapeIndexArray = vtkSmartPointer<vtkFloatArray>::New() ;

	// vtkDataArray* minCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Minimum_Curvature");
	// vtkDataArray* maxCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Maximum_Curvature");

	vtkSmartPointer<vtkDataArray> minCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Minimum_Curvature");
	vtkSmartPointer<vtkDataArray> maxCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Maximum_Curvature");

	shapeIndexArray->SetName("Shape_Index");

	for (int i=0; i<nbPoints; i++)
	{
		double k1 = minCurvArray->GetTuple1(i);
		double k2 = maxCurvArray->GetTuple1(i);
		
		double value = (2 / M_PI) * (atan( (k2 + k1) / (k2 - k1) ) );
		if( value != value )
        	value = 0;

		shapeIndexArray->InsertNextTuple1(value);
	}

	this->intermediateSurface->GetPointData()->SetActiveScalars("Shape_Index");
	this->intermediateSurface->GetPointData()->SetScalars(shapeIndexArray);
}

void SurfaceFeaturesExtractor::compute_curvedness()			// C
{
	// std::cout<<" :: Function compute_curvedness"<<std::endl;

	int nbPoints = this->intermediateSurface->GetNumberOfPoints();

	vtkSmartPointer<vtkFloatArray> curvednessArray = vtkSmartPointer<vtkFloatArray>::New() ;

	// vtkDataArray* minCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Minimum_Curvature");
	// vtkDataArray* maxCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Maximum_Curvature");

	vtkSmartPointer<vtkDataArray> minCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Minimum_Curvature");
	vtkSmartPointer<vtkDataArray> maxCurvArray = this->intermediateSurface->GetPointData()->GetScalars("Maximum_Curvature");

	curvednessArray->SetName("Curvedness");

	for (int i=0; i<nbPoints; i++)
	{
		double k1 = minCurvArray->GetTuple1(i);
		double k2 = maxCurvArray->GetTuple1(i);
		
		double value = sqrt( (pow(k1, 2) + pow(k2, 2) ) / 2);

		curvednessArray->InsertNextTuple1(value);

		this->intermediateSurface->GetPointData()->SetActiveScalars("Curvedness");
		this->intermediateSurface->GetPointData()->SetScalars(curvednessArray);
	}

}

void SurfaceFeaturesExtractor::scalar_indexPoint()
{
	int nbPoints = this->intermediateSurface->GetNumberOfPoints();

	vtkSmartPointer<vtkFloatArray> indexPointArray = vtkSmartPointer<vtkFloatArray>::New() ;

	indexPointArray->SetName("Index_Points");

	for (int i=0; i<nbPoints; i++)
	{
		indexPointArray->InsertNextTuple1(i);

		this->intermediateSurface->GetPointData()->SetActiveScalars("Index_Points");
		this->intermediateSurface->GetPointData()->SetScalars(indexPointArray);
	}
}

void SurfaceFeaturesExtractor::store_landmarks_vtk()
{
	//std::cout << " Functions store landmarks_vtk " << std::endl;

	// Build a locator
	vtkSmartPointer<vtkPointLocator> pointLocator = vtkPointLocator::New();
	pointLocator->SetDataSet(this->intermediateSurface);
	pointLocator->BuildLocator();


	// ---------- Reading FCSV file ----------

	#define NB_LINES 250
	#define NB_WORDS 250

	// Get the Surface filename from the command line
	std::fstream fcsvfile(this->landmarkFile[0].c_str());
	std::string line, mot;
	std::string words[NB_LINES][NB_WORDS]; // !!!! WARNING DEFINE AND TO PROTECT IF SUPERIOR TO 20
	int i,j;
	int* landmarkPids; 
	int NbLandmarks;

	if(fcsvfile)
	{
		getline(fcsvfile, line);
		fcsvfile>>mot;
		while(mot=="#")
		{
			if(getline(fcsvfile, line))
				fcsvfile>>mot;
			else
				mot="#";
		}

		i=0;
		do
		{
			std::size_t pos_end, pos1;
			j=0;
			do
			{
				std::size_t pos0 = 0;
				pos1 = mot.find(',');
				pos_end = mot.find(",,");
				words[i][j] = mot.substr(pos0, pos1-pos0);
				mot = mot.substr(pos1+1);
				j++;
			}
			while(pos1+1<pos_end);
			i++;
		}
		while(fcsvfile>>mot);

		NbLandmarks = i;
		landmarkPids = new int[NbLandmarks]; 

		for (int i = 0; i < NbLandmarks; ++i)
		{
			double x = atof(words[i][1].c_str());
			double y = atof(words[i][2].c_str());
			double z = atof(words[i][3].c_str());
                        
            // Find closest point
            vtkIdType ptId;
            double p[] = {0.0, 0.0, 0.0};
            p[0] = x; p[1] = y; p[2] = z;
            ptId = pointLocator->FindClosestPoint(p);
            landmarkPids[i] = ptId;

		// std::cout << "landmark " << i << " position " << x << "," << y << "," << z << " and the corresponding Pid is " << landmarkPids[i] << std::endl;
		}
	}
	else
		std::cout<<"Error !";


	// ---------- Encode landmarks in  FCSV file ----------
	vtkSmartPointer<vtkDoubleArray> landmarksArray = vtkSmartPointer<vtkDoubleArray>::New();
	landmarksArray->SetName("Landmarks");
	landmarksArray->SetNumberOfComponents(1);

	for(int ID = 0; ID < this->intermediateSurface->GetNumberOfPoints(); ID++)
	{
		double exists = 0.0;
		for (int i = 0; i < NbLandmarks; ++i)
		{
			double diff = landmarkPids[i] - ID;
			if (diff == 0)
			{
				exists = i+1;
				//std::cout << "Landmark ID " << exists << std::endl;
				break;
			} 
		}
		landmarksArray->InsertNextValue(exists);
	}
	landmarksArray->Resize(this->intermediateSurface->GetNumberOfPoints());
	this->intermediateSurface->GetPointData()->AddArray(landmarksArray);

	//delete[] landmarkPids;

}


/**
 * Function Update()
 */
void SurfaceFeaturesExtractor::Update()
{
	this->init_output();

	// Compute normal for each vertex
	this->compute_normals();

	// Compute position of each point
	this->compute_positions();

	// Compute distance to each mean groups
	if (this->meanShapesList.size() > 0)
		this->compute_distances();

	// Compute curvatures at each point
	this->compute_maxcurvatures();
	this->compute_mincurvatures();
	this->compute_gaussiancurvatures();
	this->compute_meancurvatures();

	// Shape Index and Curvedness
	this->compute_shapeindex();
	this->compute_curvedness();
	

	// this->scalar_indexPoint();

	if (this->landmarkFile.size() == 1)
		this->store_landmarks_vtk();

	this->outputSurface = this->intermediateSurface;
}

/**
 * Function GetOutput()
 */
vtkSmartPointer<vtkPolyData> SurfaceFeaturesExtractor::GetOutput()
{
	return this->outputSurface;
}


