#include "surfacefeaturesextractor.hxx"
#include "surfacefeaturesextractorCLP.h"

#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>

#include <iterator>


int main (int argc, char *argv[])
{
	PARSE_ARGS;
    
    if(inputMesh.compare("") == 0){
        cout<<"type --help to learn how to use this program."<<endl;
        return 1;
    }

    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    vtkSmartPointer<SurfaceFeaturesExtractor> Filter = vtkSmartPointer<SurfaceFeaturesExtractor>::New();
    vtkSmartPointer<vtkPolyData> crt_mesh = vtkSmartPointer<vtkPolyData>::New();

    std::vector< vtkSmartPointer<vtkPolyData> > distMeshList;
    std::vector< std::string> landmarkFile;


    reader->SetFileName(inputMesh.c_str());
    reader->Update();
    vtkSmartPointer<vtkPolyData> inputShape = reader->GetOutput();

    int num_points = inputShape->GetNumberOfPoints();
    if ( distMeshOn )
    {
        // Load each mesh used for distances 
        for (int k=0; k<distMesh.size(); k++) 
        {
            vtkSmartPointer<vtkPolyDataReader> readerMean = vtkSmartPointer<vtkPolyDataReader>::New();
            readerMean->SetFileName(distMesh[k].c_str());
            readerMean->Update();
            crt_mesh = readerMean->GetOutput();

            if (crt_mesh->GetNumberOfPoints() != num_points)
            {
                std::cerr << "All the shapes must have the same number of points" << std::endl;
                return EXIT_FAILURE;
            } 
            distMeshList.push_back(crt_mesh);
        }
    }


    if ( landmarksOn )
        landmarkFile.push_back(landmarks);

	Filter->SetInput(inputShape, distMeshList, landmarkFile);
    Filter->Update();

    // std::cout << "outputMesh: " << outputMesh << std::endl;
    writer->SetFileName(outputMesh.c_str());
    writer->SetInputData(Filter->GetOutput());
    writer->Update();

	return EXIT_SUCCESS;
}
