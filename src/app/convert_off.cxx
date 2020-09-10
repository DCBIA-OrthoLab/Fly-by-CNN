#include "convert_offCLP.h"

#include "vtkOFFReader.h"

#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>


int main (int argc, char *argv[])
{
	PARSE_ARGS;
    
    if(inputSurface.compare("") == 0){
        cout<<"type --help to learn how to use this program."<<endl;
        return 1;
    }

    vtkSmartPointer<vtkOFFReader> reader = vtkSmartPointer<vtkOFFReader>::New();
    reader->SetFileName(inputSurface.c_str());
    reader->Update();
    vtkSmartPointer<vtkPolyData> surf = reader->GetOutput();  

    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    
    std::cout << "Writing: " << outputName << std::endl;
    writer->SetFileName(outputName.c_str());
    writer->SetInputData(surf);
    writer->Update();

	return EXIT_SUCCESS;
}
