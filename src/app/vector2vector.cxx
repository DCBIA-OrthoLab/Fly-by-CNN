
#include "vector2vectorCLP.h"

#include <itkVectorImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkComposeImageFilter.h>

using namespace std;

int main (int argc, char * argv[]){
    PARSE_ARGS;


    if(inputImageFilename.compare("") == 0){
        commandLine.getOutput()->usage(commandLine);
        return EXIT_FAILURE;
    }

    cout << "The input image is: " << inputImageFilename << endl;

    //Read Image
    typedef double InputPixelType;
    static const int dimension = 2;
    typedef itk::VectorImage< InputPixelType, dimension> VectorInputImageType;

    typedef itk::Image< InputPixelType, dimension> InputImageType;

    typedef itk::ImageFileReader<VectorInputImageType> VectorImageReaderType;

    typedef double OutputPixelType;
    typedef itk::VectorImage< OutputPixelType, dimension> OutputImageType;
    typedef itk::ImageFileWriter< OutputImageType > OutputImageFileWriterType;
    
    VectorImageReaderType::Pointer reader = VectorImageReaderType::New();
    reader->SetFileName(inputImageFilename.c_str());
    reader->Update();

    typedef itk::ComposeImageFilter<InputImageType, VectorInputImageType> ComposeImageFilterType;
    ComposeImageFilterType::Pointer compose = ComposeImageFilterType::New();

    for(unsigned i = 0; i < extractComponents.size(); i++){

        typedef itk::VectorIndexSelectionCastImageFilter<VectorInputImageType, InputImageType> VectorIndexSelectionCastImageFilterType;
        VectorIndexSelectionCastImageFilterType::Pointer vector_selection = VectorIndexSelectionCastImageFilterType::New();
        vector_selection->SetIndex(extractComponents[i]);
        vector_selection->SetInput(reader->GetOutput());
        vector_selection->Update();

        typedef itk::RescaleIntensityImageFilter<InputImageType, InputImageType> RescaleIntensityImageFilterType;
        RescaleIntensityImageFilterType::Pointer rescale = RescaleIntensityImageFilterType::New();
        rescale->SetInput(vector_selection->GetOutput());
        rescale->SetOutputMinimum(outputMinimum);
        rescale->SetOutputMaximum(outputMaximum);
        rescale->Update();

        compose->PushBackInput(rescale->GetOutput());

    }

    compose->Update();
    

    OutputImageFileWriterType::Pointer writer = OutputImageFileWriterType::New();

    cout<<"Writing: "<<outputImageFilename<<endl;
    writer->UseCompressionOn();
    writer->SetFileName(outputImageFilename.c_str());
    writer->SetInput(compose->GetOutput());
    writer->Update();   


    return EXIT_SUCCESS;
}
