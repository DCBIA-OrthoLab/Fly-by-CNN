#include "fly_by_featuresCLP.h"
#include "vtkLinearSubdivisionFilter2.h"

#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkPoints.h>
#include <vtkIdList.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkLine.h>
#include <vtkLineSource.h>
#include <vtkExtractCells.h>
#include <vtkOBBTree.h>
#include <vtkCellLocator.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkPlaneSource.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_cross.h>

#include <itkVectorImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkComposeImageFilter.h>

typedef double VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, 2> VectorImageType;  
typedef VectorImageType::Pointer VectorImagePointerType;  
typedef itk::ImageRegionIterator<VectorImageType> VectorImageIteratorType;
typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;

typedef itk::VectorImage<VectorImagePixelType, 3> VectorImageComposeType; 
typedef itk::ImageRegionIterator<VectorImageComposeType> VectorImageComposeIteratorType;
typedef itk::ImageFileWriter<VectorImageComposeType> VectorImageComposeFileWriterType; 

using namespace std;

int main(int argc, char * argv[])
{
  
  PARSE_ARGS;

  int numFeatures = 4;
  bool createRegionLabels = regionLabels.compare("") != 0;

  vtkSmartPointer<vtkPlatonicSolidSource> icosahedron_source = vtkSmartPointer<vtkPlatonicSolidSource>::New();
  icosahedron_source->SetSolidTypeToIcosahedron();
  icosahedron_source->Update();

  vtkSmartPointer<vtkLinearSubdivisionFilter2> subdivision = vtkSmartPointer<vtkLinearSubdivisionFilter2>::New();
  subdivision->SetInputData(icosahedron_source->GetOutput());
  subdivision->SetNumberOfSubdivisions(numberOfSubdivisions);
  subdivision->Update();
  vtkSmartPointer<vtkPolyData> sphere = subdivision->GetOutput();
  cout<<"Number of fly by samples: "<<sphere->GetNumberOfPoints()<<endl;;

  
  for(unsigned i = 0; i < sphere->GetNumberOfPoints(); i++){
    double point[3];
    sphere->GetPoints()->GetPoint(i, point);
    vnl_vector<double> v = vnl_vector<double>(point, 3);
    v = v.normalize()*sphereRadius;
    sphere->GetPoints()->SetPoint(i, v.data_block());
  }

  cout<<"Reading: "<<inputSurface<<endl;
  vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
  reader->SetFileName(inputSurface.c_str());
  reader->Update();
  vtkSmartPointer<vtkPolyData> input_mesh = reader->GetOutput();

  
  bool computeMax = maxCoord == -1;

  vnl_vector<double> mean_v = vnl_vector<double>(3, 0);

  for(unsigned i = 0; i < input_mesh->GetNumberOfPoints(); i++){
    double point[3];
    input_mesh->GetPoints()->GetPoint(i, point);
    vnl_vector<double> v = vnl_vector<double>(point, 3);
    mean_v += v;
    maxCoord = computeMax? max(maxCoord, v.inf_norm()) : maxCoord;
  }

  mean_v /= input_mesh->GetNumberOfPoints();

  for(unsigned i = 0; i < input_mesh->GetNumberOfPoints(); i++){
    double point[3];
    input_mesh->GetPoints()->GetPoint(i, point);
    vnl_vector<double> v = vnl_vector<double>(point, 3);
    v -= mean_v; 
    v /= maxCoord;
    input_mesh->GetPoints()->SetPoint(i, v.data_block());
  }

  vtkSmartPointer<vtkCellLocator> tree = vtkSmartPointer<vtkCellLocator>::New();
  tree->SetDataSet(input_mesh);
  tree->BuildLocator();

  vtkSmartPointer<vtkPolyDataMapper> inputMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  inputMapper->SetInputData(input_mesh);
  vtkSmartPointer<vtkActor> inputActor = vtkSmartPointer<vtkActor>::New();
  inputActor->SetMapper(inputMapper);

  vtkSmartPointer<vtkPolyDataMapper> sphereMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  sphereMapper->SetInputData(sphere);
  vtkSmartPointer<vtkActor> sphereActor = vtkSmartPointer<vtkActor>::New();
  sphereActor->SetMapper(sphereMapper);
  sphereActor->GetProperty()->SetRepresentationToWireframe();
  sphereActor->GetProperty()->SetColor(0.89,0.81,0.34);

  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
  
  renderer->AddActor(sphereActor);
  renderer->AddActor(inputActor);

  renderer->SetBackground(.4, .5, .6);
  renderWindow->Render();

  vector<VectorImageType::Pointer> compose_v;

  vtkSmartPointer<vtkPoints> spherePoints = sphere->GetPoints();
  for(unsigned i = 0; i < spherePoints->GetNumberOfPoints(); i++){

    double sphere_point[3];
    spherePoints->GetPoint(i, sphere_point);
    vnl_vector<double> sphere_point_v = vnl_vector<double>(sphere_point, 3);
    vnl_vector<double> sphere_point_delta_v = sphere_point_v - sphere_point_v*planeSpacing;
    vnl_vector<double> sphere_point_normal_v = sphere_point_v;
    sphere_point_normal_v.normalize();

    vnl_vector<double> sphere_north_v = vnl_vector<double>(3);
    sphere_north_v[0] = 0;
    sphere_north_v[1] = 0;
    sphere_north_v[2] = 1;

    vnl_vector<double> sphere_south_v = vnl_vector<double>(3);
    sphere_south_v[0] = 0;
    sphere_south_v[1] = 0;
    sphere_south_v[2] = -1;

    vnl_vector<double> plane_orient_x_v;
    vnl_vector<double> plane_orient_y_v;
    if(sphere_point_normal_v.is_equal(sphere_north_v, 1e-8) || sphere_point_normal_v.is_equal(sphere_south_v, 1e-8)){
      plane_orient_x_v = vnl_vector<double>(3, 0);
      plane_orient_x_v[0] = 1;
      plane_orient_y_v = vnl_vector<double>(3, 0);  
      plane_orient_y_v[1] = 1;
    }else{
      plane_orient_x_v = vnl_cross_3d(sphere_point_normal_v, sphere_north_v).normalize();
      plane_orient_y_v = vnl_cross_3d(sphere_point_normal_v, plane_orient_x_v).normalize();  
    }
    

    vnl_vector<double> plane_point_origin_v = sphere_point_v - plane_orient_x_v*0.5 - plane_orient_y_v*0.5;
    vnl_vector<double> plane_point_1_v = plane_point_origin_v + plane_orient_x_v;
    vnl_vector<double> plane_point_2_v = plane_point_origin_v + plane_orient_y_v;

    vtkSmartPointer<vtkPlaneSource> planeSource = vtkSmartPointer<vtkPlaneSource>::New();
    planeSource->SetOrigin(plane_point_origin_v.data_block());
    planeSource->SetPoint1(plane_point_1_v.data_block());
    planeSource->SetPoint2(plane_point_2_v.data_block());
    planeSource->SetResolution(planeResolution - 1, planeResolution - 1);
    planeSource->Update();
    vtkSmartPointer<vtkPolyData> planeMesh = planeSource->GetOutput();

    VectorImagePointerType out_image_feat = VectorImageType::New();
    VectorImageType::SizeType size;
    size[0] = planeResolution;
    size[1] = planeResolution;
    VectorImageType::RegionType region;
    region.SetSize(size);
    
    out_image_feat->SetRegions(region);
    out_image_feat->SetVectorLength(numFeatures);
    out_image_feat->Allocate();
    VectorImageType::PixelType out_pix(numFeatures);
    out_pix.Fill(-1);
    out_image_feat->FillBuffer(out_pix);

    VectorImageIteratorType out_it = VectorImageIteratorType(out_image_feat, out_image_feat->GetLargestPossibleRegion());
    out_it.GoToBegin();

    VectorImagePointerType out_image_label;
    VectorImageIteratorType out_it_label; 

    if(createRegionLabels){
      out_image_label = VectorImageType::New();

      out_image_label->SetRegions(region);
      out_image_label->SetVectorLength(1);
      out_image_label->Allocate();
      VectorImageType::PixelType out_pix_label(1);
      out_pix_label.Fill(0);
      out_image_label->FillBuffer(out_pix_label);

      out_it_label = VectorImageIteratorType(out_image_label, out_image_label->GetLargestPossibleRegion());
      out_it_label.GoToBegin();  
    }
    
    bool writeImage = false;

    for(unsigned j = 0; j < planeMesh->GetNumberOfPoints(); j++){
      double point_plane[3];
      planeMesh->GetPoints()->GetPoint(j, point_plane);
      vnl_vector<double> point_plane_v = vnl_vector<double>(point_plane, 3);
      point_plane_v = point_plane_v*planeSpacing + sphere_point_delta_v;
      planeMesh->GetPoints()->SetPoint(j, point_plane_v.data_block());

      vnl_vector<double> point_end_v = point_plane_v - sphere_point_normal_v*sphereRadius*2.0;      

      double tol = 1.e-8;
      double t;
      double x[3];
      double pcoords[3];
      int subId;
      vtkIdType cellId = -1;

      if(tree->IntersectWithLine(point_plane_v.data_block(), point_end_v.data_block(), tol, t, x, pcoords, subId, cellId)){

        VectorImageType::PixelType out_pix = out_it.Get();
        vtkSmartPointer<vtkIdList> cellPointsIds = vtkSmartPointer<vtkIdList>::New();
        
        input_mesh->GetCellPoints(cellId, cellPointsIds);

        double point_mesh[3];
        vtkIdType point0Id = cellPointsIds->GetId(0);
        input_mesh->GetPoint(point0Id, point_mesh);
        vnl_vector<double> point_mesh_v(point_mesh, 3);
        
        double* normal = input_mesh->GetPointData()->GetArray("Normals")->GetTuple(point0Id);
        out_pix[0] = normal[0];
        out_pix[1] = normal[1];
        out_pix[2] = normal[2];
        out_pix[3] = (point_plane_v - point_mesh_v).magnitude();
        out_it.Set(out_pix);

        if(createRegionLabels){
          VectorImageType::PixelType out_pix_label = out_it_label.Get();
          out_pix_label[0] = input_mesh->GetPointData()->GetArray(regionLabels.c_str())->GetTuple(point0Id)[0] + 1;
          out_it_label.Set(out_pix_label);
          writeImage = true;  
        }
        
      }

      ++out_it;
      if(createRegionLabels){
        ++out_it_label;  
      }

      vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
      lineSource->SetPoint1(point_plane_v.data_block());
      lineSource->SetPoint2(point_end_v.data_block());

      vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
      lineMapper->SetInputConnection(lineSource->GetOutputPort());
      vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
      lineActor->SetMapper(lineMapper);

      vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
      planeMapper->SetInputData(planeMesh);
      vtkSmartPointer<vtkActor> planeActor = vtkSmartPointer<vtkActor>::New();
      planeActor->SetMapper(planeMapper);
      planeActor->GetProperty()->SetRepresentationToWireframe();
      
      renderer->AddActor(lineActor);
      renderer->AddActor(planeActor);

      renderWindowInteractor->Start();
      renderer->RemoveActor(planeActor);
      renderer->RemoveActor(lineActor);
    }

    if(flyByCompose){
      compose_v.push_back(out_image_feat);
    }else{
      if(writeImage){
        char buf[50];
        sprintf(buf, "%d", i);
        cout<<"Writing: "<<i<<endl;
        VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();
        writer->SetFileName(outputName + "/" + string(buf) + ".nrrd");
        writer->SetInput(out_image_feat);
        writer->UseCompressionOn();
        writer->Update();

        if(createRegionLabels){
          writer->SetFileName(outputName + + "/" + string(buf) + "_label.nrrd");
          writer->SetInput(out_image_label);
          writer->UseCompressionOn();
          writer->Update();
        }
      }
    }
  }

  if(flyByCompose){
    
    VectorImageComposeType::Pointer out_compose = VectorImageComposeType::New();
    VectorImageComposeType::SizeType size;
    size[0] = planeResolution;
    size[1] = planeResolution;
    size[2] = compose_v.size();

    VectorImageComposeType::RegionType region;
    region.SetSize(size);
    
    out_compose->SetRegions(region);
    out_compose->SetVectorLength(numFeatures);
    out_compose->Allocate();
    VectorImageType::PixelType out_pix(numFeatures);
    out_pix.Fill(-1);
    out_compose->FillBuffer(out_pix);

    double out_spacing[3] = {planeSpacing, planeSpacing, 1};
    out_compose->SetSpacing(out_spacing);

    VectorImageComposeIteratorType out_c_it = VectorImageComposeIteratorType(out_compose, out_compose->GetLargestPossibleRegion());
    out_c_it.GoToBegin();

    for(unsigned i = 0; i < compose_v.size(); i++){
      VectorImageIteratorType out_it = VectorImageIteratorType(compose_v[i], compose_v[i]->GetLargestPossibleRegion());
      out_it.GoToBegin();

      while(!out_it.IsAtEnd()){
        out_c_it.Set(out_it.Get());
        ++out_it;
        ++out_c_it;  
      }
    }

    cout<<"Writing: "<<outputName<<endl;
    VectorImageComposeFileWriterType::Pointer writer = VectorImageComposeFileWriterType::New();
    writer->SetFileName(outputName);
    writer->SetInput(out_compose);
    writer->UseCompressionOn();
    writer->Update();
  }

  return EXIT_SUCCESS;
}