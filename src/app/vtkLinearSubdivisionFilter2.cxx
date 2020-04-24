#include "vtkLinearSubdivisionFilter2.h"

#include "vtkIntArray.h"
#include "vtkCellData.h"
#include "vtkCellArray.h"
#include "vtkCellIterator.h"
#include "vtkEdgeTable.h"
#include "vtkIdList.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include <string>
#include <sstream>

#include "vtkCleanPolyData.h"

vtkStandardNewMacro(vtkLinearSubdivisionFilter2);



int vtkLinearSubdivisionFilter2::RequestData(vtkInformation * request, 
                                                vtkInformationVector ** inputVector, 
                                                vtkInformationVector * outputVector) 
{
  if (!this->vtkSubdivisionFilterRequestData(request, inputVector,
                                                outputVector))
  {
    return 0;
  }

  // get the info objects
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkPolyData *input = vtkPolyData::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPolyData *output = vtkPolyData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkIdType numCells;
  int factor = this->GetNumberOfSubdivisions()*this->GetNumberOfSubdivisions();
  vtkPoints *outputPts;
  vtkCellArray *outputPolys;
  vtkPointData *outputPD;
  vtkCellData *outputCD;
  vtkIntArray *edgeData;

  vtkSmartPointer<vtkCleanPolyData> cleaner =
      vtkSmartPointer<vtkCleanPolyData>::New();



  //
  // Initialize and check input
  //

  //copy the input in inputDS
  vtkPolyData *inputDS = vtkPolyData::New();
  inputDS->CopyStructure (input);
  inputDS->GetPointData()->PassData(input->GetPointData());
  inputDS->GetCellData()->PassData(input->GetCellData());

  
  // Generate topology  for the input dataset
  inputDS->BuildLinks();
  numCells = inputDS->GetNumberOfCells ();

  // Copy points from input. The new points will include the old points
  // and points calculated by the subdivision algorithm
  outputPts = vtkPoints::New();
  //outputPts->GetData()->DeepCopy(inputDS->GetPoints()->GetData());
  int subdivisions = this->GetNumberOfSubdivisions();
  // Copy pointdata structure from input
  int size = inputDS->GetNumberOfCells() * (subdivisions+1)*(subdivisions+2)/2;
  outputPD = vtkPointData::New();
  outputPD->CopyAllocate(inputDS->GetPointData(),size);



  // Copy celldata structure from input
  outputCD = vtkCellData::New();
  // outputCD->CopyAllocate(inputDS->GetCellData(),factor*numCells);

  // outputCD->Allocate(inputDS->GetCellData(),factor*numCells);

  // Create triangles
  outputPolys = vtkCellArray::New();
  outputPolys->Allocate(outputPolys->EstimateSize(factor*numCells,3));

  // Create an array to hold new location indices
  edgeData = vtkIntArray::New();
  edgeData->SetNumberOfComponents(3);
  //edgeData->SetNumberOfTuples(factor*numCells);

  if (this->GenerateSubdivisionPoints (inputDS, edgeData, outputPts, outputPD) == 0)
  {
    outputPts->Delete();
    outputPD->Delete();
    outputCD->Delete();
    outputPolys->Delete();
    inputDS->Delete();
    edgeData->Delete();
    vtkErrorMacro("Subdivision failed.");
    return 0;
  }

  this->GenerateSubdivisionCells (inputDS, edgeData, outputPolys, outputCD);

  // start the next iteration with the input set to the output we just created
  edgeData->Delete();
  inputDS->Delete();
  inputDS = vtkPolyData::New();
  inputDS->SetPoints(outputPts); outputPts->Delete();
  inputDS->SetPolys(outputPolys); outputPolys->Delete();
  inputDS->GetPointData()->PassData(outputPD); outputPD->Delete();
  inputDS->GetCellData()->PassData(outputCD); outputCD->Delete();
  inputDS->Squeeze();

  cleaner->SetInputData(inputDS);
  cleaner->Update();

  output->SetPoints(cleaner->GetOutput()->GetPoints());
  output->SetPolys(cleaner->GetOutput()->GetPolys());
  output->GetPointData()->PassData(cleaner->GetOutput()->GetPointData());
  output->GetCellData()->PassData(cleaner->GetOutput()->GetCellData());
  inputDS->Delete();

  return 1;

}

  
void vtkLinearSubdivisionFilter2::GenerateSubdivisionCells (vtkPolyData *inputDS,
                                                               vtkIntArray *edgeData, 
                                                               vtkCellArray *outputPolys, 
                                                               vtkCellData *outputCD)
{

  vtkIdType numCells = inputDS->GetNumberOfCells();
  vtkIdType cellId, newId;
  int id;
  vtkIdType npts;
  vtkIdType *pts;
  double edgePts[3];
  vtkIdType newCellPts[3];
  vtkCellData *inputCD = inputDS->GetCellData();


  int subdivisions = this->GetNumberOfSubdivisions();
  //number of subcell generated for each cell
  int nbr_gen_cell= subdivisions*subdivisions;

  

  // cellId = 0;
  for (int i = 0 ; i < edgeData->GetNumberOfTuples(); i++){
    double ids[3];
    edgeData->GetTuple(i,ids);

    id = 0;
    newCellPts[id++] = (int) ids[2];
    newCellPts[id++] = (int) ids[1];
    newCellPts[id] = (int) ids[0];
    newId = outputPolys->InsertNextCell (3, newCellPts);
    outputCD->CopyData (inputCD, floor(i/nbr_gen_cell), newId);

    
  }
}





int vtkLinearSubdivisionFilter2::GenerateSubdivisionPoints (vtkPolyData *inputDS, 
                                                               vtkIntArray *edgeData, 
                                                               vtkPoints *outputPts, 
                                                               vtkPointData *outputPD)
{

  int edgeId;

  vtkIdType npts, cellId, newId;

  vtkCellArray *inputPolys=inputDS->GetPolys();

  vtkSmartPointer<vtkIdList> cellIds =
    vtkSmartPointer<vtkIdList>::New();

  vtkSmartPointer<vtkIdList> pointIds =
    vtkSmartPointer<vtkIdList>::New();

  vtkSmartPointer<vtkEdgeTable> edgeTable =
    vtkSmartPointer<vtkEdgeTable>::New();

  vtkPoints *inputPts=inputDS->GetPoints();

  vtkPointData *inputPD=inputDS->GetPointData();

  int subdivisions = this->GetNumberOfSubdivisions();

  // Create an edge table to keep track of which edges we've processed
  edgeTable->InitEdgeInsertion(inputDS->GetNumberOfPoints());

  pointIds->SetNumberOfIds(3);

  double total = inputPolys->GetNumberOfCells();
  double curr = 0;

  //Generate the weights matrix that will be used to interpolate the points
  std::vector< std::vector< double> > weight_array;
  for (int i = 0 ; i <= subdivisions; i++){
    for (int j = 0 ; j <= subdivisions-i; j++){
      int k= subdivisions - i - j;
      std::vector< double> w(3);

      w[0] = ((double) i) / subdivisions;
      w[1] = ((double) j) / subdivisions;
      w[2] = ((double) k) / subdivisions;

      weight_array.push_back(w);
    }
  }

  //for each cell
  for (cellId=0, inputPolys->InitTraversal();
       inputPolys->GetNextCell(pointIds); cellId++)
  {

    //interpolate the new points
    std::vector<int> newIds;
    for (int i = 0; i < weight_array.size(); i++){
      //get the weights
      double weights[3] = {weight_array[i][0],weight_array[i][1],weight_array[i][2]};
      //generate new point and his associated point data
      newId = this->InterpolatePosition(inputPts,outputPts,pointIds,weights);
      outputPD->InterpolatePoint(inputPD,newId,pointIds,weights);
      newIds.push_back(newId);
    }

    //store the edges in edgedata
    int id1 = -1;
    int id2, id3, id4;
    for (int i = 0 ; i < subdivisions; i++){
      id1++;
      for (int j = 0 ; j < subdivisions-i; j++){

        id2 = id1 + 1;
        id3 = id1 + subdivisions + 1 - i;
        id4 = id3 + 1;
      
        double points[3];
        points[0]=newIds[id1];
        points[1]=newIds[id2];
        points[2]=newIds[id3];
        edgeData->InsertNextTuple(points);

        if (j < subdivisions - i - 1){
          double points[3];
          points[0]=newIds[id2];
          points[1]=newIds[id4];
          points[2]=newIds[id3];
          edgeData->InsertNextTuple(points);
        }
        id1++;
      }
    }
    this->UpdateProgress(curr / total);
    curr++;
  } // each cell

  return 1;
}




int vtkLinearSubdivisionFilter2::vtkSubdivisionFilterRequestData(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *vtkNotUsed(outputVector))
{
  // validate the input
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);

  // get the input
  vtkPolyData *input = vtkPolyData::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  if (!input)
  {
    return 0;
  }

  vtkIdType numCells, numPts;
  numPts=input->GetNumberOfPoints();
  numCells=input->GetNumberOfCells();

  if (numPts < 1 || numCells < 1)
  {
    vtkErrorMacro(<<"No data to subdivide");
    return 0;
  }


  std::map<int,int> badCellTypes;
  bool hasOnlyTris = true;
  vtkCellIterator *it = input->NewCellIterator();
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextCell())
  {
  if (it->GetCellType() != VTK_TRIANGLE)
    {
      hasOnlyTris = false;
      badCellTypes[it->GetCellType()] += 1;
      continue;
    }
  }
  it->Delete();

  if (!hasOnlyTris)
  {
    std::ostringstream msg;
    std::map <int, int>::iterator cit;
    for (cit = badCellTypes.begin(); cit != badCellTypes.end(); ++cit)
    {
      msg << "Cell type: " << cit->first << " Count: " << cit->second << "\n";
    }
    vtkErrorMacro(<< this->GetClassName() << " only operates on triangles, but "
                  "this data set has other cell types present.\n"
                  << msg.str());
    return 0;
  }
  
  return 1;
}