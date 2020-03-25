
#ifndef vtkLinearSubdivisionFilter2_h
#define vtkLinearSubdivisionFilter2_h

#include "vtkFiltersModelingModule.h" // For export macro
#include "vtkInterpolatingSubdivisionFilter.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include <map>

class vtkIntArray;
class vtkPointData;
class vtkPoints;
class vtkPolyData;

class vtkCellArray;
class vtkCellData;
class vtkIdList;



class VTKFILTERSMODELING_EXPORT vtkLinearSubdivisionFilter2 : public vtkInterpolatingSubdivisionFilter
{
public:
  //@{
  /**
   * Construct object with NumberOfSubdivisions set to 1.
   */
  static vtkLinearSubdivisionFilter2 *New();
  vtkTypeMacro(vtkLinearSubdivisionFilter2,vtkInterpolatingSubdivisionFilter);
  //@}

protected:
  vtkLinearSubdivisionFilter2 () {}
  ~vtkLinearSubdivisionFilter2 () override {}

  int RequestData(vtkInformation *, 
                  vtkInformationVector **, 
                  vtkInformationVector *) override;

  
  void GenerateSubdivisionCells (vtkPolyData *inputDS, 
                                 vtkIntArray *edgeData, 
                                 vtkCellArray *outputPolys, 
                                 vtkCellData *outputCD) ;


  int GenerateSubdivisionPoints (vtkPolyData *inputDS,
                                 vtkIntArray *edgeData,
                                 vtkPoints *outputPts,
                                 vtkPointData *outputPD) override;

  int vtkSubdivisionFilterRequestData(vtkInformation *,
                                      vtkInformationVector **, 
                                      vtkInformationVector *);

private:
  vtkLinearSubdivisionFilter2(const vtkLinearSubdivisionFilter2&) = delete;
  void operator=(const vtkLinearSubdivisionFilter2&) = delete;
};

#endif


// VTK-HeaderTest-Exclude: vtkLinearSubdivisionFilter2.h