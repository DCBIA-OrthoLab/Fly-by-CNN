#ifndef SURFACEFEATURESEXTRACTOR_H
#define SURFACEFEATURESEXTRACTOR_H

#include <vtkSmartPointer.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkCurvatures.h>
#include <vtkPointLocator.h>


class SurfaceFeaturesExtractor : public vtkPolyDataAlgorithm
{
public:
    /** Conventions for a VTK Class*/
    vtkTypeMacro(SurfaceFeaturesExtractor,vtkPolyDataAlgorithm);
    static SurfaceFeaturesExtractor *New(); 

    /** Function SetInput(std::string input, std::vector<std::string> list)
    * Set the inputs data of the filter
    * @param input : input shape
    * @param list : list of group mean shapes
    */
    void SetInput(vtkSmartPointer<vtkPolyData> input, std::vector< vtkSmartPointer<vtkPolyData> > list, std::vector<std::string> landmarkFile);

    /** Function Update()
     * Update the filter and process the output
     */
    void Update();

    /**
     * Return the output of the Filter
     * @return : output of the Filter SurfaceFeaturesExtractor
     */
    vtkSmartPointer<vtkPolyData> GetOutput();

private:
    /** Variables */
    vtkSmartPointer<vtkPolyData> inputSurface;
    vtkSmartPointer<vtkPolyData> outputSurface;

    vtkSmartPointer<vtkPolyData> intermediateSurface;
    std::vector< vtkSmartPointer<vtkPolyData> > meanShapesList;
    std::vector<std::string> landmarkFile;
    /** Function init_output()
     * Initialize outputSurface
     */
    void init_output();

    /** Function compute_normals()
     * Compute normals of the input surface
     */
    void compute_normals();

    /** Function compute_positions()
     * Compute position of each point of the shape
     */
    void compute_positions();

    /** Function compute_distances()
     * Compute distance to each mean group model
     */
    void compute_distances();

    /** Function compute_curvatures()
     * Compute surface curvature at each point
     */
    void compute_maxcurvatures();
    void compute_mincurvatures();
    void compute_gaussiancurvatures();
    void compute_meancurvatures();

    /* Compute the shape index at each point */ 
    void compute_shapeindex();
    /* Compute the curvedness at each point */ 
    void compute_curvedness();

    void scalar_indexPoint();

    void store_landmarks_vtk();


protected:
    /** Constructor & Destructor */
    SurfaceFeaturesExtractor();
    ~SurfaceFeaturesExtractor();

};

#endif // SurfaceFeaturesExtractor_H