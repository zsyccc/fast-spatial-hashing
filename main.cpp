#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
#include <vtkVersion.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include "voxelize.hpp"

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

const int default_number_samples = 500000;
// const int default_number_samples = 5000000;

const float default_leaf_size = 0.01f;
const float INF = 1000.0f;

int main(int argc, char** argv) {
    // Parse command line arguments
    int SAMPLE_POINTS_ = default_number_samples;
    float leaf_size = default_leaf_size;
    bool vis_result = true;
    const bool write_normals = true;

    const char* filename = "models/fish_512.obj";

    vtkSmartPointer<vtkPolyData> polydata1 =
        vtkSmartPointer<vtkPolyData>::New();

    vtkSmartPointer<vtkOBJReader> readerQuery =
        vtkSmartPointer<vtkOBJReader>::New();
    readerQuery->SetFileName(filename);
    readerQuery->Update();
    polydata1 = readerQuery->GetOutput();

    // make sure that the polygons are triangles!
    vtkSmartPointer<vtkTriangleFilter> triangleFilter =
        vtkSmartPointer<vtkTriangleFilter>::New();
#if VTK_MAJOR_VERSION < 6
    triangleFilter->SetInput(polydata1);
#else
    triangleFilter->SetInputData(polydata1);
#endif
    triangleFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> triangleMapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
    triangleMapper->Update();
    polydata1 = triangleMapper->GetInput();

    bool INTER_VIS = false;

    if (INTER_VIS) {
        visualization::PCLVisualizer vis;
        vis.addModelFromPolyData(polydata1, "mesh1", 0);
        vis.setRepresentationToSurfaceForAllActors();
        vis.spin();
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_1(
        new pcl::PointCloud<pcl::PointNormal>);
    uniform_sampling(polydata1, SAMPLE_POINTS_, write_normals, *cloud_1);

    if (INTER_VIS) {
        visualization::PCLVisualizer vis_sampled;
        vis_sampled.addPointCloud<pcl::PointNormal>(cloud_1);
        if (write_normals)
            vis_sampled.addPointCloudNormals<pcl::PointNormal>(
                cloud_1, 1, 0.02f, "cloud_normals");
        vis_sampled.spin();
    }

    // Voxelgrid
    VoxelGrid<PointNormal> grid_;
    grid_.setInputCloud(cloud_1);
    grid_.setLeafSize(leaf_size, leaf_size, leaf_size);

    pcl::PointCloud<pcl::PointNormal>::Ptr voxel_cloud(
        new pcl::PointCloud<pcl::PointNormal>);
    grid_.filter(*voxel_cloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr cube_cloud(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*voxel_cloud, *cube_cloud);

    float bounding_min[3] = {INF, INF, INF};
    float bounding_max[3] = {-INF, -INF, -INF};

    for (auto&& it : *cube_cloud) {
        for (int i = 0; i < 3; i++) {
            bounding_min[i] = std::min(bounding_min[i], it.data[i]);
            bounding_max[i] = std::max(bounding_max[i], it.data[i]);
        }
    }

    // for (int i = 0; i < cube_cloud->size(); i++) {
    //     auto& it = (*cube_cloud)[i];
    //     cout << it << endl;
    // }

    std::vector<float> offset(cube_cloud->size());
    for (int i = 0; i < cube_cloud->size(); i++) {
        auto& it = (*cube_cloud)[i];
        float len = INF;
        for (int j = 0; j < 3; j++) {
            float move = len;
            if (it.normal[j] > 0) {
                move = (bounding_max[j] - it.data[j]) / it.normal[j];
            } else if (it.normal[j] < 0) {
                move = (bounding_min[j] - it.data[j]) / it.normal[j];
            }
            len = std::min(len, move);
        }
        offset[i] = len;
        for (int j = 0; j < 3; j++) {
            it.data[j] += offset[i] * it.normal[j];
        }
    }

    if (vis_result) {
        visualization::PCLVisualizer vis3("VOXELIZED SAMPLES CLOUD");
        vis3.addPointCloud<pcl::PointNormal>(voxel_cloud);
        vis3.spin();
    }

    if (vis_result) {
        visualization::PCLVisualizer vis3("CUBIC SAMPLES CLOUD");
        vis3.addPointCloud<pcl::PointNormal>(cube_cloud);
        vis3.spin();
    }

    return 0;
}