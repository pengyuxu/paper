#include <pcl/io/vtk_lib_io.h>
#ifdef _OPENMP
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#else
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#endif

#include <Eigen/Core>

#include "utils.hpp"

#define N_POINTS 30000
#define FPFH_SIZE 33
#define N_BINS 16

int main(int argc, char* argv[])
{
  // Load vtk file
  pcl::PolygonMesh mesh;
  pcl::io::loadPolygonFileVTK(argv[1], mesh);
  std::vector<float> potential;
  load_potential(argv[1], potential);

  // Convert mesh to point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<float> points_potential;
  mesh_to_points(mesh, potential, N_POINTS, points, points_potential);

  // Estimate normal vectors.
#ifdef _OPENMP
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
#else
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
#endif
  ne.setInputCloud(points);
  ne.setRadiusSearch(0.1);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*normals);

  // Extract FPFH features.
#ifdef _OPENMP
  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
#else
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
#endif
  fpfh.setInputCloud(points);
  fpfh.setInputNormals(normals);
  fpfh.setRadiusSearch(0.2);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
  fpfh.compute(*descriptors);

  // Calculate statistics.
  const int n_descriptors = descriptors->size();
  Eigen::MatrixXf x(n_descriptors, FPFH_SIZE);
  for (int i = 0; i < n_descriptors; i++) {
    for (int j = 0; j < FPFH_SIZE; j++) {
      x(i, j) = descriptors->points[i].histogram[j];
    }
  }
  Eigen::VectorXf mean = x.colwise().mean();
  Eigen::MatrixXf centered = x.rowwise() - mean.transpose();
  Eigen::MatrixXf covariance = (centered.transpose() * centered) / (n_descriptors - 1);

  std::vector<float> histogram = create_histogram(points_potential, N_BINS);
  const float potential_mean = std::accumulate(points_potential.begin(), points_potential.end(), 0.0f) / points_potential.size();
  float potential_var = 0.0f;
  for (const auto& p : points_potential) {
    potential_var += (p - potential_mean) * (p - potential_mean);
  }
  potential_var /= points_potential.size();

  // Save statistics.
  std::ofstream ofs(argv[2], std::ios_base::binary | std::ios_base::out);
  for (int i = 0; i < FPFH_SIZE; i++) {
    const float value = mean(i);
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(float));
  }
  for (int i = 0; i < FPFH_SIZE; i++) {
    for (int j = i; j < FPFH_SIZE; j++) {
      const float value = covariance(i, j);
      ofs.write(reinterpret_cast<const char*>(&value), sizeof(float));
    }
  }

  for (int i = 0; i < N_BINS; i++) {
    const float value = histogram[i];
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(float));
  }
  ofs.write(reinterpret_cast<const char*>(&potential_mean), sizeof(float));
  ofs.write(reinterpret_cast<const char*>(&potential_var), sizeof(float));
  ofs.close();

  return 0;
}
