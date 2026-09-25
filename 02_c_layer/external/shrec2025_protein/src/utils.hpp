#ifndef _SHREC2025_PROTEIN_UTILS_HPP_
#define _SHREC2025_PROTEIN_UTILS_HPP_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>

#include "sobol.hpp"

double calc_area(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2, const pcl::PointXYZ& p3) {
  const double a = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
  const double b = (p3.x - p2.x) * (p3.x - p2.x) + (p3.y - p2.y) * (p3.y - p2.y) + (p3.z - p2.z) * (p3.z - p2.z);
  const double c = (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) + (p1.z - p2.z) * (p3.z - p2.z);
  const double area = 0.5 * std::sqrt(a * b - c * c);
  if (std::isnormal(area)) {
    return area;
  }
  return 0.0;
}

double calc_polygon_area(const pcl::PointCloud<pcl::PointXYZ>& vertices, const pcl::Vertices& polygon) {
  return calc_area(vertices[polygon.vertices[0]], vertices[polygon.vertices[1]], vertices[polygon.vertices[2]]);
}

void normalize(pcl::PointCloud<pcl::PointXYZ>& vertices) {
  unsigned int n_vertices = vertices.size();
  pcl::PointXYZ mean;
  mean.x = 0.0;
  mean.y = 0.0;
  mean.z = 0.0;
  for (unsigned int i = 0; i < n_vertices; i++) {
    mean.x += vertices[i].x;
    mean.y += vertices[i].y;
    mean.z += vertices[i].z;
  }
  mean.x /= n_vertices;
  mean.y /= n_vertices;
  mean.z /= n_vertices;
  for (unsigned int i = 0; i < n_vertices; i++) {
    vertices[i].x -= mean.x;
    vertices[i].y -= mean.y;
    vertices[i].z -= mean.z;
  }
  double max_distance = 0.0;
  for (unsigned int i = 0; i < n_vertices; i++) {
    const double distance = std::sqrt(
      vertices[i].x * vertices[i].x + vertices[i].y * vertices[i].y + vertices[i].z * vertices[i].z
    );
    if (distance > max_distance) {
      max_distance = distance;
    }
  }
  for (unsigned int i = 0; i < n_vertices; i++) {
    vertices[i].x /= max_distance;
    vertices[i].y /= max_distance;
    vertices[i].z /= max_distance;
  }
}

bool load_potential(const char* filename, std::vector<float>& potential) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return 1;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream iss(line);
    std::string token;
    iss >> token;
    if (token == "POINT_DATA") {
      unsigned int n_points;
      iss >> n_points;

      std::string attribute;
      file >> attribute;
      if (attribute == "SCALARS") {
        std::string dataName;
        std::string dataType;
        file >> dataName >> dataType;
        if (dataName == "Potential") {
          std::string lookup;
          file >> lookup;
          if (lookup == "LOOKUP_TABLE") {
            std::string tableName;
            file >> tableName;
            potential.clear();
            potential.reserve(n_points);
            for (int i = 0; i < n_points; ++i) {
              float value;
              file >> value;
              potential.push_back(value);
            }
            break;
          }
        }
      }
    }
  }

  file.close();

  return true;
}

void mesh_to_points(
  const pcl::PolygonMesh& mesh,
  const std::vector<float>& potential,
  const unsigned int n_points,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  std::vector<float>& points_potential
) {
  pcl::PointCloud<pcl::PointXYZ> vertices;
  pcl::fromPCLPointCloud2(mesh.cloud, vertices);
  std::vector<pcl::Vertices> polygons = mesh.polygons;

  normalize(vertices);

  double mesh_area = 0.0;
  std::vector<double> polygon_areas;
  polygon_areas.reserve(polygons.size());
  for (unsigned int i = 0; i < polygons.size(); i++) {
    const double area = calc_polygon_area(vertices, polygons[i]);
    polygon_areas.push_back(area);
    mesh_area += area;
  }

  std::vector<std::pair<unsigned int, double>> sorted_polygon_areas;
  for (unsigned int i = 0; i < polygons.size(); i++) {
    sorted_polygon_areas.push_back(std::make_pair(i, polygon_areas[i]));
  }
  std::sort(sorted_polygon_areas.begin(), sorted_polygon_areas.end());

  long long int seed = 0;
  double r[2];
  double sum_error = 0.0;
  cloud->reserve(n_points);
  while (cloud->size() < n_points) {
    for (unsigned int i = 0; i < polygons.size(); i++) {
      const unsigned int idx = sorted_polygon_areas[i].first;
      const double ratio = polygon_areas[idx] / mesh_area;
      unsigned int n_polygon_points = n_points * ratio;
      sum_error += n_points * ratio - n_polygon_points;
      if (sum_error >= 1.0) {
        n_polygon_points += 1;
        sum_error -= 1.0;
      }
      for (unsigned int j = 0; j < n_polygon_points && cloud->size() < n_points; j++) {
        i8_sobol(2, &seed, r);
        const double r1 = std::sqrt(std::fabs(r[0]));
        const double r2 = std::fabs(r[1]);
        pcl::PointXYZ point;
        point.x = (1.0 - r1) * vertices[polygons[idx].vertices[0]].x +
                    r1 * (1.0 - r2) * vertices[polygons[idx].vertices[1]].x +
                      r1 * r2 * vertices[polygons[idx].vertices[2]].x;
        point.y = (1.0 - r1) * vertices[polygons[idx].vertices[0]].y +
                    r1 * (1.0 - r2) * vertices[polygons[idx].vertices[1]].y +
                      r1 * r2 * vertices[polygons[idx].vertices[2]].y;
        point.z = (1.0 - r1) * vertices[polygons[idx].vertices[0]].z +
                    r1 * (1.0 - r2) * vertices[polygons[idx].vertices[1]].z +
                      r1 * r2 * vertices[polygons[idx].vertices[2]].z;
        cloud->push_back(point);
        points_potential.push_back(
          (1.0 - r1) * potential[polygons[idx].vertices[0]] +
          r1 * (1.0 - r2) * potential[polygons[idx].vertices[1]] +
          r1 * r2 * potential[polygons[idx].vertices[2]]
        );
      }
    }
  }
}

std::vector<float> create_histogram(const std::vector<float>& data, const int n_bins) {
  std::vector<float> histogram(n_bins, 0.0);
  if (data.empty()) {
    std::cerr << "Error: Data vector is empty." << std::endl;
    return histogram;
  }

  const float min = *std::min_element(data.begin(), data.end());
  const float max = *std::max_element(data.begin(), data.end());
  const float range = max - min;
  if (range <= 0.0) {
    histogram[0] = static_cast<float>(data.size());
    return histogram;
  }

  const float bin_width = range / static_cast<float>(n_bins);
  for (const auto& val : data) {
    const int bin_index = static_cast<int>((val - min) / bin_width);
    if (bin_index >= 0 && bin_index < n_bins) {
      histogram[bin_index] += 1.0f;
    } else if (bin_index < 0) {
      histogram[0] += 1.0f;
    } else {
      histogram[n_bins - 1] += 1.0f;
    }
  }

  const float sum = std::accumulate(histogram.begin(), histogram.end(), 0.0f);
  if (sum > 0.0) {
    for (auto& val : histogram) {
      val /= sum;
    }
  }

  return histogram;
}

#endif // _SHREC2025_PROTEIN_UTILS_HPP_
