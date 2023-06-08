#include <cstdio>
#include <iostream>

#include <fcl/broadphase/broadphase_collision_manager.h>
#include <fcl/narrowphase/collision.h>

#include <fcl/narrowphase/distance.h>
#include <fcl/fcl.h>

/** @author Ryodo Tanaka <groadpg@gmail.com> */
#include <iostream>
#include <Eigen/Core>

// Collision, Distance
#include <fcl/narrowphase/collision_object.h>
#include <fcl/narrowphase/distance.h>

// Distance Request & Result
#include <fcl/narrowphase/distance_request.h>
#include <fcl/narrowphase/distance_result.h>


#include "collision/datapoint.hpp"
#include <random>

void write_data(const DataStore &data);


void loadMeshFromObj(const std::string &filename, std::vector<Eigen::Vector3d> &vertices,
                     std::vector<Eigen::Vector3i> &triangles) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Failed to open OBJ file: " << filename << std::endl;
        return;
    }

    vertices.clear();
    triangles.clear();

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            Eigen::Vector3d vertex;
            double v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            vertex[0] = v1;
            vertex[1] = v2;
            vertex[2] = v3;
            vertices.push_back(vertex);
        } else if (type == "f") {
            int v1, v2, v3;
            char tmp;
            int tmpInt;
            iss >> v1 >> tmp >> tmpInt >> tmp >> tmpInt;
            iss >> v2 >> tmp >> tmpInt >> tmp >> tmpInt;
            iss >> v3 >> tmp >> tmpInt >> tmp >> tmpInt;
            Eigen::Vector3i tri;
            tri[0] = v1 - 1;
            tri[1] = v2 - 1;
            tri[2] = v3 - 1;
            triangles.push_back(tri);
        }
    }

    file.close();
}

bool PointInTriangle(const Eigen::Vector2d & p, const Eigen::Vector2d& a, const Eigen::Vector2d & b, const Eigen::Vector2d& c)
{
    double u;
    double v;
    double w;
//
//    auto v0 = b-a;
//    auto v1 = c-a;
//    auto v2 = p-a;
//
//    double d00 = v0.dot(v0);
//    double d01 = v0.dot(v1);
//    double d11 = v1.dot(v1);
//    double d20 = v2.dot( v0);
//    double d21 = v2.dot( v1);
//    double denom = d00 * d11 - d01 * d01;
//    v = (d11 * d20 - d01 * d21) / denom;
//    w = (d00 * d21 - d01 * d20) / denom;
//    u = 1.0f - v - w;

    auto v0 = b - a;
    auto v1 = c - a;
    auto v2 = p - a;
    float den = v0[0] * v1[1] - v1[0] * v0[1];
    v = (v2[0] * v1[1] - v1[0] * v2[1]) / den;
    w = (v0[0] * v2[1] - v2[0] * v0[1]) / den;
    u = 1.0f - v - w;

    return v>=0 && v<=1 && u>=0 && u<=1 && w>=0 && w<=1;

//    auto s = (p0[0] - p2[0]) * (p[1] - p2[1]) - (p0[1] - p2[1]) * (p[0] - p2[0]);
//    auto t = (p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0]);
//
//    if ((s < 0) != (t < 0) && s != 0 && t != 0)
//        return false;
//
//    auto d = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]);
//    return d == 0 || (d < 0) == (s + t <= 0);
}

double PointInMesh(Eigen::Vector3d p, Eigen::MatrixXd verts, Eigen::MatrixXd verts2d)
{
    Eigen::Vector2d p2 = p.block(0,0,2,1);
    int numInside = 0;
   for (int i =0 ; i < verts2d.cols(); i += 3){
       auto mid = 1.0/3.0*(verts(2,i) + verts(2,i+1) + verts(2,i+2));
       if (p[2] - mid > 0) { // less, skip because my ray is up
           continue;
       }
       Eigen::Vector2d point1 = verts2d.col(i);
       Eigen::Vector2d point2 = verts2d.col(i+1);
       Eigen::Vector2d point3 = verts2d.col(i+2);
       if (PointInTriangle(p2, point1, point2, point3)){
           numInside += 1;
       }

   }
//   std::cout << numInside << "\n";
    return 1 - 2.0*((numInside % 2) == 1); // odd means inside
}


int main(int argc, char *argv[]) {

    // Box
//    std::shared_ptr<fcl::CollisionGeometry<double>> box_geometry(new fcl::Box<double>(2, 2, 2));
//    std::shared_ptr<fcl::CollisionGeometry<double>> box_geometry(new fcl::Sphere<double>(2));

//    typedef fcl::BVHModel<fcl::OBBRSSd> Model;
    std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd> > box_geometry(new fcl::BVHModel<fcl::OBBRSSd>);


    std::vector<Eigen::Vector3d> vertices_vec;
    std::vector<Eigen::Vector3i> triangles_vec;
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision");
    auto file_path = pkg_dir / "data" / "model.obj";
    loadMeshFromObj(file_path, vertices_vec, triangles_vec);

    Eigen::MatrixXd verts = Eigen::MatrixXd(3, 3 * triangles_vec.size());
    Eigen::MatrixXd verts2d = Eigen::MatrixXd(2, 3 * triangles_vec.size());
    int ind = 0;

    for (auto tri: triangles_vec) {
        Eigen::Vector3d vert1 = vertices_vec[tri[0]];
        Eigen::Vector3d vert2 = vertices_vec[tri[1]];
        Eigen::Vector3d vert3 = vertices_vec[tri[2]];
        verts.block(0, ind++, 3, 1) = vert1;
        verts.block(0, ind++, 3, 1) = vert2;
        verts.block(0, ind++, 3, 1) = vert3;

    }
    verts2d = verts.block(0, 0, 2, verts.cols());

    Eigen::Vector3d mins = verts.rowwise().minCoeff();
    Eigen::Vector3d maxes = verts.rowwise().maxCoeff();

    std::random_device rd;
    std::default_random_engine generatorX(rd());
    std::default_random_engine generatorY(rd());
    std::default_random_engine generatorZ(rd());
    std::uniform_real_distribution<double> distributionX(mins[0]-.05, maxes[0]+.05);
    std::uniform_real_distribution<double> distributionY(mins[1]-.05, maxes[1]+.05);
    std::uniform_real_distribution<double> distributionZ(mins[2]-.05, maxes[2]+.05);

    DataStore data(128 * 64*8);
    for (int i = 0; i < 128 * 64*8; i++) {
        double x = distributionX(generatorX);
        double y = distributionY(generatorY);
        double z = distributionZ(generatorZ);

        Eigen::Vector3d point = {x, y, z};
//        trans2[0] = x;
//        trans2[1] = y;
//        trans2[2] = z;

//        sphere.setTranslation(trans2);
        // Calculate distance
//        result.clear();
        double dist = PointInMesh(point, verts, verts2d);
//        double dist = fcl::distance(&sphere, &box , request, result);

//        auto res2 = collide(&sphere,&box, request_collide, result_collide);

//        if (dist <= 0) {
//            std::cout << "hit\n";
//            dist = -1;
//        }
        data[i] = {(float)x, (float)y, (float)z, (float)dist};
//        std::cout << result.min_distance << "\n";
    }
    // Show results
//    std::cout << result.min_distance + radius << std::endl;
//    std::cout << result.nearest_points[0].transpose() << std::endl;

    write_data(data);

    return 0;
}

