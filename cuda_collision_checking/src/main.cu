// system
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "string"
#include <filesystem>
#include <cassert>

//ros
#include <rclcpp/rclcpp.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include  "visualization_msgs/msg/marker.hpp"

//package
#include "cuda_collision_checking/cuda_collision_checking.hpp"


std::vector<float> load_verts(const std::string &file_name) {
    // Initialize Assimp
    Assimp::Importer importer;

    // Load the first mesh
    const aiScene *scene = importer.ReadFile(file_name, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
    if (!scene || !scene->mRootNode) {
        throw std::runtime_error("Failed to load mesh1.obj");
    }
    assert(scene->mNumMeshes == 1);
    const aiMesh *mesh = scene->mMeshes[0];

    std::vector<float> vert(mesh->mNumFaces * 3 * 3);
    int count = 0;
    for (int f = 0; f < mesh->mNumFaces; f++) {
        auto simp_ind = mesh->mFaces[f];
        for (int i = 0; i < simp_ind.mNumIndices; i++) {
            auto ind = simp_ind.mIndices[i];
            vert[count++] = mesh->mVertices[ind].x;
            vert[count++] = mesh->mVertices[ind].y;
            vert[count++] = mesh->mVertices[ind].z;
        }
    }

    return vert;
}

std::pair<std::vector<float>, std::vector<float>> load() {
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("cuda_collision_checking");
    auto file_path = pkg_dir / "data";
    std::filesystem::create_directory(file_path);
    auto mesh_file_1 = (file_path / "mesh1.obj").string();
    auto mesh_file_2 = (file_path / "mesh2.obj").string();

    return {load_verts(mesh_file_1), load_verts(mesh_file_2)};
}

//void publish_collisions(const std::vector<Triangle> &triangle, const std::string &topic, std::vector<int> collisions,
//                        std::vector<Normal> debug_normal) {
//    auto node = std::make_shared<rclcpp::Node>("markers_node");
//    auto pub_ = node->create_publisher<visualization_msgs::msg::Marker>(topic, 10);
//
//    visualization_msgs::msg::Marker marker;
//    marker.header.frame_id = "base";
//    marker.id = 0;
//    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
//    marker.action = visualization_msgs::msg::Marker::ADD;
//    marker.scale.x = 1.0;
//    marker.scale.y = 1.0;
//    marker.scale.z = 1.0;
//
//    marker.color.a = 1.0; // Don't forget to set the alpha!
//    marker.color.r = 0.0;
//    marker.color.g = 1.0;
//    marker.color.b = 0.0;
//    int ind = 0;
//    for (auto &face: triangle) {
//        geometry_msgs::msg::Point point;
//        std_msgs::msg::ColorRGBA color;
//        float scale = 1.0;
//        if (!debug_normal.empty()) {
//            float val = debug_normal[ind].dir[0] * debug_normal[ind].dir[0] +
//                        debug_normal[ind].dir[1] * debug_normal[ind].dir[1] +
//                        debug_normal[ind].dir[2] * debug_normal[ind].dir[2];
//            val = sqrt(val);
//
//            scale = abs(debug_normal[ind].dir[2]) / val;
//            scale = 1;
//            if (scale > 1.0) scale = 1.0;
//            if (scale < 0.1) scale = 0.1;
//        }
////        if (!collisions.empty() && collisions[ind]) {
////            color.a = 1.0;
////            color.r = 1.0 * scale;
////        } else {
//        color.a = 1.0;
//        color.g = 1.0 * scale;
////        }
//        ind++;
//
//        point.x = face.v[0].x;
//        point.y = face.v[0].y;
//        point.z = face.v[0].z;
//        marker.points.push_back(point);
//        marker.colors.push_back(color);
//        point.x = face.v[1].x;
//        point.y = face.v[1].y;
//        point.z = face.v[1].z;
//        marker.points.push_back(point);
//        marker.colors.push_back(color);
//        point.x = face.v[2].x;
//        point.y = face.v[2].y;
//        point.z = face.v[2].z;
//        marker.points.push_back(point);
//        marker.colors.push_back(color);
//    }
//    pub_->publish(marker);
//
//    std::chrono::milliseconds pause{100};
//    for (int i = 0; i < 5; i++) {
//        rclcpp::sleep_for(pause);
//    }
//}


int main(int argc, char *argv[]) {

    auto [vert1, vert2] = load();

    auto handle = cuda_collision_check::init();

    std::vector<int> vert_inds = {handle->load_verts(vert1), handle->load_verts(vert2)};


    auto start = std::chrono::high_resolution_clock::now();
    bool collide;
    for (int i = 0; i < 100000; i++) { //100000
        //check collision
        collide = handle->check_collision(vert_inds[0], vert_inds[1]);
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    std::cout << "did collide?  : " << collide << std::endl;

    // visualize
    rclcpp::init(argc, argv);
//    publish_collisions(triangles_1, "mesh_1", collisions, debug_normal);
//    publish_collisions(triangles_2, "mesh_2");


    return 0;

}




