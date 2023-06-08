#include <cstdio>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
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
#include <cuda_fp16.h>

#include "tiny-cuda-nn/cpp_api.h"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include <filesystem>
#include <collision/datapoint.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>


struct GPUData {
    GPUData(std::vector<float> data, int dim) {
        size_ = data.size();
        stride_ = 1;
        dim_ = dim;
        cudaMalloc(&data_gpu_, size_ * sizeof(float));
        cudaMemcpy(data_gpu_, data.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);

        std::srand(static_cast<unsigned>(std::time(nullptr)));
    }

    GPUData(int size, int dim) {
        size_ = size;
        stride_ = 1;
        dim_ = dim;
        cudaMalloc(&data_gpu_, size_ * sizeof(float));
    }

    ~GPUData() {
        cudaFree(data_gpu_);
    }

    int sampleInd(int num_elements) {
        assert(size_ / dim_ - num_elements >= 0);
        int offset = (std::rand() % (1 + size_/ dim_ - num_elements));
        return offset;
    }

    float *sample(int offset) {
        return data_gpu_ + offset * dim_;
    }

    std::vector<float> toCPU() {
        std::vector<float> out(size_ / stride_);
        if (stride_ == 1) {
            cudaMemcpy(out.data(), data_gpu_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::vector<float> buf(size_);
            cudaMemcpy(buf.data(), data_gpu_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < size_ / stride_; i++) {
                out[i] = buf[stride_ * i];
            }
        }

        return out;
    }

    float *data_gpu_;
    int size_;
    int dim_;
    int stride_;
};


void predict(cudaStream_t const *stream_ptr, tcnn::cpp::Module *network,
             float *params, const GPUData &inputs, GPUData &output);

void publish_pointcloud(const std::vector<float> &features,
                        const std::vector<float> &pred_targets, std::shared_ptr<rclcpp::Node> &node,
                        std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_);

void publish_pointcloud(const std::vector<float> &features,
                        const std::vector<float> &pred_targets, std::shared_ptr<rclcpp::Node> &node,
                        std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> &pub_) {
    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    auto x_field = sensor_msgs::msg::PointField();
    auto y_field = sensor_msgs::msg::PointField();
    auto z_field = sensor_msgs::msg::PointField();

    x_field.name = "x";
    x_field.count = 1;
    x_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    x_field.offset = 0;

    y_field.name = "y";
    y_field.count = 1;
    y_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    y_field.offset = 4;

    z_field.name = "z";
    z_field.count = 1;
    z_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    z_field.offset = 8;

    auto color_field = sensor_msgs::msg::PointField();
    color_field.name = "rgba";
    color_field.count = 1;
    color_field.datatype = sensor_msgs::msg::PointField::UINT32;
    color_field.offset = 12;

    msg->width = pred_targets.size();
    msg->height = 1;
    msg->header.stamp = node->get_clock()->now();
    msg->header.frame_id = "base";

    msg->fields = {x_field, y_field, z_field, color_field};
//    msg->is_dense = true;
    msg->point_step = 16;
    msg->row_step = msg->width * msg->point_step;
//    msg->is_bigendian = true;

    msg->data.assign(16 * pred_targets.size(), 0);

    for (int i = 0; i < pred_targets.size(); i++) {
        float point[3] = {features[3 * i + 0], features[3 * i + 1], features[3 * i + 2]};


        memcpy(msg->data.data() + 16 * i, &point[0], 3 * sizeof(float));

        uint8_t color[4] = {0, 0, 0, 255};
        if (pred_targets[i] < 0) {
            color[1] = 255; // green
        } else {
            color[2] = 255;
            color[3] = 0;
        }
//        color[1] = 128;
        memcpy(msg->data.data() + 12 + 16 * i, &color[0], 4 * sizeof(uint8_t));

    }

    pub_->publish(*msg);
}

void predict(cudaStream_t const *stream_ptr, tcnn::cpp::Module *network,
             float *params, const GPUData &inputs, GPUData &output) {

    auto batch_size = output.size_ / network->n_output_dims();
    assert(output.size_ == network->n_output_dims() * batch_size);

    output.stride_ = 16;
    tcnn::cpp::Context ctx = network->forward(*stream_ptr, batch_size, inputs.data_gpu_, output.data_gpu_, params,
                                              false);

}

int main(int argc, char *argv[]) {
    // ros2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("point_cloud_vis");
    auto pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", 10);

    // load data
    auto data = read_data();
    std::vector<float> targets(data.size());
    std::vector<float> features(3 * data.size());
    for (int i = 0; i < data.size(); i++) {
        auto &datpoint = data[i];
        targets[i] = datpoint.dist;
        features[3 * i + 0] = datpoint.x;
        features[3 * i + 1] = datpoint.y;
        features[3 * i + 2] = datpoint.z;
//        features[i + 0*targets.size()] = datpoint.x;
//        features[i + 1*targets.size()] = datpoint.y;
//        features[i + 2*targets.size()] = datpoint.z;
    }

    GPUData features_gpu{features, 3};
    GPUData targets_gpu{targets, 1};
    GPUData pred_targets_gpu((int) 16 * targets.size(), 1);

    // load config
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision");
    auto json_file_path = pkg_dir / "config" / "config.json";
    std::fstream file(json_file_path.string());
    std::stringstream buffer;  // Create a stringstream to store the file contents
    buffer << file.rdbuf();  // Read the file into the stringstream
    std::string config_string = buffer.str(); // "{\"encoding\":{\"base_resolution\":16,\"log2_hashmap_size\":19,\"n_features_per_level\":2,\"n_levels\":16,\"otype\":\"HashGrid\",\"per_level_scale\":2.0},\"loss\":{\"otype\":\"L2\"},\"network\":{\"activation\":\"ReLU\",\"n_hidden_layers\":2,\"n_neurons\":64,\"otype\":\"FullyFusedMLP\",\"output_activation\":\"None\"},\"optimizer\":{\"learning_rate\":0.001,\"otype\":\"Adam\"}}";
    nlohmann::json config = nlohmann::json::parse(config_string);

    // load network and cuda
    constexpr uint32_t n_input_dims = 3;
    constexpr uint32_t n_output_dims = 1;
    uint32_t batch_size = targets.size();

    auto stream_ptr = new cudaStream_t();
    cudaStreamCreate(stream_ptr);
    auto trainable_model = tcnn::cpp::create_trainable_model(n_input_dims, n_output_dims, config);


    for (int i = 0; i < 1000; ++i) {
        int ind = features_gpu.sampleInd(batch_size);
        float *training_batch_inputs = features_gpu.sample(ind);
        float *training_batch_targets = targets_gpu.sample(ind);

        auto ctx = trainable_model->training_step(*stream_ptr, batch_size, training_batch_inputs,
                                                  training_batch_targets);
        if (0 == i % 100) {
            float loss = trainable_model->loss(*stream_ptr, ctx);
            std::cout << "iteration=" << i << " loss=" << loss << std::endl;
            auto network_config = config.value("network", nlohmann::json::object());
            auto encoding_config = config.value("encoding", nlohmann::json::object());
            auto network = tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding_config,
                                                                         network_config);
            float *params = trainable_model->params();
            predict(stream_ptr, network, params, features_gpu, pred_targets_gpu);
            auto pred_targets = pred_targets_gpu.toCPU();
            publish_pointcloud(features, pred_targets, node, pub_);

        }
    }

    auto network = trainable_model->get_network();
    float *params = trainable_model->params();
    for (int i = 0; i < 20; i++) {
        predict(stream_ptr, network, params, features_gpu, pred_targets_gpu);
        auto pred_targets = pred_targets_gpu.toCPU();
        publish_pointcloud(features, pred_targets, node, pub_);

    }

    auto planning_scene_diff_publisher = node->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 10);
    moveit_msgs::msg::PlanningScene msg_scene;
    msg_scene.is_diff = true;
    msg_scene.world.collision_objects.resize(1);
    msg_scene.world.collision_objects[0].header.frame_id = "base";
    msg_scene.world.collision_objects[0].id = "complex_shape";
    msg_scene.world.collision_objects[0].operation = moveit_msgs::msg::CollisionObject::ADD;
    msg_scene.world.collision_objects[0].network_poses.resize(1);
    msg_scene.world.collision_objects[0].network_poses[0].position.x = 0;
    msg_scene.world.collision_objects[0].network_poses[0].position.y = 0;
    msg_scene.world.collision_objects[0].network_poses[0].position.z = 0;
    msg_scene.world.collision_objects[0].network_poses[0].orientation.w = 1;
    msg_scene.world.collision_objects[0].network_poses[0].orientation.x = 0;
    msg_scene.world.collision_objects[0].network_poses[0].orientation.y = 0;
    msg_scene.world.collision_objects[0].network_poses[0].orientation.z = 0;
    msg_scene.world.collision_objects[0].networks.resize(1);
    msg_scene.world.collision_objects[0].networks[0].config.data = to_string(config);
    std::vector<float> params_cpu(network->n_params());
    cudaMemcpy(params_cpu.data(), params, network->n_params() * sizeof(float), cudaMemcpyDeviceToHost);
    msg_scene.world.collision_objects[0].networks[0].weights.data = params_cpu;
    msg_scene.world.collision_objects[0].networks[0].aabb.resize(3);
    msg_scene.world.collision_objects[0].networks[0].aabb[0] = 1;
    msg_scene.world.collision_objects[0].networks[0].aabb[1] = 1;
    msg_scene.world.collision_objects[0].networks[0].aabb[2] = 1;

    for (int i = 0; i < 20; i++) {
        planning_scene_diff_publisher->publish(msg_scene);
    }

    cudaStreamDestroy(*stream_ptr);


    return 0;
}


