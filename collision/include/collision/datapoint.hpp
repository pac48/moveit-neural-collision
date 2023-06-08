#include "ament_index_cpp/get_package_share_directory.hpp"
#include "string"
#include <filesystem>

#pragma once

struct DataPoint {
    float x = 0;
    float y = 0;
    float z = 0;
    float dist = 0;

    DataPoint() {}

    DataPoint(float xIn, float yIn, float zIn, float distIn) {
        x = xIn;
        y = yIn;
        z = zIn;
        dist = distIn;
    }
};

typedef std::vector<DataPoint> DataStore;

void write_data(const std::vector<DataPoint> &data) {
    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision");
    auto file_path = pkg_dir / "data";
    std::filesystem::create_directory(file_path);

    std::ofstream file((file_path / "data.bin").string(), std::ios::binary);  // Open the file in binary mode

    int size = data.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(int));
    for (const auto &item: data) {
        file.write(reinterpret_cast<const char *>(&item), sizeof(item));
    }
}

DataStore read_data() {


    std::filesystem::path pkg_dir = ament_index_cpp::get_package_share_directory("collision");
    auto file_path = pkg_dir / "data";
    std::filesystem::create_directory(file_path);


    std::ifstream file((file_path / "data.bin").string(), std::ios::binary);  // Open the file in binary mode


    // Read each struct from the file
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    DataStore data;
    data.assign(size, {});
    for (int i = 0; i < size; ++i) {
        DataPoint item;
        file.read(reinterpret_cast<char*>(&item), sizeof(item));
        data[i] = item;
    }

    file.close();
    return data;

}
