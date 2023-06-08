#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>


namespace cuda_collision_check {
    struct Vert {
        float x;
        float y;
        float z;
    };
    struct Triangle {
        Vert v[3];
        Vert center;
        float radius;
    };

    template<typename T>
    struct GPUData {
        GPUData(std::vector<T> data);

        GPUData();

        ~GPUData();

        std::vector<T> to_cpu();

        int size_;
        T *data_gpu_;

    };

    class CollisionChecker {
    public:

        CollisionChecker();

        int load_verts(const std::vector<float> &verts);

        bool check_collision(int vert_handle_1, int vert_handle_2);

    private:
        std::unordered_map<int, std::shared_ptr<GPUData<Triangle>>> verts_map_;
        std::unique_ptr<GPUData<float>> done_gpu_;
    };

    std::shared_ptr<CollisionChecker> init();

}
