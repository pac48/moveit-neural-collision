// cuda
#include <cuda_runtime.h>
#include "cuda_collision_checking/cuda_collision_checking.hpp"


namespace cuda_collision_check {

    struct Normal {
        float dir[3];
    };

    void fillTriangles(const std::vector<float> &vert1, std::vector<Triangle> &triangles_1) {
        for (int i = 0; i < triangles_1.size(); i++) {
            for (int j = 0; j < 3; j++) {
                triangles_1[i].v[j].x = vert1[9 * i + 3 * j + 0];
                triangles_1[i].v[j].y = vert1[9 * i + 3 * j + 1];
                triangles_1[i].v[j].z = vert1[9 * i + 3 * j + 2];
                triangles_1[i].center.x += triangles_1[i].v[j].x;
                triangles_1[i].center.y += triangles_1[i].v[j].y;
                triangles_1[i].center.z += triangles_1[i].v[j].z;
            }
            triangles_1[i].center.x /= 3.0;
            triangles_1[i].center.y /= 3.0;
            triangles_1[i].center.z /= 3.0;

            for (int j = 0; j < 3; j++) {
                float tmpx = (triangles_1[i].v[j].x - triangles_1[i].center.x);
                float tmpy = (triangles_1[i].v[j].y - triangles_1[i].center.y);
                float tmpz = (triangles_1[i].v[j].z - triangles_1[i].center.z);
                float dist = tmpx * tmpx + tmpy * tmpy + tmpz * tmpz;
                triangles_1[i].radius = std::max(triangles_1[i].radius, dist);
            }
            triangles_1[i].radius = std::sqrt(triangles_1[i].radius);
        }
    }


#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x)                                                                                               \
    do {                                                                                                                  \
        cudaError_t result = x;                                                                                           \
        if (result != cudaSuccess)                                                                                        \
            throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result));  \
    } while(0)


    template<typename T>
    GPUData<T>::GPUData(std::vector<T> data) {
        size_ = data.size();
        CUDA_CHECK_THROW(cudaMalloc(&data_gpu_, size_ * sizeof(T)));
        CUDA_CHECK_THROW(cudaMemcpy(data_gpu_, data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    GPUData<T>::GPUData() {
        size_ = 0;
    }

    template<typename T>
    GPUData<T>::~GPUData() {
        if (size_ > 0) {
            cudaFree(data_gpu_);
        }
    }

    template<typename T>
    std::vector<T> GPUData<T>::to_cpu() {
        std::vector<T> out(size_);
        CUDA_CHECK_THROW(cudaMemcpy(out.data(), data_gpu_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }



// cuda hints
// set CUDA_LAUNCH_BLOCKING env var to 1 to ensure kernel finishes synchronously
// up to 48k bytes per block of shared memory. Need  to specify in the third entry of <<<>>>

    constexpr int batch = 1;

    __global__ void
    checkCollision(const Triangle *g_triangle_1, const Triangle *g_triangle_2, int size_1, int size_2, float *done) {
        int ind_b = batch * (blockIdx.x * blockDim.x + threadIdx.x);
#pragma unroll
        for (int ind = ind_b; ind < ind_b + batch; ind++) {

            int ind_2 = ind % size_2;
            int ind_1 = ind / size_2;

            if (ind_1 < size_1 && done[0] == 0) { //
                float tmpx = (g_triangle_1[ind_1].center.x - g_triangle_2[ind_2].center.x);
                float tmpy = (g_triangle_1[ind_1].center.y - g_triangle_2[ind_2].center.y);
                float tmpz = (g_triangle_1[ind_1].center.z - g_triangle_2[ind_2].center.z);
                float dist = g_triangle_1[ind_1].radius + g_triangle_2[ind_2].radius;

                float near =
                        tmpx * tmpx + tmpy * tmpy + tmpz * tmpz - dist * dist;
                if (near < 0) {

                    auto &v1 = g_triangle_1[ind_1].v[0];
                    auto &v2 = g_triangle_1[ind_1].v[1];
                    auto &v3 = g_triangle_1[ind_1].v[2];

#pragma unroll
                    for (short index = 0; index < 3; index++) {
                        auto &p1 = g_triangle_2[ind_2].v[index];
                        auto &p2 = g_triangle_2[ind_2].v[(index + 1) % 3];

                        float den =
                                p1.x * v1.y * v2.z - p1.x * v1.y * v3.z - p1.x * v1.z * v2.y + p1.x * v1.z * v3.y +
                                p1.x * v2.y * v3.z -
                                p1.x * v2.z * v3.y - p1.y * v1.x * v2.z + p1.y * v1.x * v3.z + p1.y * v1.z * v2.x -
                                p1.y * v1.z * v3.x -
                                p1.y * v2.x * v3.z + p1.y * v2.z * v3.x + p1.z * v1.x * v2.y - p1.z * v1.x * v3.y -
                                p1.z * v1.y * v2.x +
                                p1.z * v1.y * v3.x + p1.z * v2.x * v3.y - p1.z * v2.y * v3.x - p2.x * v1.y * v2.z +
                                p2.x * v1.y * v3.z +
                                p2.x * v1.z * v2.y - p2.x * v1.z * v3.y - p2.x * v2.y * v3.z + p2.x * v2.z * v3.y +
                                p2.y * v1.x * v2.z -
                                p2.y * v1.x * v3.z - p2.y * v1.z * v2.x + p2.y * v1.z * v3.x + p2.y * v2.x * v3.z -
                                p2.y * v2.z * v3.x -
                                p2.z * v1.x * v2.y + p2.z * v1.x * v3.y + p2.z * v1.y * v2.x - p2.z * v1.y * v3.x -
                                p2.z * v2.x * v3.y +
                                p2.z * v2.y * v3.x;
                        float den_inv = 1.0 / den;

                        float u = (-p1.x * p2.y * v1.z + p1.x * p2.y * v3.z + p1.x * p2.z * v1.y - p1.x * p2.z * v3.y -
                                   p1.x * v1.y * v3.z + p1.x * v1.z * v3.y + p1.y * p2.x * v1.z - p1.y * p2.x * v3.z -
                                   p1.y * p2.z * v1.x + p1.y * p2.z * v3.x + p1.y * v1.x * v3.z - p1.y * v1.z * v3.x -
                                   p1.z * p2.x * v1.y + p1.z * p2.x * v3.y + p1.z * p2.y * v1.x - p1.z * p2.y * v3.x -
                                   p1.z * v1.x * v3.y + p1.z * v1.y * v3.x + p2.x * v1.y * v3.z - p2.x * v1.z * v3.y -
                                   p2.y * v1.x * v3.z + p2.y * v1.z * v3.x + p2.z * v1.x * v3.y - p2.z * v1.y * v3.x) *
                                  den_inv;

                        float v =
                                (p1.x * p2.y * v1.z - p1.x * p2.y * v2.z - p1.x * p2.z * v1.y + p1.x * p2.z * v2.y +
                                 p1.x * v1.y * v2.z -
                                 p1.x * v1.z * v2.y - p1.y * p2.x * v1.z + p1.y * p2.x * v2.z + p1.y * p2.z * v1.x -
                                 p1.y * p2.z * v2.x -
                                 p1.y * v1.x * v2.z + p1.y * v1.z * v2.x + p1.z * p2.x * v1.y - p1.z * p2.x * v2.y -
                                 p1.z * p2.y * v1.x +
                                 p1.z * p2.y * v2.x + p1.z * v1.x * v2.y - p1.z * v1.y * v2.x - p2.x * v1.y * v2.z +
                                 p2.x * v1.z * v2.y +
                                 p2.y * v1.x * v2.z - p2.y * v1.z * v2.x - p2.z * v1.x * v2.y + p2.z * v1.y * v2.x) *
                                den_inv;

                        float t =
                                (p1.x * v1.y * v2.z - p1.x * v1.y * v3.z - p1.x * v1.z * v2.y + p1.x * v1.z * v3.y +
                                 p1.x * v2.y * v3.z -
                                 p1.x * v2.z * v3.y - p1.y * v1.x * v2.z + p1.y * v1.x * v3.z + p1.y * v1.z * v2.x -
                                 p1.y * v1.z * v3.x -
                                 p1.y * v2.x * v3.z + p1.y * v2.z * v3.x + p1.z * v1.x * v2.y - p1.z * v1.x * v3.y -
                                 p1.z * v1.y * v2.x +
                                 p1.z * v1.y * v3.x + p1.z * v2.x * v3.y - p1.z * v2.y * v3.x - v1.x * v2.y * v3.z +
                                 v1.x * v2.z * v3.y +
                                 v1.y * v2.x * v3.z - v1.y * v2.z * v3.x - v1.z * v2.x * v3.y + v1.z * v2.y * v3.x) *
                                den_inv;

                        int collide = (u > 0.0) * (u < 1.0) * (v > 0.0) * (v < 1.0) *
                                      (t > 0.0) * (t < 1.0) * ((v + u) < 1.0);

                        if (collide > 0) {
                            atomicAdd(&done[0], collide);
                        }

                    }


                    if (done[0] == 0) { //
                        auto &v1 = g_triangle_2[ind_2].v[0];
                        auto &v2 = g_triangle_2[ind_2].v[1];
                        auto &v3 = g_triangle_2[ind_2].v[2];

#pragma unroll
                        for (short index = 0; index < 3; index++) {
                            auto &p1 = g_triangle_1[ind_1].v[index];
                            auto &p2 = g_triangle_1[ind_1].v[(index + 1) % 3];

                            float den =
                                    p1.x * v1.y * v2.z - p1.x * v1.y * v3.z - p1.x * v1.z * v2.y + p1.x * v1.z * v3.y +
                                    p1.x * v2.y * v3.z -
                                    p1.x * v2.z * v3.y - p1.y * v1.x * v2.z + p1.y * v1.x * v3.z + p1.y * v1.z * v2.x -
                                    p1.y * v1.z * v3.x -
                                    p1.y * v2.x * v3.z + p1.y * v2.z * v3.x + p1.z * v1.x * v2.y - p1.z * v1.x * v3.y -
                                    p1.z * v1.y * v2.x +
                                    p1.z * v1.y * v3.x + p1.z * v2.x * v3.y - p1.z * v2.y * v3.x - p2.x * v1.y * v2.z +
                                    p2.x * v1.y * v3.z +
                                    p2.x * v1.z * v2.y - p2.x * v1.z * v3.y - p2.x * v2.y * v3.z + p2.x * v2.z * v3.y +
                                    p2.y * v1.x * v2.z -
                                    p2.y * v1.x * v3.z - p2.y * v1.z * v2.x + p2.y * v1.z * v3.x + p2.y * v2.x * v3.z -
                                    p2.y * v2.z * v3.x -
                                    p2.z * v1.x * v2.y + p2.z * v1.x * v3.y + p2.z * v1.y * v2.x - p2.z * v1.y * v3.x -
                                    p2.z * v2.x * v3.y +
                                    p2.z * v2.y * v3.x;
                            float den_inv = 1.0 / den;

                            float u = (-p1.x * p2.y * v1.z + p1.x * p2.y * v3.z + p1.x * p2.z * v1.y -
                                       p1.x * p2.z * v3.y -
                                       p1.x * v1.y * v3.z + p1.x * v1.z * v3.y + p1.y * p2.x * v1.z -
                                       p1.y * p2.x * v3.z -
                                       p1.y * p2.z * v1.x + p1.y * p2.z * v3.x + p1.y * v1.x * v3.z -
                                       p1.y * v1.z * v3.x -
                                       p1.z * p2.x * v1.y + p1.z * p2.x * v3.y + p1.z * p2.y * v1.x -
                                       p1.z * p2.y * v3.x -
                                       p1.z * v1.x * v3.y + p1.z * v1.y * v3.x + p2.x * v1.y * v3.z -
                                       p2.x * v1.z * v3.y -
                                       p2.y * v1.x * v3.z + p2.y * v1.z * v3.x + p2.z * v1.x * v3.y -
                                       p2.z * v1.y * v3.x) *
                                      den_inv;

                            float v =
                                    (p1.x * p2.y * v1.z - p1.x * p2.y * v2.z - p1.x * p2.z * v1.y + p1.x * p2.z * v2.y +
                                     p1.x * v1.y * v2.z -
                                     p1.x * v1.z * v2.y - p1.y * p2.x * v1.z + p1.y * p2.x * v2.z + p1.y * p2.z * v1.x -
                                     p1.y * p2.z * v2.x -
                                     p1.y * v1.x * v2.z + p1.y * v1.z * v2.x + p1.z * p2.x * v1.y - p1.z * p2.x * v2.y -
                                     p1.z * p2.y * v1.x +
                                     p1.z * p2.y * v2.x + p1.z * v1.x * v2.y - p1.z * v1.y * v2.x - p2.x * v1.y * v2.z +
                                     p2.x * v1.z * v2.y +
                                     p2.y * v1.x * v2.z - p2.y * v1.z * v2.x - p2.z * v1.x * v2.y +
                                     p2.z * v1.y * v2.x) *
                                    den_inv;

                            float t =
                                    (p1.x * v1.y * v2.z - p1.x * v1.y * v3.z - p1.x * v1.z * v2.y + p1.x * v1.z * v3.y +
                                     p1.x * v2.y * v3.z -
                                     p1.x * v2.z * v3.y - p1.y * v1.x * v2.z + p1.y * v1.x * v3.z + p1.y * v1.z * v2.x -
                                     p1.y * v1.z * v3.x -
                                     p1.y * v2.x * v3.z + p1.y * v2.z * v3.x + p1.z * v1.x * v2.y - p1.z * v1.x * v3.y -
                                     p1.z * v1.y * v2.x +
                                     p1.z * v1.y * v3.x + p1.z * v2.x * v3.y - p1.z * v2.y * v3.x - v1.x * v2.y * v3.z +
                                     v1.x * v2.z * v3.y +
                                     v1.y * v2.x * v3.z - v1.y * v2.z * v3.x - v1.z * v2.x * v3.y +
                                     v1.z * v2.y * v3.x) *
                                    den_inv;

                            int collide = (u > 0.0) * (u < 1.0) * (v > 0.0) * (v < 1.0) *
                                          (t > 0.0) * (t < 1.0) * ((v + u) < 1.0);

                            if (collide > 0) {
                                atomicAdd(&done[0], collide);
                            }

                        }
                    }
                }
            }


        }
    }

    int CollisionChecker::load_verts(const std::vector<float> &verts) {
        std::vector<Triangle> triangles(verts.size() / 9);
        fillTriangles(verts, triangles);
        auto triangles_1_gpu = std::make_shared<GPUData<Triangle>>(triangles);

        int handle = verts_map_.size();
        verts_map_[handle] = triangles_1_gpu;
        return handle;
    }

    bool CollisionChecker::check_collision(int vert_handle_1, int vert_handle_2) {
        auto &triangles_1_gpu = verts_map_[vert_handle_1];
        auto &triangles_2_gpu = verts_map_[vert_handle_2];

        int blockSize = 256;
        int numThreads = (triangles_2_gpu->size_ * triangles_1_gpu->size_) / batch;
        int gridSize = (numThreads + blockSize - 1) / blockSize;

        checkCollision<<<gridSize, blockSize>>>(triangles_1_gpu->data_gpu_, triangles_2_gpu->data_gpu_,
                                                triangles_1_gpu->size_, triangles_2_gpu->size_, done_gpu_->data_gpu_);
        CUDA_CHECK_THROW(cudaGetLastError());

        auto done = done_gpu_->to_cpu();

        return done[0] >= 1;

    }

    CollisionChecker::CollisionChecker() {
        std::vector<float> done = {0};
        done_gpu_ = std::make_unique<GPUData<float>>(done);
    }

    std::shared_ptr<CollisionChecker> init() {
        auto ptr = std::make_shared<CollisionChecker>();
        return ptr;
    }
}