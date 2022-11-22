#include <bits/stdc++.h>
#include <experimental/random>
#include <cuda.h>

#define SIZE 32
#define WIDTH SIZE * sizeof(int)

int* gen_rand();
int* mul_matrices(int* m_1, int* m_2);
void print(int* m);
bool are_equal(int* a, int* b);

__device__ inline void count_sum(int *tmp)
{
    const auto i = threadIdx.y * SIZE + threadIdx.x;
    constexpr int l_count = 5;
    
    auto sum = tmp[i];
    unsigned mask = 0xffff;
    for (int i = 1; i <= l_count; ++i)
    {
        mask >>= i;
        mask &= 0xAAAA;
        const auto offset = 1 << (i - 1);
        sum += __shfl_xor_sync(mask, sum, offset);
    }
    tmp[i] = sum;
    __syncwarp();
}

__device__ inline int* row_i(int* m, int i, size_t pitch)
{
    char* m_c = (char*)m;
    m_c += pitch * i;
    return (int*)m_c;
}

__global__ void mul_matrices_kernel(int* a, int* b, int* res, size_t p_a, size_t p_b, size_t p_res)
{
    /// copy the whole matrix into shared
    __shared__ int b_sh[SIZE * SIZE];
    __shared__ int c_sh[SIZE * SIZE];
    const int index = threadIdx.y * SIZE + threadIdx.x;
    b_sh[index] = row_i(b, threadIdx.y, p_b)[threadIdx.x];

    __syncthreads();
    /// fill columns by single row from a
    __shared__ int tmp[SIZE];
    if (threadIdx.y == 0)
    {
        tmp[threadIdx.x] = row_i(a, blockIdx.x, p_a)[threadIdx.x];
    }

    __syncthreads();

    /// mul one by one
    c_sh[index] = tmp[threadIdx.x] * b_sh[threadIdx.x * SIZE + threadIdx.y];

    count_sum(c_sh);
    __syncthreads();
    if (threadIdx.y == 0)
    {
        row_i(res, blockIdx.x, p_res)[threadIdx.x] = c_sh[threadIdx.x * SIZE];
    }
    __syncthreads();
}

void _check_err(cudaError_t code)
{
    if (code == cudaSuccess)
    {
        return;
    }
    std::cerr << "\n\n" << cudaGetErrorString(code) << "\n" << std::endl;
    exit(code);
}

int main()
{
    int* m_1 = gen_rand();
    int* m_2 = gen_rand();
    int* res = (int*)calloc(SIZE * SIZE, sizeof(int));

    int* dm_1 = nullptr;
    int* dm_2 = nullptr;
    int* dm_res = nullptr;
    
    size_t pitch_1{};
    size_t pitch_2{};
    size_t pitch_res{};

    cudaMallocPitch(&dm_1, &pitch_1, WIDTH, SIZE);
    cudaMallocPitch(&dm_2, &pitch_2, WIDTH, SIZE);
    cudaMallocPitch(&dm_res, &pitch_res, WIDTH, SIZE);

    cudaMemcpy2D(dm_1, pitch_1, m_1, WIDTH, WIDTH, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dm_2, pitch_2, m_2, WIDTH, WIDTH, SIZE, cudaMemcpyHostToDevice);

    dim3 block;
    block.x = SIZE;
    block.y = SIZE;
    block.z = 1;
    dim3 grid;
    grid.x = SIZE;
    grid.y = 1;
    grid.z = 1;
    mul_matrices_kernel<<<grid, block>>>(dm_1, dm_2, dm_res, pitch_1, pitch_2, pitch_res);
    _check_err(cudaGetLastError());

    cudaMemcpy2D(res, WIDTH, dm_res, pitch_res, WIDTH, SIZE, cudaMemcpyDeviceToHost);
    _check_err(cudaGetLastError());

    cudaDeviceSynchronize();

    int* expected_result = mul_matrices(m_1, m_2);
    if (!are_equal(expected_result, res))
    {
        std::cout << "expected" << std::endl;
        print(expected_result);
        std::cout << "received" << std::endl;
        print(res);
        std::cerr << "\nfailure" << std::endl;
    }
    else
    {
        std::cout << "success" << std::endl;
    }
    free(m_1);
    free(m_2);
    free(res);
    cudaFree(dm_1);
    cudaFree(dm_2);
    cudaFree(dm_res);
    return 0;
}

int* gen_rand()
{
    int* m = (int*)malloc(SIZE * WIDTH);
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            m[i * SIZE + j] = 1;///std::experimental::randint(0, 1);
        }
    }
    return m;
}

bool are_equal(int* a, int* b)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            if (a[i * SIZE + j] != b[i * SIZE + j])
            {
                return false;
            }
        }
    }
    return true;
}

void print(int* m)
{
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            std::cout << m[i * SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
}

int* mul_matrices(int* m_1, int* m_2)
{
    int* res = (int*)malloc(SIZE * WIDTH);
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            res[i * SIZE + j] = 0;
            for (int k = 0; k < SIZE; ++k)
            {
                res[i * SIZE + j] += m_1[i * SIZE + k] * m_2[k * SIZE + j];
            }
        }
    }

    return res;
}

