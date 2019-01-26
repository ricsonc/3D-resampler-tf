#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
typedef TTypes<float>::ConstFlat FlatT;


void checkerr(cudaError_t err){
    if (err != cudaSuccess)
    {
        printf("cuda error: \"%s\".\n", cudaGetErrorString(err));
    }
}


int updiv(int num, int quot) {
    return (num+quot-1)/quot;
}


int4 sliceShapeToint4(const TensorShape &x) {
    return {(int) x.dim_size(1),
            (int) x.dim_size(2),
            (int) x.dim_size(3),
            (int) x.dim_size(4)};
}


__device__ int linear_index_3D(const int b, const int d, const int h, const int w,
                               const int c, const int4 &s) {
    const int col_stride = s.w;
    const int row_stride = s.z*col_stride;
    const int slice_stride = s.y*row_stride;
    const int batch_stride = s.x*slice_stride;
    return b*batch_stride + d*slice_stride + h*row_stride + w*col_stride + c;
}


__device__ float interpolateValue3D(const float* source, int4 s,
                                    int b, int c, float z, float y, float x, const bool soft) {

    //printf("%f %f %f %d %d\n", z, y, x, b, c);
    int zl = static_cast<int>(floor(z));
    int yl = static_cast<int>(floor(y));
    int xl = static_cast<int>(floor(x));
    int zu = zl+1;
    int yu = yl+1;
    int xu = xl+1;

    int D = s.x; //pay attention!
    int H = s.y; 
    int W = s.z;

    float dz = z - floor(z);
    float dx = x - floor(x);
    float dy = y - floor(y);

    // begin boundary mode //
    
    bool zlb = 0 <= zl && zl <= D-1;
    bool zub = 0 <= zu && zu <= D-1;
    bool ylb = 0 <= yl && yl <= H-1;
    bool yub = 0 <= yu && yu <= H-1;
    bool xlb = 0 <= xl && xl <= W-1;
    bool xub = 0 <= xu && xu <= W-1;
    
    float b000 = static_cast<float>(zub && yub && xub);
    float b001 = static_cast<float>(zub && yub && xlb);
    float b010 = static_cast<float>(zub && ylb && xub);
    float b011 = static_cast<float>(zub && ylb && xlb);
    float b100 = static_cast<float>(zlb && yub && xub);
    float b101 = static_cast<float>(zlb && yub && xlb);
    float b110 = static_cast<float>(zlb && ylb && xub);
    float b111 = static_cast<float>(zlb && ylb && xlb);

    // end boundary mode //
    
    zl = zl < 0 ? 0 : (zl >= D ? D-1 : zl);
    zu = zu < 0 ? 0 : (zu >= D ? D-1 : zu);
    yl = yl < 0 ? 0 : (yl >= H ? H-1 : yl);
    yu = yu < 0 ? 0 : (yu >= H ? H-1 : yu);
    xl = xl < 0 ? 0 : (xl >= W ? W-1 : xl);
    xu = xu < 0 ? 0 : (xu >= W ? W-1 : xu);

    float v000 = source[linear_index_3D(b, zl, yl, xl, c, s)];
    float v001 = source[linear_index_3D(b, zl, yl, xu, c, s)];
    float v010 = source[linear_index_3D(b, zl, yu, xl, c, s)];
    float v011 = source[linear_index_3D(b, zl, yu, xu, c, s)];
    float v100 = source[linear_index_3D(b, zu, yl, xl, c, s)];
    float v101 = source[linear_index_3D(b, zu, yl, xu, c, s)];
    float v110 = source[linear_index_3D(b, zu, yu, xl, c, s)];
    float v111 = source[linear_index_3D(b, zu, yu, xu, c, s)];

    float w000 = (1-dz)*(1-dy)*(1-dx);
    float w001 = (1-dz)*(1-dy)*dx;
    float w010 = (1-dz)*dy*(1-dx);
    float w011 = (1-dz)*dy*dx;
    float w100 = dz*(1-dy)*(1-dx);
    float w101 = dz*(1-dy)*dx;
    float w110 = dz*dy*(1-dx);
    float w111 = dz*dy*dx;

    if (!soft) {
        w000 *= b000;
        w001 *= b001;
        w010 *= b010;
        w011 *= b011;
        w100 *= b100;
        w101 *= b101;
        w110 *= b110;
        w111 *= b111;
    }
    
    return (w000*v000 + w001*v001 + w010*v010 + w011*v011 +
            w100*v100 + w101*v101 + w110*v110 + w111*v111);
}


__global__ void InterpolateKernel3D(
    const float* source,
    const float* grid,
    float* output,
    const int4 s,
    const int4 g,
    const int B,
    const bool soft)
{
    int tmp = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tmp / s.x;
    int i = tmp % s.x; // z x y
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    float z = grid[linear_index_3D(b, i, j, k, 0, g)];
    float y = grid[linear_index_3D(b, i, j, k, 1, g)];
    float x = grid[linear_index_3D(b, i, j, k, 2, g)];

    if (i >= s.x || j >= s.y || k >= s.z || b >= B) {
        return;
    }

    for(int c = 0; c < s.w; c++) {
        output[linear_index_3D(b, i, j, k, c, s)] = \
            interpolateValue3D(source, s, b, c, z, y, x, soft);
    }
}


void GridInterpolate3DKernelLauncher(
    const float* source,
    const float* grid,
    float* output,
    const TensorShape &s,
    const TensorShape &g,
    const bool soft)
{
    int B = s.dim_size(0);
    int TD = 1;
    dim3 threads(TD, TD, TD);
    dim3 blocks(
        updiv(B*s.dim_size(1), TD),
        updiv(s.dim_size(2), TD),
        updiv(s.dim_size(3), TD));
    
    InterpolateKernel3D<<<blocks, threads>>>(
        source, grid, output,
        sliceShapeToint4(s), sliceShapeToint4(g), B, soft);
    
    checkerr(cudaPeekAtLastError());
    checkerr(cudaDeviceSynchronize());
}

//for the gradient

__device__ void computeGrad3D(
    const float grad_out,
    const float* source,
    float* grad_source,
    float* grad_grid,
    float3 & tmp_grid,
    const int4 s, const int4 g, 
    const int b,
    const int c,
    const float z, const float y, const float x,
    const bool soft)
{
    //printf("%f %f %f %d %d\n", z, y, x, b, c);
    int zl = static_cast<int>(floor(z));
    int yl = static_cast<int>(floor(y));
    int xl = static_cast<int>(floor(x));
    int zu = zl+1;
    int yu = yl+1;
    int xu = xl+1;

    int D = s.x; //pay attention!
    int H = s.y; 
    int W = s.z;

    float dz = z - floor(z);
    float dx = x - floor(x);
    float dy = y - floor(y);

    // begin boundary mode //
    
    bool zlb = 0 <= zl && zl <= D-1;
    bool zub = 0 <= zu && zu <= D-1;
    bool ylb = 0 <= yl && yl <= H-1;
    bool yub = 0 <= yu && yu <= H-1;
    bool xlb = 0 <= xl && xl <= W-1;
    bool xub = 0 <= xu && xu <= W-1;
    
    float b000 = static_cast<float>(zub && yub && xub);
    float b001 = static_cast<float>(zub && yub && xlb);
    float b010 = static_cast<float>(zub && ylb && xub);
    float b011 = static_cast<float>(zub && ylb && xlb);
    float b100 = static_cast<float>(zlb && yub && xub);
    float b101 = static_cast<float>(zlb && yub && xlb);
    float b110 = static_cast<float>(zlb && ylb && xub);
    float b111 = static_cast<float>(zlb && ylb && xlb);

    // end boundary mode //

    zl = zl < 0 ? 0 : (zl >= D ? D-1 : zl);
    zu = zu < 0 ? 0 : (zu >= D ? D-1 : zu);
    yl = yl < 0 ? 0 : (yl >= H ? H-1 : yl);
    yu = yu < 0 ? 0 : (yu >= H ? H-1 : yu);
    xl = xl < 0 ? 0 : (xl >= W ? W-1 : xl);
    xu = xu < 0 ? 0 : (xu >= W ? W-1 : xu);

    int i000 = linear_index_3D(b, zl, yl, xl, c, s);
    int i001 = linear_index_3D(b, zl, yl, xu, c, s);
    int i010 = linear_index_3D(b, zl, yu, xl, c, s);
    int i011 = linear_index_3D(b, zl, yu, xu, c, s);
    int i100 = linear_index_3D(b, zu, yl, xl, c, s);
    int i101 = linear_index_3D(b, zu, yl, xu, c, s);
    int i110 = linear_index_3D(b, zu, yu, xl, c, s);
    int i111 = linear_index_3D(b, zu, yu, xu, c, s);

    float v000 = source[i000];
    float v001 = source[i001];
    float v010 = source[i010];
    float v011 = source[i011];
    float v100 = source[i100];
    float v101 = source[i101];
    float v110 = source[i110];
    float v111 = source[i111];

    float w000 = (1-dz)*(1-dy)*(1-dx);
    float w001 = (1-dz)*(1-dy)*dx;
    float w010 = (1-dz)*dy*(1-dx);
    float w011 = (1-dz)*dy*dx;
    float w100 = dz*(1-dy)*(1-dx);
    float w101 = dz*(1-dy)*dx;
    float w110 = dz*dy*(1-dx);
    float w111 = dz*dy*dx;

    if (!soft) {
        w000 *= b000;
        w001 *= b001;
        w010 *= b010;
        w011 *= b011;
        w100 *= b100;
        w101 *= b101;
        w110 *= b110;
        w111 *= b111;

        //this is a dirty hack to get the correct gradients wrt grid
        v000 *= b000;
        v001 *= b001;
        v010 *= b010;
        v011 *= b011;
        v100 *= b100;
        v101 *= b101;
        v110 *= b110;
        v111 *= b111;
    }

    //gradients wrt grid
    tmp_grid.x += grad_out * \
        (+ (v100 - v000) * (1-dy)*(1-dx)
         + (v101 - v001) * (1-dy)*dx
         + (v110 - v010) * dy*(1-dx)
         + (v111 - v011) * dy*dx);
    tmp_grid.y += grad_out * \
        (+ (v010 - v000) * (1-dz)*(1-dx)
         + (v011 - v001) * (1-dz)*dx
         + (v110 - v100) * dz*(1-dx)
         + (v111 - v101) * dz*dx);
    tmp_grid.z += grad_out * \
        (+ (v001 - v000) * (1-dz)*(1-dy)
         + (v011 - v010) * (1-dz)*dy
         + (v101 - v100) * dz*(1-dy)
         + (v111 - v110) * dz*dy);

    //gradients wrt 8 source elements
    atomicAdd(grad_source + i000, w000 * grad_out);
    atomicAdd(grad_source + i001, w001 * grad_out);
    atomicAdd(grad_source + i010, w010 * grad_out);
    atomicAdd(grad_source + i011, w011 * grad_out);
    atomicAdd(grad_source + i100, w100 * grad_out);
    atomicAdd(grad_source + i101, w101 * grad_out);
    atomicAdd(grad_source + i110, w110 * grad_out);
    atomicAdd(grad_source + i111, w111 * grad_out);
}
    
__global__ void InterpolateKernel3DGrad(
    const float* grad,
    const float* source,
    const float* grid,
    float* grad_source,
    float* grad_grid,
    const int4 s,
    const int4 g,
    const int B,
    const bool soft)
{
    int tmp = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tmp / s.x;
    int i = tmp % s.x; // z x y
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    float z = grid[linear_index_3D(b, i, j, k, 0, g)];
    float y = grid[linear_index_3D(b, i, j, k, 1, g)];
    float x = grid[linear_index_3D(b, i, j, k, 2, g)];
    
    if (i >= s.x || j >= s.y || k >= s.z || b >= B) {
        return;
    }

    float3 tmp_grid = {0,0,0};
    
    for(int c = 0; c < s.w; c++) {
        float grad_out = grad[linear_index_3D(b, i, j, k, c, s)];
        computeGrad3D(grad_out, source, grad_source, grad_grid, tmp_grid,
                      s, g, b, c, z, y, x, soft);
    }

    grad_grid[linear_index_3D(b, i, j, k, 0, g)] = tmp_grid.x;
    grad_grid[linear_index_3D(b, i, j, k, 1, g)] = tmp_grid.y;
    grad_grid[linear_index_3D(b, i, j, k, 2, g)] = tmp_grid.z;

}

void GridInterpolate3DGradKernelLauncher(
    const float* grad,
    const float* source,
    const float* grid,
    float* grad_source,
    float* grad_grid,
    const TensorShape &s,
    const TensorShape &g,
    const bool soft)
{
    int B = s.dim_size(0);
    int TD = 1;
    dim3 threads(TD, TD, TD);
    dim3 blocks(
        updiv(B*s.dim_size(1), TD),
        updiv(s.dim_size(2), TD),
        updiv(s.dim_size(3), TD));

    checkerr(cudaMemset(grad_source, 0, s.num_elements()*sizeof(float)));
    checkerr(cudaMemset(grad_grid, 0, g.num_elements()*sizeof(float)));
    
    InterpolateKernel3DGrad<<<blocks, threads>>>(
        grad, source, grid, grad_source, grad_grid,
        sliceShapeToint4(s), sliceShapeToint4(g), B, soft);
    
    checkerr(cudaPeekAtLastError());
    checkerr(cudaDeviceSynchronize());
}
