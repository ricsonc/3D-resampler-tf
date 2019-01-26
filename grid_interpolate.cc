#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
typedef TTypes<float>::ConstFlat FlatT;

REGISTER_OP("GridInterpolate3D")
    .Input("source: float") 
    .Input("grid: float")
    .Attr("soft_boundary: bool = false")
    .Output("interpolated: float");

void GridInterpolate3DKernelLauncher(
    const float* source,
    const float* grid,
    float* output,
    const TensorShape &s,
    const TensorShape &g,
    const bool soft_boundary);

class GridInterpolate3DOpGPU : public OpKernel {
public:
    explicit GridInterpolate3DOpGPU(OpKernelConstruction* context) : OpKernel(context) {
        context->GetAttr("soft_boundary", &soft_boundary);
    }

    void Compute(OpKernelContext* context) override {
        //printf("calling gpu kernel\n");
      
        // Grab the input tensor
        const Tensor& source_tensor = context->input(0);
        auto source_flat = source_tensor.flat<float>();

        const Tensor& grid_tensor = context->input(1);
        auto grid_flat = grid_tensor.flat<float>();
        
        //Check shapes...
        OP_REQUIRES(context, source_tensor.shape().dims() == 5,
                    errors::InvalidArgument("bad src rank")); 
        OP_REQUIRES(context, grid_tensor.shape().dims() == 5,
                    errors::InvalidArgument("bad grid rank"));
        
        for(int d = 0; d < 4; d++){
            OP_REQUIRES(context, source_tensor.dim_size(d) == grid_tensor.dim_size(d),
                        errors::InvalidArgument("src and grid dont match"));
        }
        OP_REQUIRES(context, grid_tensor.dim_size(4) == 3,
                    errors::InvalidArgument("indices not size 3"));
    
        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, source_tensor.shape(), &output_tensor));
        auto output_flat = output_tensor->flat<float>();

        GridInterpolate3DKernelLauncher(
            source_flat.data(),
            grid_flat.data(),
            output_flat.data(),
            source_tensor.shape(),
            grid_tensor.shape(),
            this->soft_boundary);
    }
private:
    bool soft_boundary;
    
};

REGISTER_KERNEL_BUILDER(Name("GridInterpolate3D").Device(DEVICE_GPU), GridInterpolate3DOpGPU);

//3D gradients

REGISTER_OP("GridInterpolate3DGrad")
    .Input("grad: float")
    .Input("source: float") 
    .Input("grid: float")
    .Attr("soft_boundary: bool")
    .Output("grad_source: float")
    .Output("grad_grid: float");

void GridInterpolate3DGradKernelLauncher(
    const float* grad,
    const float* source,
    const float* grid,
    float* grad_source,
    float* grad_grid, 
    const TensorShape &s,
    const TensorShape &g,
    const bool soft_boundary);

class GridInterpolate3DGradOpGPU : public OpKernel {
public:
    explicit GridInterpolate3DGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
        context->GetAttr("soft_boundary", &soft_boundary);
    }

    void Compute(OpKernelContext* context) override {
        //printf("calling gpu grad\n");
      
        // Grab the input tensor
        const Tensor& grad_tensor = context->input(0);
        auto grad_flat = grad_tensor.flat<float>();
    
        const Tensor& source_tensor = context->input(1);
        auto source_flat = source_tensor.flat<float>();

        const Tensor& grid_tensor = context->input(2);
        auto grid_flat = grid_tensor.flat<float>();
        
        //Check shapes...
        OP_REQUIRES(context, grad_tensor.shape().dims() == 5,
                    errors::InvalidArgument("bad grad rank")); 
        OP_REQUIRES(context, source_tensor.shape().dims() == 5,
                    errors::InvalidArgument("bad src rank")); 
        OP_REQUIRES(context, grid_tensor.shape().dims() == 5,
                    errors::InvalidArgument("bad grid rank"));
    
        for(int d = 0; d < 4; d++){
            OP_REQUIRES(context, source_tensor.dim_size(d) == grid_tensor.dim_size(d),
                        errors::InvalidArgument("src and grid dont match"));
            OP_REQUIRES(context, grad_tensor.dim_size(d) == grid_tensor.dim_size(d),
                        errors::InvalidArgument("src and grid dont match"));
        }
        OP_REQUIRES(context, grid_tensor.dim_size(4) == 3,
                    errors::InvalidArgument("indices not size 3"));
    
        // Create output tensors
        Tensor* grad_source = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, source_tensor.shape(), &grad_source));
        auto grad_source_flat = grad_source->flat<float>();

        Tensor* grad_grid = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, grid_tensor.shape(), &grad_grid));
        auto grad_grid_flat = grad_grid->flat<float>();
    
        GridInterpolate3DGradKernelLauncher(
            grad_flat.data(),
            source_flat.data(),
            grid_flat.data(),
            grad_source_flat.data(),
            grad_grid_flat.data(),
            source_tensor.shape(),
            grid_tensor.shape(),
            this->soft_boundary);
    }

private:
    bool soft_boundary;
};

REGISTER_KERNEL_BUILDER(Name("GridInterpolate3DGrad").Device(DEVICE_GPU), GridInterpolate3DGradOpGPU);

