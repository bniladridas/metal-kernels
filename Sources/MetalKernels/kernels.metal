#include <metal_stdlib>
using namespace metal;

// ========== BASIC OPERATIONS ==========
// 1. Array addition
kernel void add_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] + b[id];
}

// 2. Element-wise multiplication
kernel void multiply_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] * b[id];
}

// 3. Scale/multiply by constant
kernel void scale_array(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    device const float& scale [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = input[id] * scale;
}

// 4. Absolute value
kernel void absolute_value(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = abs(input[id]);
}

// 5. ReLU activation
kernel void relu(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = max(0.0f, input[id]);
}

// 6. Matrix-vector multiply (rows in parallel)
kernel void matrix_vector_multiply(
    device const float* matrix [[buffer(0)]],
    device const float* vector [[buffer(1)]],
    device float* result [[buffer(2)]],
    device const uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        sum += matrix[row * cols + col] * vector[col];
    }
    result[row] = sum;
}

// ========== CUDA TRANSLATION EXAMPLES ==========
// CUDA: __global__ void vector_add(float *a, float *b, float *c, int n)
// Metal equivalent:
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}

// CUDA: Fused multiply-add (CUDA: a[i] * b[i] + c[i])
kernel void fused_multiply_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* result [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = fma(a[id], b[id], c[id]);  // fused multiply-add
}

// ========== SHARED MEMORY / THREADGROUP OPTIMIZATION ==========
// Reduction with shared memory (CUDA __shared__ equivalent)
kernel void sum_reduction(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Load data into shared memory
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = 1; stride < 32; stride *= 2) {
        if (lid % (stride * 2) == 0) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result from first thread
    if (lid == 0) {
        output[id / 32] = shared[0];
    }
}

// ========== IMAGE PROCESSING ==========
// 2D Convolution with 3x3 kernel
kernel void convolution_2d(
    device const float* input [[buffer(0)]],
    device const float* kernel_data [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint& width [[buffer(3)]],
    device const uint& height [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        return;
    }

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            uint ix = x + kx;
            uint iy = y + ky;
            float pixel = input[iy * width + ix];
            float k = kernel_data[(ky + 1) * 3 + (kx + 1)];
            sum += pixel * k;
        }
    }
    output[y * width + x] = sum;
}

// Gaussian blur (separable, x-direction)
kernel void gaussian_blur_x(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& width [[buffer(2)]],
    device const uint& height [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint x = gid.x;
    uint y = gid.y;

    if (x < 2 || x >= width - 2 || y >= height) return;

    float kernel_vals[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625};
    float sum = 0.0f;

    for (int i = -2; i <= 2; i++) {
        sum += input[y * width + x + i] * kernel_vals[i + 2];
    }

    output[y * width + x] = sum;
}

// ========== ML OPERATIONS ==========
// Softmax activation (optimized with threadgroup reduction)
kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Parallel reduction for max (assuming threadgroup_size = 32, n <= 1024)
    float local_max = -INFINITY;
    for (uint i = lid; i < n; i += 32) {
        local_max = max(local_max, input[i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max across threadgroup
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    // Parallel computation of sum_exp
    float local_sum = 0.0f;
    for (uint i = lid; i < n; i += 32) {
        local_sum += exp(input[i] - max_val);
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum across threadgroup
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared[0];

    if (id < n) {
        output[id] = exp(input[id] - max_val) / sum_exp;
    }
}

// Matrix multiply (general dimensions)
kernel void matrix_multiply(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const uint& m [[buffer(3)]],
    device const uint& k [[buffer(4)]],
    device const uint& n [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;

    if (i >= m || j >= n) return;

    float sum = 0.0f;
    for (uint p = 0; p < k; p++) {
        sum += a[i * k + p] * b[p * n + j];
    }
    c[i * n + j] = sum;
}

// ========== ADVANCED PATTERNS ==========
// Prefix scan (exclusive scan)
kernel void exclusive_scan(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    shared[lid] = (lid > 0) ? input[id - 1] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < 32; stride *= 2) {
        float val = (lid >= stride) ? shared[lid - stride] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lid] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[id] = shared[lid];
}

#define BLOCK_SIZE 32

// Tiling pattern (2D block multiplication)
kernel void tiled_matrix_multiply(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const uint& n [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]]
) {
    uint blockRow = gid.x / BLOCK_SIZE;
    uint blockCol = gid.y / BLOCK_SIZE;
    uint row = blockRow * BLOCK_SIZE + lid.x;
    uint col = blockCol * BLOCK_SIZE + lid.y;

    if (row >= n || col >= n) return;

    float sum = 0.0f;

    for (uint t = 0; t < n; t += BLOCK_SIZE) {
        // Load tile from A into shared memory
        tileA[lid.x * BLOCK_SIZE + lid.y] = a[row * n + t + lid.y];
        // Load tile from B into shared memory
        tileB[lid.x * BLOCK_SIZE + lid.y] = b[(t + lid.x) * n + col];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial sum using shared memory
        for (uint k = 0; k < BLOCK_SIZE; k++) {
            sum += tileA[lid.x * BLOCK_SIZE + k] * tileB[k * BLOCK_SIZE + lid.y];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[row * n + col] = sum;
}

// ========== NEURAL NETWORK LAYERS ==========

// Batch normalization
kernel void batch_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float& mean [[buffer(2)]],
    device const float& variance [[buffer(3)]],
    device const float& gamma [[buffer(4)]],
    device const float& beta [[buffer(5)]],
    device const float& epsilon [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    float normalized = (input[id] - mean) / sqrt(variance + epsilon);
    output[id] = gamma * normalized + beta;
}

// Sigmoid activation
kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}

// Tanh activation
kernel void tanh_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = tanh(input[id]);
}

// GELU approximation (fast)
kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    float cdf = 0.5f * (1.0f + tanh(sqrt(2.0f / 3.141592653589793f) * (x + 0.044715f * x * x * x)));
    output[id] = x * cdf;
}

// Convolution with batch (input: [batch][height][width][channels])
kernel void conv2d_batch(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint& batch [[buffer(4)]],
    device const uint& in_channels [[buffer(5)]],
    device const uint& out_channels [[buffer(6)]],
    device const uint& kernel_size [[buffer(7)]],
    device const uint& input_size [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.x;
    uint y = gid.y;
    uint x = gid.z;
    
    if (b >= batch || y >= input_size || x >= input_size) return;
    
    uint output_size = input_size - kernel_size + 1;
    if (y >= output_size || x >= output_size) return;

    for (uint oc = 0; oc < out_channels; oc++) {
        float sum = bias[oc];
        
        for (uint ic = 0; ic < in_channels; ic++) {
            for (uint ky = 0; ky < kernel_size; ky++) {
                for (uint kx = 0; kx < kernel_size; kx++) {
                    uint input_idx = ((b * in_channels + ic) * input_size + (y + ky)) * input_size + (x + kx);
                    uint weight_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
        
        uint output_idx = ((b * out_channels + oc) * output_size + y) * output_size + x;
        output[output_idx] = sum;
    }
}

// Depthwise separable convolution (efficient for mobile)
kernel void depthwise_conv2d(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint& channels [[buffer(4)]],
    device const uint& kernel_size [[buffer(5)]],
    device const uint& input_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint y = gid.x;
    uint x = gid.y;
    
    uint output_size = input_size - kernel_size + 1;
    if (y >= output_size || x >= output_size) return;

    for (uint c = 0; c < channels; c++) {
        float sum = bias[c];
        
        for (uint ky = 0; ky < kernel_size; ky++) {
            for (uint kx = 0; kx < kernel_size; kx++) {
                uint input_idx = (c * input_size + (y + ky)) * input_size + (x + kx);
                uint weight_idx = (c * kernel_size + ky) * kernel_size + kx;
                sum += input[input_idx] * weights[weight_idx];
            }
        }
        
        uint output_idx = (c * output_size + y) * output_size + x;
        output[output_idx] = sum;
    }
}

// Layer normalization (optimized with parallel reduction)
kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float& gamma [[buffer(2)]],
    device const float& beta [[buffer(3)]],
    device const uint& n [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    // Load input to shared memory
    shared[lid] = input[id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute local sum_x and sum_x2
    float local_sum_x = shared[lid];
    float local_sum_x2 = shared[lid] * shared[lid];

    // Parallel reduction for sum_x
    shared[lid] = local_sum_x;
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_x = shared[0];

    // Parallel reduction for sum_x2
    shared[lid] = local_sum_x2;
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_x2 = shared[0];

    // Compute mean and variance
    float mean = sum_x / 32.0f;
    float var = (sum_x2 - sum_x * sum_x / 32.0f) / 32.0f;

    float normalized = (input[id] - mean) / sqrt(var + 1e-5f);
    output[id] = gamma * normalized + beta;
}

// Batch processing: multiple elements per thread
kernel void matmul_batched(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const uint& batch_size [[buffer(3)]],
    device const uint& m [[buffer(4)]],
    device const uint& k [[buffer(5)]],
    device const uint& n [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    uint batch = id / (m * n);
    uint idx = id % (m * n);
    uint i = idx / n;
    uint j = idx % n;

    if (batch >= batch_size) return;

    float sum = 0.0f;
    for (uint p = 0; p < k; p++) {
        sum += a[(batch * m + i) * k + p] * b[(batch * k + p) * n + j];
    }
    c[(batch * m + i) * n + j] = sum;
}