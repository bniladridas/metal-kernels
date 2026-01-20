# Contributing to Metal Compute Kernels

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. **Fork the repository**

   ```bash
   git clone https://github.com/bniladridas/metal-kernels.git
   cd metal-kernels
   ```

2. **Create a feature branch**

   ```bash
   git checkout -b feature/your-kernel-name
   ```

3. **Build and test**

   ```bash
   swift build
   swift run MetalKernels
   ```

## Areas for Contribution

### 1. New Kernels

Add new Metal compute kernels for:

* RNN layers (LSTM, GRU)
* Transformer attention
* Quantization operations
* Custom loss functions
* Physics simulations

**Steps:**

1. Add Metal kernel code to the `metalCode` string in `MetalCompute.init()`
2. Add Swift host function in `MetalCompute` class
3. Add test case in `main.swift`
4. Document in `README.md`

### 2. Performance Optimization

* Profile existing kernels with Metal Debugger
* Optimize memory access patterns
* Reduce kernel launch overhead
* Add threadgroup memory optimizations
* Vectorize operations

### 3. Documentation

* Write kernel implementation guides
* Create usage examples
* Benchmark different chip generations
* Document best practices

### 4. Testing

* Add comprehensive test suite
* Benchmark against CPU and other GPU libraries
* Test on multiple Mac/iPad models
* Verify edge cases (NaN, infinity, zero)

### 5. Cross-platform Support

* Vulkan compute backend for non-Apple devices
* Metal on iOS, tvOS, macOS variants
* Support for older Metal versions

## Coding Style

### Swift Code

```swift
// Use clear variable names
let inputBuffer = device.makeBuffer(bytes: input, length: bufferSize)

// Add comments for complex logic
// Parallel reduction: log(n) steps with threadgroup_barrier
for stride in 1..<32 {
    if lid >= stride {
        shared[lid] += shared[lid - stride]
    }
    threadgroup_barrier(mem_flags::mem_threadgroup)
}

// Group related functions with MARK comments
// MARK: - Neural Network Layers
```

### Metal (MSL) Code

```metal
// Use clear parameter names and attributes
kernel void my_operation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& param [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // Implementation
}

// Add comments for non-obvious operations
float normalized = (input[id] - mean) / sqrt(var + epsilon);
```

## Commit Message Format

```
[Category] Brief description (50 chars max)

Longer description explaining the change, why it was needed,
and any important implementation details. Wrap at 72 characters.

Related issues: #123
```

**Categories:**

* `[Feature]` - New kernel or capability
* `[Fix]` - Bug fix
* `[Perf]` - Performance improvement
* `[Docs]` - Documentation update
* `[Test]` - Test additions
* `[Refactor]` - Code reorganization

Example:

```
[Feature] Add LSTM kernel for sequence processing

Implements long short-term memory cell with peephole connections.
Uses 4 matrix multiplications per timestep. Supports batching.
Tested on M1/M2/M3 with 100x speedup over CPU.

Related issues: #42
```

## Pull Request Process

1. **Create PR with clear title and description**

2. **Ensure code builds**

   ```bash
   swift build -v
   swift run MetalKernels
   ```

3. **Add tests**

4. **Update documentation**

5. **Wait for review**

## Performance Benchmarking

Include benchmarks comparing GPU and CPU performance.

## Debugging Tips

### Metal Shader Compilation Errors

* Check syntax
* Verify buffer indices
* Ensure threadgroup memory
* Look for reserved keyword conflicts

### Runtime Issues

* Check `MTLBuffer` sizes
* Verify buffer indices
* Use `waitUntilCompleted()` for debugging
* Print thread counts

## Testing on Multiple Devices

* [ ] M1 MacBook Air
* [ ] M2 MacBook Pro
* [ ] M3 MacBook Pro
* [ ] M1/M2 iPad Pro
* [ ] iPhone 15 Pro (A17)

## Documentation Requirements

Every kernel needs:

1. In-code comments
2. README.md entry with example
3. Performance table
4. Use case description

## Licensing

All contributions are under the MIT License.

## Questions?

* GitHub Issues: for bugs and features
* GitHub Discussions: for questions and ideas
* Email: [harpertoken@icloud.com](mailto:harpertoken@icloud.com)

---

Thank you for contributing!