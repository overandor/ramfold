#include <metal_stdlib>
using namespace metal;

kernel void bandwidth_copy(
    device const float* a [[buffer(0)]],
    device float* c [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    c[id] = a[id] * 1.0001f;
}

kernel void fma_hot(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    float x = a[id];
    float y = b[id];
    float z = x * y + 0.1f;
    z = z * 1.0001f + x;
    z = z * 0.9999f + y;
    z = z * y + 0.00001f;
    c[id] = z;
}

kernel void saxpy(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] * 1.0001f + b[id];
}
