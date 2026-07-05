import Foundation
import Metal

func now() -> Double { CFAbsoluteTimeGetCurrent() }

let shader = """
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
"""

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device available")
}

let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(source: shader, options: nil)
let bwPSO = try device.makeComputePipelineState(function: library.makeFunction(name: "bandwidth_copy")!)
let fmaPSO = try device.makeComputePipelineState(function: library.makeFunction(name: "fma_hot")!)

print("RAMFold Metal Hot-Tensor Probe")
print("device=\(device.name)")
print("unifiedMemory=\(device.hasUnifiedMemory)")
print("recommendedMaxWorkingSetMB=\(device.recommendedMaxWorkingSetSize / 1024 / 1024)")
print("kind,elements,iterations,seconds,estimated_GBps,estimated_GOPS")

func makeBuffer(count: Int, fill: Float) -> MTLBuffer {
    let buf = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: [.storageModeShared])!
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = fill + Float(i % 17) * 0.001 }
    return buf
}

func run(kind: String, n: Int, iterations: Int) {
    let a = makeBuffer(count: n, fill: 1.0)
    let b = makeBuffer(count: n, fill: 2.0)
    let c = makeBuffer(count: n, fill: 0.0)
    var nn = UInt32(n)
    let nbuf = device.makeBuffer(bytes: &nn, length: MemoryLayout<UInt32>.stride, options: [.storageModeShared])!
    let pso = kind == "fma" ? fmaPSO : bwPSO
    let threads = MTLSize(width: min(256, pso.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    let groups = MTLSize(width: (n + threads.width - 1) / threads.width, height: 1, depth: 1)

    func encodeOnce() {
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        if kind == "fma" {
            enc.setBuffer(a, offset: 0, index: 0)
            enc.setBuffer(b, offset: 0, index: 1)
            enc.setBuffer(c, offset: 0, index: 2)
            enc.setBuffer(nbuf, offset: 0, index: 3)
        } else {
            enc.setBuffer(a, offset: 0, index: 0)
            enc.setBuffer(c, offset: 0, index: 1)
            enc.setBuffer(nbuf, offset: 0, index: 2)
        }
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    for _ in 0..<3 { encodeOnce() }
    let t0 = now()
    for _ in 0..<iterations { encodeOnce() }
    let dt = now() - t0

    let bytesPerElement = kind == "fma" ? 12.0 : 8.0
    let opsPerElement = kind == "fma" ? 8.0 : 1.0
    let gbps = Double(n) * bytesPerElement * Double(iterations) / dt / 1_000_000_000.0
    let gops = Double(n) * opsPerElement * Double(iterations) / dt / 1_000_000_000.0
    print("\(kind),\(n),\(iterations),\(String(format: "%.4f", dt)),\(String(format: "%.2f", gbps)),\(String(format: "%.2f", gops))")
}

for n in [1 << 20, 4 << 20, 16 << 20, 64 << 20] {
    autoreleasepool { run(kind: "bandwidth", n: n, iterations: 25) }
    autoreleasepool { run(kind: "fma", n: n, iterations: 25) }
}
