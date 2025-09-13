const std = @import("std");

pub const MetalContext = struct {
    // Placeholder for device/queue/state. Keep API stable for future work.
    initialized: bool = false,

    pub fn init() MetalContext {
        return .{ .initialized = true };
    }

    pub fn matmul_msl_source() []const u8 {
        // Future: compile this MSL at runtime once we wire Zig <-> Metal.
        return 
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void matmul_f32(
        \\  device const float* W [[buffer(0)]],
        \\  device const float* X [[buffer(1)]],
        \\  device float* OUT [[buffer(2)]],
        \\  constant uint& N [[buffer(3)]],
        \\  constant uint& D [[buffer(4)]],
        \\  uint gid [[thread_position_in_grid]] ) {
        \\  if (gid >= D) return;
        \\  float sum = 0.0f;
        \\  const device float* row = W + gid * N;
        \\  for (uint j = 0; j < N; ++j) sum += row[j] * X[j];
        \\  OUT[gid] = sum;
        \\}
        ;
    }
};

pub fn matmul_f32(_ctx: *const MetalContext, out: []f32, w: []const f32, x: []const f32, n: usize, d: usize) void {
    // Stub: fall back to CPU until Metal kernels are implemented.
    _ = _ctx;
    std.debug.assert(out.len == d);
    std.debug.assert(w.len == d * n);
    std.debug.assert(x.len == n);
    var i: usize = 0;
    while (i < d) : (i += 1) {
        var sum: f32 = 0;
        const row = w[i * n .. i * n + n];
        var j: usize = 0;
        while (j < n) : (j += 1) sum += row[j] * x[j];
        out[i] = sum;
    }
}
