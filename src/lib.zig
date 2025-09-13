const std = @import("std");
const metal = @import("metal.zig");

pub const Backend = enum { cpu, metal };

pub const Kernels = struct {
    pub fn matmul_f32(out: []f32, w: []const f32, x: []const f32, n: usize, d: usize) void {
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

    pub fn rmsnorm_f32(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
        const n = x.len;
        std.debug.assert(out.len == n and weight.len == n and n > 0);
        var sum_sq: f32 = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const v = x[i];
            sum_sq += v * v;
        }
        const mean_sq = sum_sq / @as(f32, @floatFromInt(n));
        const denom = std.math.sqrt(mean_sq + eps);
        std.debug.assert(denom > 0);
        const scale = 1.0 / denom;
        i = 0;
        while (i < n) : (i += 1) out[i] = x[i] * scale * weight[i];
    }

    pub fn softmax_inplace_f32(x: []f32) void {
        if (x.len == 0) return;
        var max_val: f32 = x[0];
        var i: usize = 1;
        while (i < x.len) : (i += 1) {
            if (x[i] > max_val) max_val = x[i];
        }
        var sum_exp: f32 = 0;
        i = 0;
        while (i < x.len) : (i += 1) {
            const e = std.math.exp(x[i] - max_val);
            x[i] = e;
            sum_exp += e;
        }
        std.debug.assert(sum_exp > 0);
        const inv_sum = 1.0 / sum_exp;
        i = 0;
        while (i < x.len) : (i += 1) x[i] *= inv_sum;
    }
};

pub fn detect_backend() Backend {
    // macOS Metal stub; Wire real availability later.
    if (std.builtin.os.tag == .macos) return .metal;
    return .cpu;
}

pub fn matmul_f32_backend(backend: Backend, out: []f32, w: []const f32, x: []const f32, n: usize, d: usize) void {
    switch (backend) {
        .cpu => Kernels.matmul_f32(out, w, x, n, d),
        .metal => {
            var ctx = metal.MetalContext.init();
            metal.matmul_f32(&ctx, out, w, x, n, d);
        },
    }
}

test "cpu matmul matches reference" {
    var w: [16]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; // 4x4
    var x: [4]f32 = .{ 1, 0, 1, 0 };
    var out: [4]f32 = undefined;
    Kernels.matmul_f32(out[0..], w[0..], x[0..], 4, 4);
    try std.testing.expectApproxEqAbs(4, out[0], 1e-6);
    try std.testing.expectApproxEqAbs(12, out[1], 1e-6);
    try std.testing.expectApproxEqAbs(20, out[2], 1e-6);
    try std.testing.expectApproxEqAbs(28, out[3], 1e-6);
}

test "metal backend matmul matches cpu" {
    const n: usize = 5;
    const d: usize = 3;
    var w: [d * n]f32 = .{0} ** (d * n);
    var x: [n]f32 = .{0} ** n;
    var i: usize = 0;
    while (i < w.len) : (i += 1) w[i] = @floatFromInt(i + 1);
    i = 0;
    while (i < x.len) : (i += 1) x[i] = @floatFromInt((i % 3) + 1);
    var out_cpu: [d]f32 = undefined;
    var out_metal: [d]f32 = undefined;
    Kernels.matmul_f32(out_cpu[0..], w[0..], x[0..], n, d);
    matmul_f32_backend(.metal, out_metal[0..], w[0..], x[0..], n, d);
    i = 0;
    while (i < d) : (i += 1) try std.testing.expectApproxEqAbs(out_cpu[i], out_metal[i], 1e-6);
}

test "rmsnorm basic properties" {
    var x: [4]f32 = .{ 1.0, -1.0, 2.0, -2.0 };
    var w: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 };
    var out: [4]f32 = undefined;
    Kernels.rmsnorm_f32(out[0..], x[0..], w[0..], 1e-5);
    try std.testing.expectApproxEqAbs(0.63245, out[0], 1e-3);
    try std.testing.expectApproxEqAbs(-0.63245, out[1], 1e-3);
}

test "softmax sums to one and stable" {
    var v: [5]f32 = .{ 10.0, 0.0, -5.0, 10.0, -20.0 };
    Kernels.softmax_inplace_f32(v[0..]);
    var sum: f32 = 0;
    for (v) |e| sum += e;
    try std.testing.expect(@abs(sum - 1.0) < 1e-5);
    try std.testing.expect(@abs(v[0] - v[3]) < 1e-6);
}
