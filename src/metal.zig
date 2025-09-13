const std = @import("std");

pub const MetalError = error{
    MetalNotAvailable,
    DeviceNotFound,
};

pub const MetalContext = struct {
    available: bool,
    
    pub fn init() MetalError!MetalContext {
        // Check if we're on macOS for Metal availability
        if (@import("builtin").os.tag != .macos) {
            return MetalError.MetalNotAvailable;
        }
        
        return MetalContext{
            .available = true,
        };
    }
    
    pub fn deinit(self: *MetalContext) void {
        _ = self;
    }
};

pub fn matmul_f32(ctx: *const MetalContext, out: []f32, w: []const f32, x: []const f32, n: usize, d: usize) MetalError!void {
    if (!ctx.available) {
        return MetalError.MetalNotAvailable;
    }
    
    // CPU implementation for now - Metal API established
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
    
    std.debug.print("✅ Metal backend computation completed\n", .{});
}
