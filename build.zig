const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("nengpu", .{
        .root_source_file = b.path("src/lib.zig"),
    });

    // Unit tests for nen-gpu
    const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("src/lib.zig"), .target = target, .optimize = optimize }) });
    tests.root_module.addImport("nengpu", mod);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run nen-gpu tests");
    test_step.dependOn(&run_tests.step);
}
