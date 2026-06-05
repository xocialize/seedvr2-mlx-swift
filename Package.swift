// swift-tools-version: 5.9
//
// seedvr2-mlx-swift — standalone MLX-Swift port of SeedVR2 (ByteDance, ICLR 2026)
// one-step diffusion super-resolution, for MLXEngine / ForgeUpscaler (Export tier).
//
// Reference (oracle): filipstrand/mflux  src/mflux/models/seedvr2/ (MLX-Python).
// This is an MLX-Python -> MLX-Swift translation; isomorphic module structure.
//
import PackageDescription

let package = Package(
    name: "SeedVR2MLX",
    platforms: [.macOS(.v14), .iOS(.v17), .visionOS(.v1)],
    products: [
        .library(name: "SeedVR2MLX", targets: ["SeedVR2MLX"]),
        .executable(name: "seedvr2-upscale", targets: ["RunUpscale"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.2" ..< "0.32.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "SeedVR2MLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/SeedVR2MLX"
        ),
        .executableTarget(
            name: "RunUpscale",
            dependencies: [
                "SeedVR2MLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/RunUpscale"
        ),
        .testTarget(
            name: "SeedVR2MLXTests",
            dependencies: [
                "SeedVR2MLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/SeedVR2MLXTests"
        ),
    ]
)
