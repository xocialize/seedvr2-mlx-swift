// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
import Foundation
import MLX

/// Loads SeedVR2 weights exported by `seedvr2-mlx/scripts/prepare_swift.py`
/// (transformer.safetensors + vae.safetensors + pos_emb.safetensors + config.json).
/// Local directory now; HF-Hub download (mlx-community/SeedVR2-3B-mlx) to follow.
public struct SeedVR2Weights {
    public let config: SeedVR2Config
    public let transformer: [String: MLXArray]
    public let vae: [String: MLXArray]
    public let posEmb: MLXArray

    public init(directory url: URL) throws {
        let cfgData = try Data(contentsOf: url.appendingPathComponent("config.json"))
        let raw = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any] ?? [:]
        var cfg = SeedVR2Config.r3B
        if let variant = raw["variant"] as? String, variant.contains("7b") { cfg = .r7B }
        if let ov = raw["transformer_overrides"] as? [String: Int] { cfg.apply(overrides: ov) }
        self.config = cfg

        self.transformer = try MLX.loadArrays(url: url.appendingPathComponent("transformer.safetensors"))
        self.vae = try MLX.loadArrays(url: url.appendingPathComponent("vae.safetensors"))
        let pe = try MLX.loadArrays(url: url.appendingPathComponent("pos_emb.safetensors"))
        guard let emb = pe["embedding"] else { throw WeightError.missing("pos_emb.embedding") }
        self.posEmb = emb.ndim == 2 ? emb.expandedDimensions(axis: 0) : emb
    }

    enum WeightError: Error { case missing(String) }
}
