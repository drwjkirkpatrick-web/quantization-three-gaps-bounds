# quantization-three-gaps-bounds

## Theoretical Bounds for the Three Gaps in Learned Image Compression Quantization

A research package deriving tight upper bounds for the three approximation gaps identified by Pan et al. (2021) in learned image compression.

### The Three Gaps

1. **Discrete Gap** — error from replacing hard quantization with additive uniform noise during training
2. **Entropy Estimation Gap** — error from using a learned entropy model instead of the true marginal distribution
3. **Local Smoothness Gap** — error from assuming constant density within each quantization bin

### Key Results

- **Discrete gap:** ≤ d·log₂(1 + Δ/(2σ_eff))
- **Entropy gap:** ≤ √(d·N_model/M) + d·H(Δ)
- **Smoothness gap:** ≤ (d/2)·log₂(1 + (LΔ)²/12)
- **Total gap:** additive composition of all three

### Installation

```bash
git clone https://github.com/drwjkirkpatrick-web/quantization-three-gaps-bounds.git
cd quantization-three-gaps-bounds
pip install -e ".[test]"
```

### Testing

```bash
python -m pytest tests/ -v
```

### License

MIT