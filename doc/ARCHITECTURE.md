# Prime Plot Architecture

## Overview

Prime Plot is a Python tool for visualizing prime number patterns and detecting primes using machine learning. The system supports 2D spiral visualizations (Ulam, Sacks, Klauber) and experimental 3D volumetric approaches.

## Directory Structure

```
prime_plot/
    doc/
        ARCHITECTURE.md     # This file

    src/prime_plot/
        __init__.py
        cli.py              # Unified command-line interface

        core/
            __init__.py
            sieve.py        # Prime sieve implementations (NumPy fallback + optional primesieve)
            polynomials.py  # Prime-generating polynomial analysis

        visualization/
            __init__.py
            ulam.py         # Ulam spiral (square spiral from center)
            sacks.py        # Sacks spiral (Archimedean polar)
            klauber.py      # Klauber triangle
            vogel.py        # Vogel spiral (golden angle)
            fibonacci.py    # Fibonacci spiral variants
            modular.py      # Modular arithmetic visualizations
            renderer.py     # Image export utilities

        analysis/
            __init__.py
            patterns.py     # Diagonal/line pattern detection
            density.py      # Prime density calculations

        evaluation/
            __init__.py
            metrics.py      # Pattern quality metrics (SNR, entropy, line density)
            detectors.py    # Pattern detection (Hough, FFT, autocorrelation)
            baseline.py     # Random baseline generation
            pipeline.py     # Evaluation orchestration and ranking

        ml/
            __init__.py
            models.py       # 2D neural networks (SimpleUNet, PrimeUNet, PrimeClassifier)
            dataset.py      # Training data generation
            train.py        # Training loop utilities

    tests/
        test_sieve.py
        test_ulam.py
        test_models.py

    # Root-level experimental scripts
    models_3d.py            # 3D U-Net architecture
    ulam_3d.py              # Stacked 2D spirals (z = different start values)
    ulam_3d_true.py         # True 3D cubic spiral (shell expansion)
    train_3d.py             # Training for stacked approach
    train_3d_true.py        # Training for true 3D spiral
    train_3d_precision.py   # Precision-focused 21-channel training
    train_improved.py       # Enhanced 2D with focal loss
    train_fast.py           # Quick iteration training
    train_with_features.py  # 13-channel feature engineering
    train_extended_features.py  # 17-channel features
    run_inference.py        # Basic inference
    run_inference_features.py
    run_inference_extended.py
    run_inference_3d.py
    run_inference_3d_true.py
    benchmark_inference.py  # Performance measurement
    optimized_inference.py  # Batch inference
    check_probabilities.py  # Model output analysis
    evolve_linear_genome.py # GA evolution of scale-invariant coordinates
    test_linear_coordinates.py # Linear coordinate testing
```

## Component Details

### Core Module

#### sieve.py
Prime number generation using the Sieve of Eratosthenes.

- `compute_sieve(limit)` - Returns boolean array where index i is True if i is prime
- NumPy-based implementation as fallback
- Optional primesieve library for high-performance generation (10-100x faster)
- GPU acceleration via CuPy when available

#### polynomials.py
Analysis of prime-generating polynomials like Euler's n^2 + n + 41.

- `PrimePolynomial` class for evaluating and analyzing polynomials
- `FAMOUS_POLYNOMIALS` dictionary of known high-density polynomials
- `find_dense_polynomials()` searches parameter space for new candidates
- Density calculation over specified ranges

### Visualization Module

#### ulam.py
Ulam spiral implementation where integers spiral outward from center.

- `UlamSpiral(size, start=1)` - Creates spiral grid
- `generate_grid()` - Returns 2D array of integer values at each position
- `render_primes(use_gpu=False)` - Returns binary image (1=prime, 0=composite)
- `integer_to_coords(n, size)` - Static method for coordinate calculation
- Direction: clockwise (right, down, left, up) matching image coordinates

#### sacks.py
Sacks spiral using polar coordinates (Archimedean spiral).

- `SacksSpiral(max_n, image_size)` - Creates polar spiral
- Position: r = sqrt(n), theta = 2*pi*sqrt(n)
- Better for visualizing gaps between primes

#### klauber.py
Klauber triangle arrangement.

- `KlauberTriangle(rows)` - Creates triangular grid
- Numbers arranged in rows, primes marked
- Reveals vertical line patterns

#### vogel.py
Vogel spiral using the golden angle (~137.5 degrees).

- `VogelSpiral(max_n, image_size, scaling)` - Golden angle spiral
- Points placed at angle = n * golden_angle, radius = sqrt(n)
- Creates sunflower-seed-like patterns
- Primes align along specific rays
- Scaling modes: sqrt (default), linear, log

#### fibonacci.py
Fibonacci-based spiral visualizations.

- `FibonacciSpiral(max_n, image_size)` - Shells based on Fibonacci intervals
  - Radius grows by golden ratio per shell
  - Angle uses golden angle within shell
- `ReverseFibonacciSpiral(max_n, image_size)` - Converges toward center
  - Inverted radius relationship
  - Explores conjugate golden ratio behavior
- `FibonacciShellPlot(max_n, image_size)` - Concentric shell visualization
  - Radial = shell number, angular = position within shell
  - `shell_statistics()` - Prime counts between Fibonacci numbers

#### modular.py
Modular arithmetic visualizations.

- `ModularGrid(max_n, mod1, mod2)` - 2D grid at (n mod m1, n mod m2)
  - Reveals residue class patterns
  - Common: mod 6 shows primes at (1,1), (1,5), (5,1), (5,5)
- `ModularClock(max_n, modulus, image_size)` - Circular clock face
  - Angle = 2*pi*(n mod m)/m, radius = log(n)
  - Creates radial spoke patterns
- `ModularMatrix(max_n, max_modulus)` - Matrix of residue class densities
  - Row = modulus, column = residue
  - `find_forbidden_residues()` - Find empty coprime residue classes
- `CageMatch(max_n, modulus, image_size)` - Fibonacci modular arithmetic
  - x = F_n mod m, y = n mod m
  - Reveals Pisano period patterns
  - `pisano_period()` - Computes period of Fibonacci mod m

#### renderer.py
Common image export utilities.

- `save_raw_image(array, path)` - Saves NumPy array as image
- Supports PNG output via Pillow

### Analysis Module

#### patterns.py
Pattern detection in prime visualizations.

- `detect_diagonal_patterns(grid, min_length, min_density)` - Finds high-density diagonals
- `extract_high_density_regions(image, window_size, min_density)` - Sliding window analysis
- Returns pattern objects with density, length, position info

#### density.py
Prime density calculations.

- Local density in windows
- Comparison to expected density (1/ln(n))

### ML Module

#### models.py
Neural network architectures for prime detection.

**SimpleUNet**
- Lightweight U-Net with configurable feature depths
- Default: [64, 128, 256, 512] encoder channels
- Bilinear upsampling in decoder
- Input: configurable channels, Output: 1 channel (prime probability)

**PrimeUNet**
- U-Net with pretrained ResNet encoder (18/34/50)
- ImageNet weights for transfer learning
- Better for limited training data

**PrimeClassifier**
- Simple CNN for patch classification
- Classifies regions as high/low prime density

#### dataset.py
Training data generation.

- `create_dataloader(block_size, num_blocks, batch_size, seed, augment)` - Creates PyTorch DataLoader
- Generates random spiral blocks on-the-fly
- Augmentation: random flips

#### train.py
Training utilities.

- `train_model(model, train_loader, val_loader, epochs, learning_rate, checkpoint_dir)` - Training loop
- Supports checkpointing
- Returns training history

### Evaluation Module

Framework for comparing visualization methods by pattern quality.

#### metrics.py
Pattern quality metrics.

- `PatternMetrics` - Container for all quality scores
- `calculate_line_density(image)` - Measures linear pattern alignment
- `calculate_diagonal_density(image)` - Measures diagonal patterns (Ulam-relevant)
- `calculate_cluster_coherence(image)` - Measures cluster definition quality
- `calculate_entropy(image)` - Shannon entropy (lower = more structured)
- `calculate_snr(signal, noise)` - Signal-to-noise ratio vs baseline
- `compute_all_metrics(image, name, baseline)` - Compute all metrics at once

#### detectors.py
Pattern detection algorithms.

- `detect_lines(image)` - Hough transform-based line detection
- `detect_clusters(image)` - Connected component cluster detection
- `compute_fft_spectrum(image)` - 2D FFT for periodic patterns
- `compute_autocorrelation(image)` - Self-similarity detection
- `detect_radial_patterns(image)` - Radial ray pattern detection

#### baseline.py
Random baseline generation for comparison.

- `generate_random_baseline(shape, density)` - Uniform random points
- `generate_density_matched_baseline(reference)` - Match reference density
- `generate_radial_density_baseline(shape, points)` - Radial distribution
- `generate_local_density_baseline(reference)` - Match local densities
- `compute_baseline_statistics(image)` - Statistical comparison with baselines

#### pipeline.py
Main evaluation orchestration.

- `EvaluationPipeline` - Orchestrates full evaluation
  - `run_evaluation(methods)` - Evaluate all or specified methods
  - `get_ranking()` - Get methods ranked by score
  - `generate_report(output_dir)` - Create comprehensive report
- `VisualizationResult` - Container for single method results
- `run_full_evaluation()` - Convenience function
- `compare_methods_at_scales()` - Compare across number ranges

### CLI Module (cli.py)

Unified command-line interface with subcommands:

```
python -m prime_plot.cli <command> [options]

Commands:
    ulam        Generate Ulam spiral image
    sacks       Generate Sacks spiral image
    klauber     Generate Klauber triangle image
    vogel       Generate Vogel spiral (golden angle)
    fibonacci   Generate Fibonacci-based spiral (forward/reverse/shell)
    modular     Generate modular arithmetic viz (grid/clock/matrix/cage)
    evaluate    Evaluate and compare visualization methods
    train       Train ML model
    analyze     Analyze prime patterns
    polynomial  Analyze/search prime-generating polynomials
```

## 3D Experimental Architecture

### Two Approaches

**Stacked 2D (ulam_3d.py)**
- Stack multiple 2D Ulam spirals along z-axis
- Each z-level starts at different integer
- Not spatially coherent (adjacent z cells not adjacent integers)
- Simpler implementation

**True 3D Cubic Spiral (ulam_3d_true.py)**
- Numbers wind outward from center in 3D
- Shell-by-shell expansion (Chebyshev distance ordering)
- Adjacent cells have adjacent integers
- Uses numba JIT for performance
- `generate_3d_cubic_spiral(size, start)` - Main generator

### 3D U-Net (models_3d.py)

**SimpleUNet3D**
- 3D convolutions (Conv3d, MaxPool3d, ConvTranspose3d)
- Input: (batch, channels, depth, height, width)
- Configurable base features and input channels

### Feature Engineering

Models use multi-channel input instead of raw binary:

**21-Channel Features (train_3d_precision.py)**
1. Position X (normalized -1 to 1)
2. Position Y
3. Position Z
4. Distance from center (Euclidean)
5. Shell number (Chebyshev distance)
6. Log-normalized integer value
7-18. Modular features: mod 2, 3, 5, 6, 7, 11, 13, 17, 19, 23, 29, 30
19. Angle in XY plane
20. Angle in XZ plane
21. Density hint (1/ln(n))

### Loss Functions

**FocalLoss**
- Addresses class imbalance (primes ~6% of integers)
- Parameters: alpha (class weight), gamma (focusing), pos_weight

**F1Loss**
- Differentiable F1 score optimization
- Beta parameter: >1 favors recall, <1 favors precision

**CombinedLoss**
- Weighted combination of focal and F1 losses
- Best results with focal_weight=0.6, f1_weight=0.4

## Data Flow

```
1. Prime Generation
   primesieve/numpy sieve -> boolean array of primes

2. Spiral Construction
   Integer range -> 2D/3D coordinates via spiral algorithm

3. Feature Engineering
   Raw grid -> Multi-channel tensor (position, modular, density features)

4. Model Training
   Features + Labels -> U-Net -> Binary predictions
   Loss: Combined focal + F1

5. Inference
   New grid -> Features -> Trained model -> Prime probability map

6. Analysis
   Probability map -> Threshold -> Pattern detection
```

## Hardware Acceleration

### GPU (CUDA)
- PyTorch CUDA for model training/inference
- CuPy for array operations (optional)
- Automatic fallback to CPU

### NPU (Planned)
- Intel: OpenVINO export
- AMD: ONNX + Vitis AI

## Key Design Decisions

1. **Optional dependencies**: Heavy libraries (primesieve, cupy, numba) are optional with graceful fallbacks

2. **Modular structure**: Core/visualization/analysis/ml separation allows independent development

3. **Feature engineering over architecture**: Multi-channel input features more impactful than deeper networks

4. **Precision focus**: Class imbalance handling critical - lower pos_weight than natural ratio improves precision

5. **Experimental scripts at root**: 3D experiments kept separate from stable src/ code until validated

6. **Scale-invariant coordinates**: Linear/modular coordinates (n mod m) maintain consistent spread at any scale, unlike sqrt(n) which collapses at billion scale

## Linear Genome Evolution

### The Scaling Problem

The original evolved visualization used sqrt(n) coordinates, which causes coordinate collapse at large scales:
- At 1B scale, a 100K range has 83x less coordinate spread than at 100K scale
- Primes and composites cluster together, destroying spatial patterns

### Linear Coordinate Solution

`evolve_linear_genome.py` implements genetic algorithm evolution of scale-invariant parameters:

**Coordinate Formula:**
```
r = r_const + r_scale*n + r_mod*(n mod r_mod_base)
theta = t_const + t_mod*(n mod t_mod_base)*(2*pi/t_mod_base)
x_grid = x_mod*(n mod x_mod_base)
y_grid = y_mod*(n mod y_mod_base)
final = blend*polar + (1-blend)*grid + qr_effect + digit_sum_effect
```

**Key Discovery - Mod-3 Dominance:**

The GA discovered that t_mod_base=3 and y_mod_base=3 work best because:
- All primes > 3 are either 1 or 2 mod 3
- This creates a fundamental binary split highly informative for classification
- Simpler than the originally assumed mod-19 or mod-30 patterns

**Evolved Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| r_mod_base | 109 | Large prime for radial variety |
| t_mod_base | 3 | Fundamental prime structure |
| y_mod_base | 3 | Same fundamental structure |
| qr_base | 47 | Prime quadratic residue base |
| digit_sum_effect | 0.471 | Exploits digit sum mod 9 |

**Performance:**
- Billion-scale correlation: 0.2321 (vs ~0 with sqrt coordinates)
- Prime search efficiency: 2.22x at 1B scale, 2.61x at 10M scale
