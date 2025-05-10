# Label Noise Correction with Siamese Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Instance-Dependent Label Noise Correction via Contrastive Learning and Ensemble Disagreement"** (NeurIPS 2025). This framework detects and corrects label noise in datasets using Siamese networks with nested cross-validation and consensus-based relabeling.


## 🔑 Key Features
- **Siamese Network Architecture**: Twin-branch model for contrastive learning and classification.
- **Nested Cross-Validation**: Prevents data leakage with stratified outer/inner splits.
- **Ensemble Disagreement**: Identifies noisy samples using model consensus.
- **Relabeling Score**: Quantifies correction quality without ground-truth labels.
- **Multiple Dataset Support**: CIFAR-10, Fashion-MNIST, and CIFAR-10N with configurable noise levels.


## 🚀 Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.6+

```bash
pip install -r requirements.txt
```


## 📂 Getting Started

### Dataset Preparation
Datasets will be automatically downloaded through `torchvision` when you run the framework. Supported datasets:
- CIFAR-10 (synthetic noise levels: 20%, 30%, 40%)
- Fashion-MNIST (synthetic noise levels: 20%, 30%, 40%)
- CIFAR-10N (real-world noise)

---

## 🛠️ Configuration
The framework uses a comprehensive configuration system in `models/config.py` with pre-defined parameter dictionaries for each dataset and noise level:

- **CIFAR-10**: `CIFAR10_20_PARAMS`, `CIFAR10_30_PARAMS`, `CIFAR10_40_PARAMS`
- **Fashion-MNIST**: `FashionMNIST_20_PARAMS`, `FashionMNIST_30_PARAMS`, `FashionMNIST_40_PARAMS`
- **CIFAR-10N**: `CIFAR10N_PARAMS`

Each configuration includes parameters for:
- Network architecture (ResNet34/ResNet50)
- Training settings (batch size, learning rate, etc.)
- Embedding dimensions
- Thresholds for noise detection and relabeling
- Path configurations for saving models and results

---

## 🏋️ Training
The simplest way to run the framework is through the command-line interface:

```bash
# Train on CIFAR-10 with 30% synthetic noise
python runner.py --dataset cifar10 --noise_ratio 30 --output_dir results/cifar10_30

# Train on Fashion-MNIST with 40% synthetic noise
python runner.py --dataset fashionmnist --noise_ratio 40 --output_dir results/fmnist_40

# Train on CIFAR-10N with real-world noise
python runner.py --dataset cifar10n --noise_ratio n --output_dir results/cifar10n
```

The `runner.py` script:
1. Selects the appropriate dataset configuration based on arguments
2. Initializes the NoiseCleaner with the proper parameters
3. Runs the noise detection and correction pipeline
4. Saves the cleaned dataset to the specified output directory

For custom configurations, you can modify the parameter dictionaries in `models/config.py`.

Predictions are included so model can output clean dataset with desire thresholds without training again
<!-- 
---

## 🔍 Evaluation
After training, you can evaluate the noise detection performance:

```bash
python evaluate.py --dataset cifar10 --noise_ratio 30 --checkpoint results/cifar10_30/best_model.pth
```

Generate visualizations:
```bash
python visualize.py --dataset cifar10 --noise_ratio 30 --input_dir results/cifar10_30 --output_dir figures/
``` -->

---

## 📂 Repository Structure
```
.
├── models/              # Core implementation
│   ├── cleaner.py       # Noise detection and correction pipeline
│   ├── config.py        # Configuration parameters for all datasets
│   ├── siamese.py       # Siamese network architecture
│   ├── utils.py         # Utility functions and constants
│   └── final_model_tester.py  # Final model evaluation
├── data/                # Directory where datasets are downloaded
├── runner.py            # Command-line interface for running experiments
├── main.ipynb           # Aggregated results can be found here
└── requirements.txt
```

---
<!-- 
## 📜 Citation
If you use this work, please cite:
```bibtex
@inproceedings{aref2025noisecorrection,
  title={Instance-Dependent Label Noise Correction via Contrastive Learning and Ensemble Disagreement},
  author={Your Name and Collaborators},
  booktitle={NeurIPS},
  year={2023}
}
```

--- -->

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.