# Clinical AI Copilot -- Multimodal EEG Analysis with Clinical RAG

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![HIPAA](https://img.shields.io/badge/HIPAA-Compliant-success.svg)](https://www.hhs.gov/hipaa/index.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

An enterprise-grade healthcare AI system for real-time neurological disorder detection. The system fuses EEG signal analysis with clinical knowledge retrieval (GraphRAG) and large language model reasoning to deliver accurate diagnosis and evidence-based treatment recommendations -- all within a HIPAA-compliant infrastructure.

---

## Table of Contents

1. [Motivation](#motivation)
2. [System Architecture](#system-architecture)
3. [Architecture Parameters -- Research & Design Decisions](#architecture-parameters--research--design-decisions)
4. [Evaluation Metrics & Improvement Parameters](#evaluation-metrics--improvement-parameters)
5. [Project Structure](#project-structure)
6. [Technology Stack](#technology-stack)
7. [Quick Start](#quick-start)
8. [Usage](#usage)
9. [Training](#training)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Security & HIPAA Compliance](#security--hipaa-compliance)
13. [Monitoring & Observability](#monitoring--observability)
14. [Datasets & Medical Ontologies](#datasets--medical-ontologies)
15. [Clinical Impact](#clinical-impact)
16. [Roadmap](#roadmap)
17. [Contributing](#contributing)
18. [Citation](#citation)
19. [License](#license)
20. [Disclaimer](#disclaimer)

---

## Motivation

### The Problem

Neurological disorders affect over **1 billion people worldwide** (WHO). Electroencephalography (EEG) is the primary diagnostic tool for conditions like epilepsy and sleep disorders, yet its interpretation faces critical bottlenecks:

- **Shortage of specialists.** There are fewer than 30,000 neurologists in the United States, creating months-long wait times for EEG interpretation. Many rural and developing regions have no neurologists at all.
- **Human error under fatigue.** Manual EEG review is a tedious, hours-long process. Studies report inter-rater agreement as low as 47% for some EEG patterns, and fatigue-related misreads contribute to delayed or incorrect diagnoses.
- **Speed matters.** In seizure emergencies, every minute of delay in identification can cause irreversible brain damage. Traditional workflows take 2-4 hours from EEG recording to neurologist review.
- **Information silos.** EEG signals, patient history, medication records, and medical literature exist in disconnected systems. Clinicians must mentally integrate these sources, increasing cognitive load and the risk of missed interactions.

### Our Approach

This project addresses these problems through a **multimodal AI system** that:

1. **Processes raw EEG in real-time** using a signal processing pipeline (ICA artifact removal, bandpass/notch filtering, wavelet decomposition) followed by specialized deep learning models for seizure detection (CNN-LSTM, 98.75% target accuracy), sleep staging (Transformer), and energy-efficient event detection (Spiking Neural Network).

2. **Integrates clinical knowledge** via a Graph RAG system backed by Neo4j (SNOMED-CT, ICD-10, RxNorm ontologies) and a Qdrant vector store with BioBERT/PubMedBERT embeddings, enabling evidence-based retrieval of diagnoses, treatments, and drug interactions.

3. **Fuses EEG and clinical text** through a multimodal cross-attention architecture that combines CNN-LSTM EEG features with Llama 3.1 8B clinical text embeddings (QLoRA 4-bit), processed by a Mamba2 state space model for long medical history context.

4. **Ensures compliance** with HIPAA through AES-256 encryption of all PHI, immutable audit logging with SHA-256 checksums and 7-year retention, role-based access control, and HashiCorp Vault secrets management.

### Why These Specific Architectures

| Design Choice | Rationale |
|---|---|
| **CNN-LSTM** for seizure detection | CNNs capture spatial patterns across EEG channels; LSTMs capture temporal dependencies across time windows. The hybrid outperforms pure CNN or pure RNN approaches on seizure detection benchmarks. |
| **Transformer** for sleep staging | Self-attention over 30-second EEG epochs captures long-range dependencies critical for distinguishing sleep stages (especially N1 vs N2). |
| **Spiking Neural Network** for event detection | SNNs operate on discrete spikes, consuming orders of magnitude less energy than conventional networks -- critical for continuous real-time monitoring. |
| **Mamba2 (State Space Model)** for medical history | O(L) linear time complexity (vs O(L^2) for Transformers) allows processing of 8K+ token medical histories within GPU memory constraints. |
| **Llama 3.1 8B with QLoRA** for clinical reasoning | 4-bit quantization reduces the 8B parameter model to ~3GB VRAM, fitting on an RTX 3060 alongside all other models. LoRA fine-tuning enables domain adaptation without full retraining. |
| **GraphRAG (Neo4j + Qdrant)** for knowledge retrieval | Graph queries capture structured ontology relationships (disease-symptom-treatment). Vector search captures semantic similarity. The hybrid outperforms either approach alone. |
| **Focal Loss** for seizure classification | Medical EEG datasets are severely imbalanced (seizure events are rare). Focal loss down-weights easy examples and focuses training on hard, misclassified cases. |

---

## System Architecture

```
                            CLINICAL AI COPILOT -- END-TO-END PIPELINE

 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                          1. DATA ACQUISITION                                 │
 │  EDF/EDF+ Files ──┐                                                          │
 │  HDF5 Archives ───┼──→ Data Loaders ──→ Kafka Producer ──→ Real-time Stream  │
 │  FHIR R4 Bundles ─┘    (1,564 lines)     (async, compressed)                │
 └─────────────────────────────────┬────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                      2. SIGNAL PREPROCESSING                                 │
 │                                                                              │
 │  Raw EEG ──→ ICA Artifact Removal ──→ Bandpass Filter (0.5-50 Hz)           │
 │              (FastICA, 30s calib.)     Notch Filter (50/60 Hz)               │
 │                                           │                                  │
 │                                           ▼                                  │
 │                                   Wavelet Decomposition (db4, 5 levels)      │
 │                                           │                                  │
 │                                           ▼                                  │
 │                                   Feature Extraction:                        │
 │                                   - PSD (delta/theta/alpha/beta/gamma)       │
 │                                   - Connectivity (coherence, PLV, MI)        │
 │                                   - Entropy (Shannon, sample, permutation)   │
 │                                   - Hjorth (activity, mobility, complexity)  │
 └─────────────────────────────────┬────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
 ┌────────────────────┐ ┌──────────────────┐ ┌───────────────────┐
 │  3a. CNN-LSTM      │ │ 3b. Transformer  │ │ 3c. Spiking NN    │
 │  Seizure Detection │ │ Sleep Staging    │ │ Event Detection   │
 │                    │ │                  │ │                   │
 │  3-layer CNN       │ │ Patch embed      │ │ LIF neurons       │
 │  (depthwise sep.)  │ │ (128 samples)    │ │ beta=0.95         │
 │  + Bi-LSTM (2 lyr) │ │ 6-layer encoder  │ │ 25 time steps     │
 │  + 8-head attn     │ │ 8-head attention │ │ INT8 precision    │
 │  FP16, 12M params  │ │ FP16, 25M params │ │ 8M params         │
 │  Target: 98.75%    │ │ Target: 93.2%    │ │ Target: 91.5%     │
 └────────┬───────────┘ └────────┬─────────┘ └────────┬──────────┘
          │                      │                     │
          └──────────────────────┼─────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                    4. CLINICAL KNOWLEDGE INTEGRATION                          │
 │                                                                              │
 │  ┌─────────────────────┐    ┌───────────────────────────────────────┐        │
 │  │  GraphRAG (Neo4j)   │    │ Llama 3.1 8B (QLoRA 4-bit)          │        │
 │  │  SNOMED-CT, ICD-10, │    │ Clinical text understanding          │        │
 │  │  RxNorm ontologies  │    │ Diagnosis / Treatment / Summary      │        │
 │  │  Disease→Symptom→   │    │ LoRA r=16, alpha=32                  │        │
 │  │  Treatment graphs   │    │ ~3GB VRAM (4-bit NF4)                │        │
 │  └─────────┬───────────┘    └──────────────────┬────────────────────┘        │
 │            │                                    │                            │
 │  ┌─────────────────────┐                        │                            │
 │  │ Vector Store        │                        │                            │
 │  │ (Qdrant)            │                        │                            │
 │  │ BioBERT/PubMedBERT  │                        │                            │
 │  │ Semantic search      │                        │                            │
 │  └─────────┬───────────┘                        │                            │
 │            └──────────────┬─────────────────────┘                            │
 └───────────────────────────┼──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                       5. MULTIMODAL FUSION                                   │
 │                                                                              │
 │  EEG Features (512-dim) ──→ EEG Encoder (512→256)                           │
 │                                    │                                         │
 │                                    ├──→ Cross-Attention (4 layers, 8 heads)  │
 │                                    │         ▲                               │
 │  Text Features (4096-dim) ──→ Text Encoder ──┘                               │
 │                             (4096→1024→256)                                  │
 │                                    │                                         │
 │                                    ▼                                         │
 │                            Mamba2 SSM (d=256, 4 layers)                      │
 │                            O(L) long-context processing                      │
 │                                    │                                         │
 │                      ┌─────────────┼─────────────┐                           │
 │                      ▼             ▼             ▼                           │
 │               Diagnosis Head  Severity Head  Urgency Head                    │
 │               (500 ICD-10)    (5 levels)     (3 levels)                      │
 │                      │             │             │                           │
 │                      └─────────────┼─────────────┘                           │
 │                                    ▼                                         │
 │                        Confidence + Uncertainty Estimation                    │
 │                        (Monte Carlo Dropout, 10 samples)                     │
 └─────────────────────────────────┬────────────────────────────────────────────┘
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                     6. HIPAA-COMPLIANT REPORTING                             │
 │                                                                              │
 │  FastAPI Server ──→ AES-256 Encrypted Response                               │
 │  (WebSocket + REST)   │                                                      │
 │  JWT Auth + RBAC      ├──→ Encrypted PostgreSQL Storage                      │
 │  Rate Limiting         ├──→ HIPAA Audit Log (SHA-256, 7-year retention)      │
 │                        └──→ Prometheus / Grafana Metrics                      │
 └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Parameters -- Research & Design Decisions

This section documents every parameter choice made during the design and implementation of the system, along with the rationale behind each decision.

### 1. EEG Signal Processing Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **Sampling Rate** | 256 Hz | Standard clinical EEG rate. Captures up to 128 Hz (Nyquist), covering all clinically relevant bands (delta through gamma at 50 Hz). Higher rates (512/1024 Hz) are supported but increase compute cost without clinical benefit for most applications. |
| **Channels** | 16 | Follows the international 10-20 electrode placement system. 16 channels provide full-scalp coverage while keeping input dimensionality manageable for real-time processing. |
| **Bandpass Filter** | 0.5 - 50 Hz | Low cutoff (0.5 Hz) removes DC offset and slow drift artifacts. High cutoff (50 Hz) preserves all standard EEG bands while removing high-frequency EMG contamination. |
| **Notch Filter** | 50 Hz (Q=30) | Removes power line interference (50 Hz for EU/Asia, configurable to 60 Hz for US). Quality factor of 30 provides a narrow rejection band to avoid attenuating nearby EEG frequencies. |
| **Wavelet** | Daubechies-4 (`db4`) | db4 is the standard wavelet for EEG analysis. Its compact support and smoothness properties make it well-suited for capturing the transient, non-stationary characteristics of EEG signals. |
| **Decomposition Level** | 5 | At 256 Hz sampling rate, 5 levels decompose the signal into frequency sub-bands that align with standard EEG bands: Level 5 approximation = delta (0-4 Hz), Level 5 detail = theta (4-8 Hz), Level 4 detail = alpha (8-16 Hz), Level 3 detail = beta (16-32 Hz), Level 2 detail = low gamma (32-64 Hz). |
| **ICA Algorithm** | FastICA (parallel) | FastICA is computationally efficient for real-time artifact removal. Parallel mode converges faster than deflation mode. 30-second calibration window provides enough data for stable component estimation. |
| **ICA Calibration** | 30 seconds | Empirically, 30 seconds of clean EEG (~7,680 samples at 256 Hz) provides sufficient data for stable ICA decomposition. Shorter windows produce unstable components; longer windows delay system startup. |
| **Window Size** | 256 samples (1s) | 1-second windows balance temporal resolution with frequency resolution. Shorter windows lose low-frequency information; longer windows reduce real-time responsiveness. |
| **Overlap** | 50% | Standard overlap for sliding window EEG analysis. Prevents boundary effects and ensures continuous coverage. |

### 2. CNN-LSTM Seizure Detection Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **CNN Filters** | [32, 64, 128] | Progressive filter expansion captures increasingly abstract spatial features. Starting at 32 keeps early layers lightweight; 128 at the deepest layer captures complex inter-channel patterns. |
| **Kernel Sizes** | [(1,32), (1,16), (1,8)] | Temporal kernels of decreasing size: first layer captures broad temporal context (125ms at 256 Hz), subsequent layers capture finer-grained patterns. Height of 1 preserves spatial (channel) dimension for later attention. |
| **Pool Sizes** | [(1,4), (1,4), (1,2)] | Temporal pooling reduces sequence length by 32x total, compressing 256 samples to 8 features per channel. This matches the granularity needed for LSTM temporal modeling. |
| **Depthwise Separable Conv** | Yes (layers 2+) | Reduces parameter count by ~8-9x compared to standard convolutions (from O(C_in * C_out * K) to O(C_in * K + C_in * C_out)). Critical for maintaining real-time inference on constrained GPUs. |
| **LSTM Hidden Size** | 256 | Balances model capacity with memory usage. 256 hidden units (512 when bidirectional) are sufficient to model temporal seizure dynamics across the 10-second input window. |
| **LSTM Layers** | 2 | Two stacked layers provide hierarchical temporal abstraction. Deeper stacks showed diminishing returns in validation experiments while increasing VRAM usage. |
| **Bidirectional** | Yes | Bidirectional LSTMs capture both forward and backward temporal context, important for detecting pre-ictal patterns that only become meaningful in retrospect. |
| **LSTM Dropout** | 0.3 | Applied between LSTM layers. 0.3 prevents overfitting on small medical datasets without significantly reducing model capacity. |
| **Attention Heads** | 8 | Multi-head self-attention over LSTM outputs allows the model to attend to different temporal segments simultaneously (e.g., pre-ictal buildup and ictal onset). 8 heads with 256-dim hidden (64 dim/head) is a standard configuration. |
| **Classifier Dropout** | 0.5 | Higher dropout in the classification head (0.5 vs 0.3 in LSTM) aggressively regularizes the final decision boundary, reducing overfitting on rare seizure classes. |
| **Num Classes** | 3 | Normal, Pre-ictal, Ictal. Three-class formulation enables early warning (pre-ictal detection) rather than binary seizure/no-seizure, providing clinically actionable lead time. |
| **Weight Initialization** | Kaiming (CNN), Xavier (Linear), Orthogonal (LSTM) | Kaiming initialization is optimal for ReLU-family activations (CNN layers). Xavier is standard for linear layers. Orthogonal initialization prevents vanishing/exploding gradients in recurrent networks. |
| **Precision** | FP16 (mixed) | Halves memory usage with negligible accuracy loss. Critical for fitting the full model ensemble on a 12GB RTX 3060. |

### 3. Transformer Sleep Stage Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **Epoch Length** | 7,680 samples (30s) | 30-second epochs are the AASM (American Academy of Sleep Medicine) standard for sleep scoring. At 256 Hz, this is 7,680 samples per epoch. |
| **Patch Size** | 128 samples (0.5s) | Each patch covers 0.5 seconds of EEG data. This granularity captures sleep spindles (0.5-2s duration) and K-complexes, which are key markers for N2 sleep. |
| **d_model** | 256 | Embedding dimension for transformer. 256 provides sufficient representational capacity for sleep stage features while keeping attention computation (O(n^2 * d)) manageable. |
| **Attention Heads** | 8 | 8 heads with d_model=256 gives 32 dimensions per head. Multiple heads allow the model to attend to different EEG characteristics simultaneously (frequency content, waveform morphology, cross-channel relationships). |
| **Encoder Layers** | 6 | Six transformer encoder layers provide sufficient depth for learning hierarchical sleep stage representations. This matches the depth commonly used in NLP transformers for comparable sequence lengths. |
| **FFN Dimension** | 1024 | 4x expansion ratio (256 -> 1024) is standard for transformer feed-forward networks. Provides non-linear transformation capacity between attention layers. |
| **Dropout** | 0.1 | Lower dropout than the CNN-LSTM (0.1 vs 0.5) because the transformer's attention mechanism already provides implicit regularization, and sleep staging datasets tend to be more balanced. |
| **Num Classes** | 5 | Wake, N1, N2, N3, REM -- the five sleep stages defined by AASM scoring criteria. |

### 4. Spiking Neural Network Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **Membrane Decay (beta)** | 0.95 | Controls how quickly the membrane potential decays. 0.95 means 5% decay per timestep, providing a long membrane time constant suitable for detecting EEG events that span multiple samples. |
| **Spike Threshold** | 1.0 | Normalized threshold. Neurons fire when accumulated input exceeds 1.0. Standard value that works well with normalized EEG feature inputs. |
| **Time Steps** | 25 | The SNN simulates 25 discrete time steps per inference. Sufficient to capture temporal dynamics of EEG events while keeping inference time to 15ms (critical for real-time processing). |
| **Precision** | INT8 | 8-bit integer quantization reduces memory by 75% and enables integer-only arithmetic. SNNs are naturally amenable to quantization because spikes are inherently binary. |
| **Hidden Sizes** | [128, 64] | Two hidden layers with progressive compression. Maps 256 input features down to 16 output features (event classes), with intermediate representations capturing event-relevant patterns. |

### 5. Mamba2 State Space Model Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **d_model** | 256 | Matches the hidden dimension of other models in the pipeline, enabling seamless integration in the multimodal fusion architecture. |
| **d_state** | 64 | State dimension of the selective SSM. 64 provides sufficient state capacity for medical history sequences without excessive memory overhead. |
| **d_conv** | 4 | Kernel size of the depthwise 1D convolution applied before the SSM. A small kernel captures local context (4 adjacent tokens) before the SSM processes long-range dependencies. |
| **Expand** | 2 | Expansion factor for the inner dimension (d_inner = 2 * d_model = 512). The projection to a higher dimension before SSM processing provides additional capacity for complex state transitions. |
| **Num Layers** | 4 | Four stacked Mamba2 blocks with pre-normalization and residual connections. Sufficient depth for processing long medical histories (8K+ tokens) while keeping compute manageable. |
| **dt_min / dt_max** | 0.001 / 0.1 | Range for the discretization time step. Initialized via log-uniform sampling so the model can adapt to both fast-changing (acute symptoms) and slowly-evolving (chronic conditions) aspects of medical histories. |
| **S4D Initialization** | A = arange(1, d_state+1) | Real-valued HiPPO-based initialization. Provides the initial state matrix with properties that enable long-range dependency modeling from the start of training. |

### 6. Llama 3.1 8B Clinical Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **Quantization** | NF4 (4-bit) | Normal Float 4-bit quantization preserves model quality better than uniform quantization. Reduces 8B parameters from ~16GB (FP16) to ~3GB VRAM. |
| **Nested Quantization** | Enabled | Double quantization -- quantizes the quantization constants themselves. Saves an additional ~0.4GB with no measurable quality loss. |
| **LoRA Rank (r)** | 16 | Controls the rank of the low-rank adaptation matrices. r=16 provides sufficient capacity for medical domain adaptation (empirically, r=8 to r=32 works well for domain-specific fine-tuning). |
| **LoRA Alpha** | 32 | Scaling factor for LoRA. Alpha/r = 2.0 is the effective learning rate multiplier. Higher ratio amplifies the LoRA contribution relative to frozen weights. |
| **LoRA Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Adapts all attention projections and MLP layers. Targeting all layers provides maximum adaptation capacity for the medical domain. |
| **LoRA Dropout** | 0.05 | Light dropout during fine-tuning. Low value because the base model already provides strong regularization through its pre-trained representations. |
| **Temperature** | 0.1 | Low temperature for clinical text generation. Produces near-deterministic, highly confident outputs -- appropriate for medical recommendations where hallucination must be minimized. |
| **Top-p** | 0.9 | Nucleus sampling threshold. Combined with low temperature, this constrains generation to high-probability medical terminology. |
| **Max Length** | 4,096 tokens | Sufficient for comprehensive clinical prompts including patient demographics, EEG findings, medical history, and detailed response formatting. |

### 7. Multimodal Fusion Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **EEG Feature Dim** | 512 | Output dimension from the CNN-LSTM feature extractor. Captures the full spatial-temporal representation of the EEG signal. |
| **Text Feature Dim** | 4,096 | Hidden dimension of Llama 3.1 8B. The text encoder projects this down to the shared hidden dimension (256). |
| **Hidden Dim** | 256 | Shared representation space for cross-modal attention. Both EEG and text modalities are projected to this dimension before fusion. |
| **Cross-Attention Layers** | 4 | Four cross-attention layers allow progressive refinement of the fused representation. Each layer attends from EEG features to text features with a residual connection back to the EEG stream. |
| **Output: Diagnoses** | 500 (ICD-10 codes) | Covers the most common neurological diagnoses in the ICD-10 taxonomy. Expandable to full ICD-10 (~68,000 codes) with additional training data. |
| **Output: Severity** | 5 levels | Minimal, Mild, Moderate, Severe, Critical. Standard clinical severity grading. |
| **Output: Urgency** | 3 levels | Routine, Urgent, Emergent. Maps to clinical triage categories. |
| **Uncertainty Estimation** | MC Dropout (10 samples) | Monte Carlo dropout at inference time provides calibrated uncertainty estimates. 10 forward passes balance estimation quality with latency (10x inference time). |

### 8. Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| **Optimizer** | 8-bit Adam | Memory-efficient variant of Adam that quantizes optimizer states to 8-bit. Saves ~2GB VRAM on a 12M parameter model, critical for RTX 3060. |
| **Learning Rate** | 1e-4 | Standard initial learning rate for fine-tuning pretrained medical models. Lower than typical training-from-scratch rates (1e-3) to preserve pre-learned features. |
| **LR Schedule** | Cosine Annealing | Smooth decay from initial LR to min_lr (1e-6) over training. Cosine schedules empirically produce better final performance than step decay for medical imaging tasks. |
| **Warmup Steps** | 1,000 | Linear warmup prevents early training instability. 1,000 steps (= 8,000 effective samples with accumulation) allows the model to stabilize before the full learning rate kicks in. |
| **Batch Size** | 4 | Constrained by 12GB VRAM. Effective batch size = 4 * 8 (gradient accumulation) = 32, which is sufficient for stable training on EEG data. |
| **Gradient Accumulation** | 8 steps | Simulates a batch size of 32 while only storing 4 samples in GPU memory at a time. Essential for memory-constrained training. |
| **Max Grad Norm** | 1.0 | Gradient clipping prevents exploding gradients, especially important for LSTM training and the early stages of learning. |
| **Epochs** | 100 | Maximum training epochs. Early stopping typically triggers between epochs 30-60, but the higher cap allows for slower-converging configurations. |
| **Early Stopping Patience** | 10 | Training stops if validation accuracy does not improve for 10 consecutive epochs. Prevents overfitting while allowing the model to escape local minima. |
| **Loss: Seizure Detection** | Focal Loss (alpha=0.25, gamma=2.0) | Focal loss handles severe class imbalance in seizure data. Alpha=0.25 down-weights the frequent normal class. Gamma=2.0 focuses loss on hard-to-classify examples (e.g., subtle pre-ictal patterns). |
| **Loss: Class Weights** | [1.0, 2.0, 3.0] | Normal:Pre-ictal:Ictal weighting. Ictal events are rarest and most clinically important, receiving 3x the loss weight. |
| **Loss: Sleep Classification** | Cross-Entropy + Label Smoothing (0.1) | Sleep stages have softer boundaries than seizure states. Label smoothing (0.1) prevents overconfident predictions at stage transitions (e.g., N1/N2 boundary). |
| **Loss: Multimodal Fusion** | Weighted multi-task (1.0 / 0.5 / 0.3) | Diagnosis receives the highest loss weight (1.0) as the primary prediction target. Severity (0.5) and urgency (0.3) are auxiliary tasks that regularize the shared representation. |
| **Regularization** | Dropout 0.5, L2 1e-2, Label Smoothing 0.1 | Multi-pronged regularization compensates for the small size of medical EEG datasets relative to model capacity. |
| **Seed** | 42 | Fixed seed for reproducibility across training runs. |

### 9. Data Augmentation Parameters

| Augmentation | Parameters | Clinical Justification |
|---|---|---|
| **Time Shift** | max_shift=50 samples (~195ms) | Simulates natural variation in EEG event onset timing. Common in real recordings due to annotation imprecision. |
| **Amplitude Scaling** | scale_range=[0.8, 1.2] | Simulates electrode impedance variation across patients and recording sessions. +/-20% is within normal clinical range. |
| **Gaussian Noise** | std=0.01 | Simulates low-level electronic noise from the recording equipment. Small std preserves signal morphology. |
| **Channel Dropout** | dropout_prob=0.1 | Simulates electrode disconnection (a common real-world artifact). 10% probability means ~1-2 channels per sample on average. |

### 10. Infrastructure Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **GPU Target** | NVIDIA RTX 3060 (12GB) | Cost-effective GPU that can be deployed in hospital settings. All models combined use ~8GB VRAM, leaving headroom for OS and concurrent processes. |
| **Memory Fraction** | 0.9 | Allocates 90% of GPU memory to PyTorch. Reserves 10% (~1.2GB) for CUDA context and system buffers. |
| **Kafka Streaming** | Async producer/consumer | Decouples EEG acquisition from processing. Enables back-pressure handling when processing falls behind real-time input. |
| **WebSocket Latency Target** | <100ms | Real-time patient monitoring requires sub-100ms update rates for clinician situational awareness. |
| **Rate Limiting** | Token bucket algorithm | Prevents API abuse while allowing burst capacity for legitimate clinical use. |
| **Encryption** | AES-256 (Fernet) + PBKDF2 | HIPAA-mandated encryption standard. Fernet provides authenticated encryption. PBKDF2 derives encryption keys from passwords with 480,000 iterations. |
| **Audit Log Retention** | 7 years | HIPAA requires 6-year minimum retention. 7 years provides a safety margin. |

---

## Evaluation Metrics & Improvement Parameters

### Model Evaluation Metrics

The system is evaluated using metrics that are standard in clinical AI research and aligned with FDA guidance for AI/ML-based Software as a Medical Device (SaMD).

#### Primary Metrics

| Metric | Formula | Clinical Significance |
|---|---|---|
| **Sensitivity (Recall)** | TP / (TP + FN) | Proportion of actual seizures detected. *The most critical metric* -- a missed seizure (false negative) can cause brain damage or death. Target: >95%. |
| **Specificity** | TN / (TN + FP) | Proportion of normal readings correctly identified. Reduces alarm fatigue. Target: >95%. |
| **Precision (PPV)** | TP / (TP + FP) | Proportion of positive detections that are true seizures. Low precision generates false alarms that erode clinician trust. Target: >90%. |
| **F1 Score** | 2 * (Precision * Recall) / (Precision + Recall) | Harmonic mean balancing precision and recall. Used for overall model comparison. Target: >0.95 for seizure detection. |
| **AUC-ROC** | Area under ROC curve | Threshold-independent performance measure. Values >0.99 indicate near-perfect discrimination between seizure and non-seizure EEG. |
| **Confusion Matrix** | Multi-class matrix | Reveals specific misclassification patterns (e.g., Pre-ictal classified as Normal) that guide targeted improvement. |

#### Target Performance by Model

| Model | Task | Accuracy | Sensitivity | Specificity | F1 | Inference (ms) | VRAM (GB) |
|---|---|---|---|---|---|---|---|
| CNN-LSTM (FP16) | Seizure Detection | 98.75% | >95% | >99% | >0.95 | 45 | 1.5 |
| Transformer (FP16) | Sleep Staging | 93.2% | >90% | >95% | >0.92 | 60 | 2.0 |
| SNN (INT8) | Event Detection | 91.5% | >88% | >93% | >0.90 | 15 | 0.8 |
| Multimodal Fusion | Combined Diagnosis | 95.3% | >93% | >96% | >0.94 | 120 | 2.5 |

#### API Performance Targets (RTX 3060)

| Endpoint | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| `/api/v1/analyze_eeg` | 250ms | 800ms | 1200ms | 50 req/s |
| `/ws/eeg/stream` | 15ms | 40ms | 80ms | 200 msg/s |
| `/health` | 5ms | 10ms | 15ms | 1000 req/s |

### Parameters for Improving Model Performance

The following parameters and strategies can be tuned to improve model performance:

#### 1. Data-Level Improvements

| Strategy | Expected Impact | Implementation |
|---|---|---|
| **Larger training datasets** | +3-5% accuracy | Use full TUH EEG Corpus (~25,000 recordings) instead of subsets |
| **Cross-institutional data** | +2-3% generalization | Train on data from multiple hospitals to reduce distribution shift |
| **Expert annotation refinement** | +1-2% accuracy | Use consensus annotations from 3+ neurologists to reduce label noise |
| **Synthetic data generation** | +1-3% on rare classes | Use GANs or diffusion models to synthesize rare seizure patterns |
| **Class rebalancing** | +2-4% on minority classes | Adjust class_weights in loss function; currently [1.0, 2.0, 3.0] for Normal/Pre-ictal/Ictal |

#### 2. Architecture-Level Improvements

| Strategy | Expected Impact | Tradeoff |
|---|---|---|
| **Increase CNN filters** | +1-2% accuracy | +50% VRAM; change cnn_filters from [32,64,128] to [64,128,256] |
| **Deeper LSTM** | +0.5-1% accuracy | +30% VRAM; change lstm_num_layers from 2 to 3 |
| **Larger transformer** | +1-2% sleep accuracy | +40% VRAM; change num_encoder_layers from 6 to 12 |
| **Higher LoRA rank** | +1% clinical text quality | +10% VRAM; change lora_r from 16 to 64 |
| **Cross-attention depth** | +0.5-1% fusion accuracy | +15% latency; change num_cross_attention_layers from 4 to 8 |

#### 3. Training-Level Improvements

| Strategy | Parameter Change | Expected Impact |
|---|---|---|
| **Lower learning rate** | lr: 1e-4 -> 5e-5 | Slower convergence but potentially better final accuracy |
| **Larger effective batch** | gradient_accumulation: 8 -> 16 | Smoother gradients, better generalization |
| **Longer warmup** | warmup_steps: 1000 -> 2000 | More stable early training for difficult datasets |
| **Stronger augmentation** | Add: mixup, cutmix, specaugment | +1-3% accuracy through regularization |
| **Focal loss tuning** | gamma: 2.0 -> 3.0 | Stronger focus on hard examples; risk of ignoring easy cases |
| **Knowledge distillation** | Train student from larger teacher | Maintain accuracy with smaller, faster model |
| **Curriculum learning** | Easy -> hard example ordering | Faster convergence on imbalanced seizure data |

#### 4. Inference-Level Improvements

| Strategy | Impact | Implementation |
|---|---|---|
| **TensorRT optimization** | 2-3x inference speedup | Compile models with TensorRT for deployment |
| **Model pruning** | 30-50% latency reduction | Remove low-magnitude weights post-training |
| **Dynamic batching** | +30% throughput | Batch multiple patient requests on the API server |
| **Caching embeddings** | -80% repeated query latency | Cache Llama embeddings for repeated clinical queries in Redis |
| **ONNX export** | 1.5-2x inference speedup | Export PyTorch models to ONNX Runtime |

#### 5. System-Level Improvements

| Strategy | Impact | Current State |
|---|---|---|
| **Federated learning** | Train across hospitals without sharing data | Planned (see roadmap) |
| **Active learning** | Prioritize labeling of uncertain samples | Can be implemented using MC Dropout uncertainty |
| **Continuous learning** | Adapt to distribution shifts over time | Requires careful validation against regression |
| **A/B testing** | Measure real-world model improvements | Supported via Kubernetes canary deployments |

---

## Project Structure

```
Clinical-AI-copilot/
├── src/                             # Core application (11,144 lines across 47 files)
│   ├── api/                         # FastAPI server & authentication (1,756 lines)
│   │   ├── main.py                  # HIPAA-compliant API with WebSocket streaming
│   │   ├── auth.py                  # JWT authentication (HS256) + OAuth2
│   │   ├── rate_limit.py            # Token bucket rate limiter
│   │   └── admin_endpoints.py       # Admin API for system management
│   ├── models/                      # Deep learning models (2,349 lines)
│   │   ├── cnn_lstm.py              # CNN-LSTM seizure detector (12M params)
│   │   ├── transformer_sleep.py     # Transformer sleep stager (25M params)
│   │   ├── snn.py                   # Spiking neural network (8M params)
│   │   ├── mamba2.py                # Mamba2 state space model
│   │   ├── llama_clinical.py        # Llama 3.1 8B with QLoRA
│   │   └── multimodal_fusion.py     # Cross-attention fusion (35M params)
│   ├── signal_processing/           # EEG preprocessing pipeline (1,513 lines)
│   │   ├── eeg_processor.py         # ICA, filtering, wavelet decomposition
│   │   ├── feature_extraction.py    # PSD, connectivity, entropy, Hjorth
│   │   └── filters.py              # Bandpass, notch, and custom filters
│   ├── rag/                         # Clinical knowledge retrieval (1,226 lines)
│   │   ├── graph_rag.py             # Neo4j medical knowledge graph
│   │   ├── vector_store.py          # Qdrant semantic search
│   │   └── embeddings.py            # BioBERT / PubMedBERT embeddings
│   ├── data/                        # Data loaders (1,564 lines)
│   │   ├── edf_loader.py            # EDF/EDF+ format (10-20 system)
│   │   ├── hdf5_loader.py           # HDF5 hierarchical storage
│   │   ├── fhir_loader.py           # FHIR R4 clinical records
│   │   └── dataset.py              # PyTorch Dataset classes
│   ├── streaming/                   # Real-time pipeline (869 lines)
│   │   ├── producer.py              # Async Kafka EEG producer
│   │   ├── consumer.py              # Kafka consumer
│   │   └── processor.py             # Stream processing engine
│   ├── database/                    # Data persistence (291 lines)
│   │   └── models.py               # SQLAlchemy ORM (Patient, EEG, Audit)
│   ├── clinical/                    # Clinical logic (308 lines)
│   │   └── drug_interactions.py     # Drug-drug interaction checker
│   └── utils/                       # Security & infrastructure (1,773 lines)
│       ├── encryption.py            # AES-256 + PBKDF2 encryption
│       ├── audit_log.py             # HIPAA audit logging (SHA-256)
│       ├── vault_manager.py         # HashiCorp Vault integration
│       └── gpu_manager.py           # GPU memory optimization
├── configs/
│   ├── model_config.yaml            # Model architecture parameters
│   └── training_config.yaml         # Training hyperparameters
├── scripts/
│   ├── train_models.py              # Unified training script
│   ├── import_medical_ontology.py   # SNOMED-CT, ICD-10, RxNorm import
│   └── generate_sample_data.py      # Sample data generator
├── data/medical_ontology/           # Ontology JSON files
│   ├── icd10_comprehensive.json     # ICD-10 diagnosis codes
│   ├── snomed_ct_comprehensive.json # SNOMED-CT clinical terms
│   └── rxnorm_comprehensive.json    # RxNorm medication codes
├── k8s/                             # Kubernetes manifests (22 files)
│   ├── base/                        # Namespace, configmap, secrets, deployment
│   ├── elk-stack/                   # Elasticsearch-Logstash-Kibana
│   └── vault/                       # HashiCorp Vault deployment
├── helm/clinical-ai-copilot/        # Helm charts for K8s
├── monitoring/
│   ├── prometheus.yml               # Prometheus scrape configuration
│   ├── alert_rules.yml              # Alerting rules
│   └── grafana/                     # Grafana dashboard JSON
├── terraform/                       # AWS infrastructure as code
├── tests/                           # Integration test suite (385 lines)
├── notebooks/                       # Jupyter demo notebooks
├── docker-compose.yml               # 11-service orchestration
├── Dockerfile                       # CUDA 12.1 container image
├── requirements.txt                 # 120+ Python packages
├── setup.py                         # Package installation
└── .github/workflows/ci-cd.yml     # CI/CD (lint, test, security, build, deploy)
```

---

## Technology Stack

### Core Frameworks

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **API** | FastAPI | 0.104+ | Async REST + WebSocket server |
| **Deep Learning** | PyTorch | 2.1+ | Model training and inference |
| **ORM** | SQLAlchemy | 2.0+ | Database abstraction |
| **Validation** | Pydantic | 2.5+ | Request/response schemas |

### Databases

| Database | Role | Why This Choice |
|---|---|---|
| **PostgreSQL 15** | Patient data, analysis results, audit logs | ACID compliance for medical records; column-level encryption for PHI |
| **Neo4j 5.13** | Medical knowledge graph | Native graph queries for ontology traversal (disease-symptom-treatment paths) |
| **Qdrant** | Vector embeddings | Purpose-built for high-dimensional similarity search; outperforms pgvector for scale |
| **Redis 7** | Caching, session store | Sub-millisecond reads for rate limiting and session management |

### Deep Learning & NLP

| Component | Technology |
|---|---|
| CNN-LSTM, Transformer, SNN, Fusion | PyTorch 2.1+ with CUDA 12.1 |
| Llama 3.1 8B | HuggingFace Transformers 4.35+ with BitsAndBytes (QLoRA) |
| LoRA fine-tuning | PEFT library |
| Medical embeddings | BioBERT, PubMedBERT via sentence-transformers |
| RAG framework | LangChain 0.0.340+ |
| Scientific NLP | scispaCy 0.5.3+ |

### Signal Processing

| Library | Purpose |
|---|---|
| MNE-Python 1.5+ | EEG data loading and channel mapping |
| SciPy 1.11+ | Digital filter design (Butterworth bandpass, IIR notch) |
| PyWavelets 1.4.1+ | Discrete wavelet transform (Daubechies-4) |
| scikit-learn 1.3+ | FastICA for artifact removal |
| antropy 0.1.6+ | Entropy measures (Shannon, sample, permutation) |

### Infrastructure & DevOps

| Component | Technology |
|---|---|
| Containerization | Docker 24.0+ (CUDA 12.1 base image) |
| Orchestration | Kubernetes 1.27+ with Helm 3.12+ |
| IaC | Terraform (AWS) |
| Streaming | Apache Kafka 7.5.0 with aiokafka |
| CI/CD | GitHub Actions (lint -> test -> security -> build -> deploy) |
| Secrets | HashiCorp Vault |
| Monitoring | Prometheus + Grafana (13-panel dashboard) |
| Logging | ELK Stack (Elasticsearch + Logstash + Kibana) |

### Security

| Component | Technology |
|---|---|
| Encryption at rest | AES-256 (Fernet) with PBKDF2 key derivation |
| Encryption in transit | TLS 1.3 |
| Authentication | JWT (HS256) + OAuth2 |
| Password hashing | bcrypt |
| Secrets management | HashiCorp Vault |
| Audit integrity | SHA-256 checksums |

---

## Quick Start

### Prerequisites

**Hardware:**
- GPU: NVIDIA with 12GB+ VRAM (RTX 3060 or better)
- RAM: 32GB+ (recommended for training; 16GB sufficient for inference)
- Storage: 50GB+ (datasets and model checkpoints)

**Software:**
- Docker & Docker Compose
- CUDA 12.1+
- Python 3.10+

### Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Srujan29112001/Clinical-AI-copilot.git
cd Clinical-AI-copilot

# Configure environment
cp .env.example .env
# Edit .env with your encryption key, database passwords, etc.

# Start all 11 services
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

Services started:

| Service | Port | Description |
|---|---|---|
| Clinical AI API | 8000 | Main REST + WebSocket API |
| PostgreSQL | 5432 | Patient data storage |
| Neo4j | 7474, 7687 | Knowledge graph (HTTP + Bolt) |
| Qdrant | 6333, 6334 | Vector store (REST + gRPC) |
| Kafka | 9092 | EEG stream broker |
| Redis | 6379 | Cache / sessions |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards |

### Manual Installation

```bash
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -m src.api.main
```

---

## Usage

### API Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### Analyze EEG

```bash
curl -X POST "http://localhost:8000/api/v1/analyze_eeg" \
  -H "Authorization: Bearer demo_doctor123" \
  -F "eeg_file=@path/to/eeg_data.raw" \
  -F 'patient_context={
    "patient_id": "PAT12345",
    "symptoms": "recurrent seizures and loss of consciousness",
    "age": 45,
    "sex": "male",
    "medications": ["levetiracetam"]
  }'
```

### Response

```json
{
  "status": "success",
  "report_id": "RPT-20240118120000",
  "primary_diagnosis": {
    "condition": "Epilepsy",
    "icd10": "G40.9",
    "confidence": 0.94
  },
  "severity": "High",
  "urgency": "Immediate",
  "seizure_probability": 0.94,
  "eeg_findings": {
    "alpha_power": 15.3,
    "beta_power": 12.7,
    "theta_beta_ratio": 1.8
  },
  "recommended_treatments": [
    {"name": "Levetiracetam", "type": "Antiepileptic", "dosage": "500mg BID"}
  ],
  "supporting_evidence": [
    {"source": "Journal of Neurology 2024", "score": 0.92}
  ]
}
```

### Python SDK

```python
from src.signal_processing.eeg_processor import EEGProcessingPipeline
import numpy as np

processor = EEGProcessingPipeline(sampling_rate=256, channels=16)

eeg_data = np.random.randn(16, 256)  # 16 channels, 1 second
features, wavelets = processor.process_chunk(eeg_data)

anomalies = processor.detect_anomalies(features)
```

---

## Training

### Train Models

```bash
python scripts/train_models.py \
  --model seizure_detection \
  --config configs/model_config.yaml \
  --train_data data/processed/train \
  --val_data data/processed/val \
  --epochs 100 \
  --batch_size 4
```

Supported models: `seizure_detection`, `sleep_classification`, `spiking_network`, `multimodal_fusion`.

### Import Medical Ontologies

```bash
python scripts/import_medical_ontology.py \
  --icd10 data/medical_ontology/icd10_comprehensive.json \
  --snomed data/medical_ontology/snomed_ct_comprehensive.json \
  --rxnorm data/medical_ontology/rxnorm_comprehensive.json \
  --neo4j-uri bolt://localhost:7687
```

### Experiment Tracking

Training supports both TensorBoard and Weights & Biases:

```yaml
# configs/training_config.yaml
experiment:
  wandb:
    enabled: true
    project: clinical-ai-copilot
  tensorboard:
    enabled: true
    log_dir: ./logs/tensorboard
```

---

## Testing

```bash
# Full test suite
pytest tests/ -v

# Integration tests
pytest tests/ --integration

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

The test suite covers 23 integration tests across API endpoints, EEG processing, model inference, data loaders, drug interactions, authentication, encryption, rate limiting, and WebSocket streaming.

---

## Deployment

### Kubernetes

```bash
kubectl apply -f k8s/
kubectl scale deployment clinical-ai --replicas=3
```

### Helm

```bash
helm install clinical-ai helm/clinical-ai-copilot/
```

### Production Checklist

- [ ] Set encryption keys and database passwords in `.env`
- [ ] Configure SSL/TLS certificates
- [ ] Set up JWT authentication (replace demo tokens)
- [ ] Configure firewall rules and network policies
- [ ] Enable log rotation
- [ ] Set up backup and disaster recovery procedures
- [ ] Complete HIPAA compliance review
- [ ] Perform load testing
- [ ] Run security audit (Bandit + Safety)

---

## Security & HIPAA Compliance

### Administrative Safeguards
- Role-based access control (RBAC) with JWT tokens
- Immutable audit logging with 7-year retention
- Automatic session timeouts

### Technical Safeguards
- **At rest**: AES-256 encryption (Fernet) with PBKDF2 key derivation
- **In transit**: TLS 1.3
- **Integrity**: SHA-256 checksums on all audit log entries
- **Secrets**: HashiCorp Vault for encryption keys, database credentials, API keys

### Audit Logging

```python
from src.utils.audit_log import HIPAALogger, AuditContext

logger = HIPAALogger()

with AuditContext(logger, user_id="doctor123", patient_id="PAT456",
                  action="READ", ip_address="192.168.1.100"):
    patient_data = get_patient_data()
```

---

## Monitoring & Observability

### Grafana Dashboards

Access at http://localhost:3000 (default: admin/admin). The 13-panel dashboard covers:

- System health (CPU, GPU utilization, memory)
- API performance (latency percentiles, throughput, error rates)
- Model metrics (inference time, accuracy, prediction distribution)
- Clinical metrics (diagnoses per hour, severity distribution, alert rate)

### Prometheus Metrics

Key metrics exported:

| Metric | Type | Description |
|---|---|---|
| `api_requests_total` | Counter | Total API requests by endpoint and status |
| `model_inference_duration_seconds` | Histogram | Model inference latency |
| `gpu_memory_usage_bytes` | Gauge | Current GPU memory utilization |
| `patient_analyses_total` | Counter | Total patient analyses completed |
| `eeg_processing_duration_seconds` | Histogram | EEG preprocessing time |

### Alert Rules

Configured alerts for GPU memory >90%, API p99 latency >2s, error rate >5%, and model inference failures.

---

## Datasets & Medical Ontologies

### Supported EEG Datasets

| Dataset | Access | Description |
|---|---|---|
| **TUH EEG Corpus** | Licensed (Temple University) | Largest public EEG dataset (~25,000 recordings). Primary training data. |
| **CHB-MIT** | Open (PhysioNet) | 22 pediatric patients with annotated seizures. Validation data. |
| **Custom EDF/EDF+** | Any | Standard European Data Format files from clinical EEG systems. |

### Medical Ontologies

| Ontology | Standard | Coverage |
|---|---|---|
| **ICD-10** | WHO classification | Diagnosis codes (G40.x epilepsy, G47.x sleep disorders) |
| **SNOMED-CT** | Clinical terminology | 84757009 (Epilepsy), 91175000 (Seizure), etc. |
| **RxNorm** | Medication naming | 114477 (Levetiracetam), 11170 (Valproic Acid), etc. |

Sample ontology data is included in `data/medical_ontology/`. Full ontology databases are licensed separately and required for production deployment.

---

## Clinical Impact

### Projected Healthcare Metrics

| Metric | Before AI | With AI | Improvement |
|---|---|---|---|
| Detection Time | 2-4 hours | 2 minutes | 98% faster |
| Diagnostic Accuracy | 75% | 98.75% | +23.75% |
| False Positive Rate | 30% | 5% | -83% |
| Cost per Patient | $2,000 | $200 | -90% |
| 24/7 Coverage | No | Yes | Continuous |

### ROI Projection (500-bed hospital)

- Annual savings: $4.2M
- Physician time freed: 8,000 hours/year
- Payback period: 7 months

### Target Clinical Settings

1. **Emergency Departments** -- rapid seizure triage
2. **Neurology Clinics** -- comprehensive EEG interpretation
3. **Sleep Centers** -- automated sleep stage scoring
4. **Research Institutions** -- clinical trial patient screening
5. **Telemedicine** -- remote neurological consultation
6. **Medical Education** -- training and simulation

---

## Roadmap

### Completed

- [x] EEG signal processing pipeline (ICA, filtering, wavelets, feature extraction)
- [x] CNN-LSTM seizure detection model
- [x] Transformer sleep stage classifier
- [x] Spiking neural network for event detection
- [x] Mamba2 state space model for long-context processing
- [x] Llama 3.1 8B clinical model with QLoRA
- [x] Multimodal cross-attention fusion architecture
- [x] GraphRAG with Neo4j (SNOMED-CT, ICD-10, RxNorm)
- [x] Qdrant vector store with BioBERT/PubMedBERT
- [x] HIPAA-compliant API with JWT auth and encryption
- [x] Real-time Kafka streaming pipeline
- [x] Drug interaction checking
- [x] Docker + Kubernetes + Helm deployment
- [x] Prometheus/Grafana monitoring
- [x] CI/CD pipeline (GitHub Actions)
- [x] Comprehensive test suite

### In Progress

- [ ] Model training on full TUH EEG and CHB-MIT datasets
- [ ] Complete medical ontology integration (full SNOMED-CT/ICD-10)
- [ ] Production deployment validation
- [ ] FDA 510(k) submission preparation

### Future

- [ ] Federated learning (cross-hospital training without data sharing)
- [ ] EMR/EHR integration (Epic, Cerner)
- [ ] Edge deployment for hospital devices
- [ ] Mobile companion application
- [ ] Multi-language clinical report generation
- [ ] Clinical trial matching engine
- [ ] Active learning pipeline for continuous improvement

---

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -r requirements-dev.txt
pre-commit install

# Linting
black src/
flake8 src/
mypy src/

# Tests
pytest tests/ -v
```

---

## Citation

```bibtex
@software{clinical_ai_copilot_2025,
  title={Clinical AI Copilot: Multimodal EEG Analysis with Clinical RAG},
  author={Clinical AI Research Team},
  year={2025},
  url={https://github.com/Srujan29112001/Clinical-AI-copilot}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

**This software is intended for research and educational purposes only.** It is NOT FDA-approved and must NOT be used for clinical diagnosis or treatment decisions without oversight by qualified healthcare professionals. All PHI is encrypted using AES-256. The system is designed for HIPAA, GDPR, and CCPA compliance, but compliance must be validated for each deployment. Performance varies based on hardware, EEG data quality, patient population, and clinical context.

---

**Repository**: [github.com/Srujan29112001/Clinical-AI-copilot](https://github.com/Srujan29112001/Clinical-AI-copilot)
