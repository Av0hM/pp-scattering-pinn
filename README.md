# Physics-Informed Neural Network for Proton–Proton Scattering Probabilities

This repository presents a physics-informed neural network (PINN) framework to model elastic and inelastic probabilities in proton–proton (pp) scattering in the 300–450 MeV energy range. The model integrates sparse experimental nuclear physics data with physical constraints to produce smooth, bounded, and uncertainty-aware probability estimates.

The project focuses on probability-level modeling rather than exclusive channel cross sections, enabling robust inference even when experimental data are limited.

---

## 📌 Scientific Motivation

In the intermediate-energy regime (300–450 MeV), proton–proton scattering transitions from elastic dominance to inelastic processes driven primarily by single-pion production via the Δ(1232) resonance. While elastic differential cross sections and total cross sections are experimentally available, exclusive pion-production data are sparse and incomplete.

This work addresses the question:

> *Can physically meaningful elastic and inelastic reaction probabilities be inferred from limited experimental data using physics-informed machine learning?*

---

## 🎯 Objectives

- Integrate experimental elastic differential cross sections to obtain total elastic cross sections
- Align elastic and total pp cross sections to construct reaction probabilities
- Model elastic probability \(P_{el}(E)\) using a physics-informed neural network
- Obtain inelastic probability explicitly via \(P_{inel}(E) = 1 - P_{el}(E)\)
- Quantify epistemic uncertainty using Monte-Carlo dropout
- Ensure exact probability conservation and physical bounds

---

## 📊 Data Sources

- **Elastic differential cross sections**: EXFOR nuclear data library  
- **Total pp cross sections**: historical compilations / SAID-style datasets  
- **Energy range**: 300–450 MeV  
- **Units**: millibarns (mb), probabilities dimensionless  

Raw data are stored under `data/raw/`, and all processed datasets used for modeling are stored under `data/processed/`.

---

## 🧠 Methodology Overview

### 1. Elastic Cross-Section Integration
Elastic differential cross sections \( d\sigma/d\Omega \) are integrated over solid angle to obtain total elastic cross sections, accounting for identical-particle symmetry.

### 2. Probability Construction
Elastic and inelastic probabilities are defined as:
\[
P_{el}(E) = \frac{\sigma_{el}(E)}{\sigma_{tot}(E)}, \quad
P_{inel}(E) = 1 - P_{el}(E)
\]

### 3. Physics-Informed Neural Network
A neural network is trained to predict \(P_{el}(E)\) subject to:
- probabilistic bounds (0 ≤ P ≤ 1 via sigmoid output)
- smoothness constraints (second-derivative regularization)
- uncertainty-weighted χ² data likelihood

### 4. Uncertainty Quantification
Monte-Carlo dropout is used at inference time to estimate predictive uncertainty, yielding credible uncertainty bands over the energy domain.

---

## 📈 Key Result

The PINN predicts a smooth crossover from elastic-dominated to inelastic-dominated scattering with increasing energy, consistent with known pp scattering physics in the Δ-resonance region. The model provides uncertainty estimates that expand naturally away from sparse experimental constraints.

<p align="center">
  <img src="figures/el_inel_combined.png" width="650">
</p>

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Av0hM/pp-scattering-pinn.git
cd pp-scattering-pinn
