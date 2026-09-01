# Physical Layer Modeling using Synchronous Machine Learning Techniques

## Deep Learning Surrogate Modeling of Knee Meniscus Biomechanics

This repository contains the code developed as part of my thesis, **“Physical Layer Modeling using Synchronous Machine Learning Techniques.”**

The project investigates the use of deep learning as a surrogate for computationally expensive finite element simulations of the human knee. A finite element knee model was simulated in **FEBio** under walking-related loading conditions, including time-dependent compressive loading and knee flexion–extension. The resulting biomechanical response of the medial and lateral menisci was exported as image-based field maps and used to train a convolutional neural network.

The proposed surrogate model is based on a **2D U-Net architecture with scalar conditioning at the bottleneck**. Its objective is to approximate FEBio-derived biomechanical fields at substantially lower computational cost.

---

## Model Inputs

For each simulation state, the neural network receives two spatial inputs:

- **Meniscus mask** – binary image defining the meniscal geometry.
- **Reference height map** – grayscale representation of the initial meniscus geometry.

Three scalar variables are also provided:

- **Time** `t`
- **Normalized compressive force** `F_norm`
- **Normalized knee flexion angle** `theta_norm`

The scalar variables are processed through a multilayer perceptron and incorporated into the U-Net bottleneck.

---

## Predicted Outputs

The network simultaneously predicts four spatial biomechanical fields:

1. **Z-displacement**
2. **Effective Lagrange strain**
3. **Von Mises stress**
4. **Contact pressure**

The outputs are predicted as image-based field maps corresponding to the finite element results obtained from FEBio.

---

## Network Architecture

The surrogate model follows a standard encoder–decoder U-Net structure consisting of:

- Convolutional encoding blocks
- Max-pooling operations
- A latent bottleneck representation
- Scalar conditioning through a multilayer perceptron
- Transposed-convolution upsampling
- Skip connections between corresponding encoder and decoder levels
- A final convolutional layer producing four output channels

The skip connections preserve spatial information during reconstruction, while the scalar-conditioning branch allows the predicted biomechanical response to vary according to the loading and kinematic state.

---

## Repository Structure

```text
.
├── final_unet.py
├── meniscus_dataset.py
├── train_unet.py
├── viz_unet_outputs.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
