<div align="center">

# 🤖 Robotics Seams Types Prediction

**An end-to-end deep learning pipeline for classifying robotic seam types using a custom PyTorch architecture — from data to deployment.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Overview

**Robotics Seams Types Prediction** is a complete deep learning project that classifies different types of robotic seams (welding seams) from image data. The model is a **custom convolutional neural network** built from scratch using PyTorch — not a fine-tuned pre-trained model — with a modular training pipeline and a production-ready REST API.

This project demonstrates the full lifecycle of an AI system: **data preparation → model design → training → evaluation → deployment**.

> 🎯 Built to solve a real industrial problem: automating seam quality inspection in robotics and welding applications.

---

## ✨ Features

- 🧠 **Custom CNN architecture** built from scratch in PyTorch
- 📦 **Modular codebase** — clean separation of data, model, training, and API
- 🔄 **Advanced data augmentation** with `torchvision.transforms`
- 🚀 **FastAPI REST API** for real-time inference
- 📊 **Full training pipeline** with checkpointing, metrics tracking, and validation

- 📁 **Configurable** — change classes, image size, and hyperparameters easily

---

## 🏗️ Architecture

The model architecture is defined in [`architecture.py`](architecture.py). It's a custom CNN designed specifically for this classification task.
