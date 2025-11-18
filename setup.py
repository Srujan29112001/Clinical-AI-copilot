"""
Clinical AI Copilot - Multimodal EEG Analysis with Clinical RAG
Enterprise-grade healthcare AI system for neurological disorder detection
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clinical-ai-copilot",
    version="1.0.0",
    author="Clinical AI Research Team",
    author_email="research@clinicalai.com",
    description="Multimodal EEG Analysis with Clinical RAG for Neurological Disorders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clinical-ai/copilot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12.0",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "tensorrt": [
            "tensorrt>=8.6.0",
            "torch-tensorrt>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "clinical-ai=api.main:main",
            "eeg-process=signal_processing.eeg_processor:main",
            "train-models=scripts.train_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "healthcare",
        "eeg",
        "neurology",
        "deep-learning",
        "rag",
        "clinical-ai",
        "seizure-detection",
        "medical-ai",
        "hipaa-compliant",
    ],
    project_urls={
        "Bug Reports": "https://github.com/clinical-ai/copilot/issues",
        "Documentation": "https://docs.clinicalai.com",
        "Source": "https://github.com/clinical-ai/copilot",
    },
)
