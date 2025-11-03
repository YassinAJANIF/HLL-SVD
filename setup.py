# setup.py
from setuptools import setup, find_packages
from pathlib import Path

# Dépendances de base (sans CuPy)
base_reqs = [
    r.strip() for r in Path("requirements.txt").read_text().splitlines()
    if r.strip() and not r.strip().startswith("#") and not r.lower().startswith("cupy")
]

setup(
    name="HLL-SVD",
    packages=find_packages(),
    install_requires=base_reqs,           # pas de cupy ici
    extras_require={
        "gpu-cuda12x": ["cupy-cuda12x"],  # CUDA 12.x
        "gpu-cuda11x": ["cupy-cuda11x"],  # CUDA 11.x
    },
    # ... (le reste inchangé)
)

