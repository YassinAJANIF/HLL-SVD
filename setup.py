from setuptools import setup, find_packages

# Charger les dépendances depuis un fichier requirements.txt
def load_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="nom_du_projet",
    version="1.0.0",
    author="Votre Nom",
    author_email="votre_email@example.com",
    description="Description de votre projet",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/votre_nom/votre_projet",
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt"),  # Charge les dépendances
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Version minimale de Python
)
