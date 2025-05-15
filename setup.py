from setuptools import setup, find_packages

setup(
    name='Name',
    version='0.1.0',
    author='',
    author_email='',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',  # ici il faut mettre le lien  GitHub
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'mpi4py',
        'cupy',
        'h5py',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

