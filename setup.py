from setuptools import setup, find_packages

setup(
    name="smts",
    version="1.0.0",
    description="Starling Murmuration Topological Search (SMTS) Continuous Metaheuristic",
    author="Malik Abu Alzait",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
