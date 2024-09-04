from setuptools import setup, find_packages

# Load requirements from requirements.txt
with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setup(
    name="codi_bg",
    version="0.1.3",
    description="CoDi - Contrastive Distance reference-based cell type annotation for spatial transcriptomics.",
    url="https://github.com/stomics/codi",
    author="Vladimir Kovacevic",
    author_email="vladimirkovacevic@genomics.cn",
    license="MIT",
    package_dir={"": "core"},  # Specifies that packages are under the core directory
    # packages=find_packages(where="core"),  # Finds packages in the core directory
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'CoDi=CoDi:main',  # Points to the main function in CoDi.py
        ],
    },
    python_requires='>=3.8',
)

