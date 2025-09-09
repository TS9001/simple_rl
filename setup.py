from setuptools import setup, find_packages

setup(
    name="simple-rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Simple implementations of reinforcement learning algorithms with HuggingFace integration",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simple_rl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "wandb>=0.15.0",
        "gym>=0.26.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "simple-rl-train=simple_rl.scripts.train:main",
        ],
    },
)