from setuptools import find_packages, setup

setup(
    name='ml_project_sentiment',
    version='0.1.0',
    description='A sentiment analysis project for movie reviews.',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "torch",
        "fastapi",
        "streamlit"
    ],
)