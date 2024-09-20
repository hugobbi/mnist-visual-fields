from setuptools import setup, find_packages

setup(
    name='mnist-visual-fields',
    author='Henrique Uhlmann Gobbi',
    author_email='hgugobbi@gmail.com',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.15.0',
        'numpy',
        'matplotlib',
        'attr',
        'ipywidgets'
    ],
    url='https://github.com/hugobbi/mnist-visual-fields',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description='Code to handle creation and visualization of neural networks using Keras. It was made to work with the MNIST dataset, and the neural networks can have one or two visual fields (inputs).'
)
