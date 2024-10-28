from setuptools import setup, find_packages

setup(
    name='mnist-visual-fields',
    author='Henrique Uhlmann Gobbi',
    author_email='hgugobbi@gmail.com',
    version='2.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.15.0',
        'numpy',
        'matplotlib',
        'attrs',
        'ipywidgets'
    ],
    url='https://github.com/hugobbi/mnist-visual-fields',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description='Code to handle visualization of neural networks created in Keras. It was made\
                to visualize neural networks with two inputs that receive competitive stimuli,\
                although the code also works for other types of neural networks.'
)
