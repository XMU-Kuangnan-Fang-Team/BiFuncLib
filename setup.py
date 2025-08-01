from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='BiFuncLib',
    version='0.0.1',
    description='A Python library for biclustering with functional data',
    author='Yuhao Zhong',
    author_email='Barry57@163.com',
    url='https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'GENetLib',
        'scikit-learn',
        'scikit-learn-extra',
        'seaborn',
        'networkx'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
    ],
)
