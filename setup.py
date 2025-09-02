from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='BiFuncLib',
    version='0.0.8',
    description='A Python library for biclustering with functional data',
    author='Yuhao Zhong',
    author_email='Barry57@163.com',
    url='https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/',
    license='MIT',
    packages=find_packages(),
    package_data={
        'BiFuncLib': ['simulation_data/*'],
    },
    install_requires=[
        'numpy==1.24.3',
        'pandas',
        'matplotlib',
        'scipy',
        'GENetLib==1.2.6',
        'scikit-learn==1.2.2',
        'scikit-learn-extra==0.3.0',
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








