from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [p.strip() for p in f.readlines()]

with open('README.md') as f:
    long_description = f.read()


setup(
    name='torchfuel',
    version='0.1.0',
    description='Library to performe Friedman test and the Nemenyi post-hoc analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vturrisi/torchfuel',
    author='Victor Turrisi',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False
)
