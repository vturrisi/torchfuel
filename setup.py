from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


with open('requirements.txt') as f:
    requirements = [p.strip() for p in f.readlines()]


setup(
    name='torchfuel',
    version='0.1.3',
    description='Library build on top of pytorch to fuel productivity',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vturrisi/torchfuel',
    author='Victor Turrisi',
    author_email='vt.turrisi@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
