import codecs
import os.path

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return codecs.open(os.path.join(HERE, *parts), 'r').read()


def get_dependencies(subpackage="requirements"):
    with open(os.path.join("requirements", subpackage+".txt"), encoding="utf-8") as fh:
        return fh.read().splitlines()


setup(
    name='cloud_array',
    version='0.0.1',
    author="Michal Murawski",
    author_email="mmurawski777@gmail.com",
    description="Cloud implementation of array for Big Data",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/micmurawski/cloud-array/",
    package_dir={"": "src"},
    packages=find_packages(exclude=(
        'build',
        'tests',
    )),
    install_requires=get_dependencies(),
    extras_requires={
        "": get_dependencies(),
        "all": get_dependencies("all")
    },
    include_package_data=True,
    python_requires=">=3.7,<4.0",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
)
