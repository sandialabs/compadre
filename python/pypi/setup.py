import setuptools
#https://packaging.python.org/tutorials/packaging-projects/

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="compadre",
    version="1.0.5",
    author="Paul Kuberry",
    author_email="pkuberry@gmail.com",
    description="Compatible Particle Discretization and Remap",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SNLComputation/compadre",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
)
