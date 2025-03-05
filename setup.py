import setuptools
import os


with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join("spherical_histogram", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")

setuptools.setup(
    name="spherical_histogram",
    version=version,
    description=("This is spherical_histogram."),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/spherical_histogram",
    author="Sebastian Achim Mueller",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "spherical_histogram",
    ],
    package_data={"spherical_histogram": []},
    install_requires=[
        "spherical_coordinates>=0.1.6",
        "solid_angle_utils>=0.1.2",
        "binning_utils_sebastian-achim-mueller>=0.0.19",
        "triangle_mesh_io>=0.0.4",
        "merlict>=0.2.0.2.2.6",
        "svg_cartesian_plot>=0.0.11",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
)
