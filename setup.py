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
    author_email="Sebastian Achim Mueller@mail",
    packages=[
        "spherical_histogram",
    ],
    package_data={"spherical_histogram": []},
    install_requires=[
        "spherical_coordinates",
        "solid_angle_utils",
        "binning_utils",
        "triangle_mesh_io",
        "merlict",
        "svg_cartesian_plot",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
)
