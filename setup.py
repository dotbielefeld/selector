import pathlib
from setuptools import setup


HERE = pathlib.Path(__file__).parent


README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="selector",
    version="0.0.1",
    description=" ",
    long_description=README,
    long_description_content_type=" ",
    url=" ",
    author=" ",
    author_email=" ",
    license=" ",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=[],
    entry_points={},
)