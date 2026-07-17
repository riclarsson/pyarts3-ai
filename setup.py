from setuptools import setup, find_packages

setup(
    name="pyarts3_ai",
    version="0.1.0",
    description="A library tool that allows AI tools to use ARTS interactively via a generic Python interface.",
    authors=["Your Name <ric.larssson@gmail.com>"],
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
