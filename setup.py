from setuptools import find_packages, setup


with open("README.md") as readme:
    readme_description = readme.read()

setup(
    packages=find_packages(where="."),
    long_description=readme_description,
    include_package_data=True
)