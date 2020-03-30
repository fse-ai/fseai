"""
Code With fseai
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fseai",
    version="0.0.1",
    author="Hanoona Rasheed",
    author_email="hanoona@uniqueroboticsedu.com",
    description="Full Stack Engineering AI course package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License, Version 2.0 (Apache-2.0)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
