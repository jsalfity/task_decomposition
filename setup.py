# read the contents of your README file
from os import path

from setuptools import find_packages, setup

setup(
    name="task_decomposition",
    packages=[
        package
        for package in find_packages()
        if package.startswith("task_decomposition")
    ],
    install_requires=["robosuite", "mujoco"],
    author="Jonathan Salfity",
    author_email="j.salfity@utexas.edu",
    version="0.0.1",
)
