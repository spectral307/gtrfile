from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(name="gtrfile",
      version="1.0.0",
      py_modules=["gtrfile"],
      install_requires=requirements)
