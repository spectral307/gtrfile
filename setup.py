from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(name="gtrfile",
      version="0.0.0",
      py_modules=["gtrfile"],
      install_requires=requirements,
      test_suite="tests",
      tests_require=["unittest"])
