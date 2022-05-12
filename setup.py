import setuptools
import inspect
import sys
import os
import re

REQUIREMENTS_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "env",  "requirements.txt")
with open(REQUIREMENTS_PATH) as f:
    REQUIREMENTS = f.read().splitlines()

if not hasattr(setuptools, 'find_namespace_packages') or not inspect.ismethod(setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
          "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__))
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = re.sub(
        "<!--- long-description-skip-begin -->.*<!--- long-description-skip-end -->",
        "",
        readme_file.read(),
        flags=re.S | re.M,
    )

setuptools.setup(
    name='quantum-image-classifier',
    version=VERSION,
    description='Quatum image classifier: A library of different quantum algorithms used to classify images',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/jorgevazquezperez/Quantum-image-classifier',
    author='Jorge Vázquez Pérez',
    author_email='jorge.vazper@gamil.com',
    #license='Apache-2.0',
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering"
    ],
    keywords='qiskit quantum machine learning ml centroids',
    packages=setuptools.find_packages(include=['quantum_image_classifier']),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False
)