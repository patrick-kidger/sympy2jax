# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import re

import setuptools


_here = pathlib.Path(__file__).resolve().parent

name = "sympy2jax"

# for simplicity we actually store the version in the __version__ attribute in the source
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

author = "Patrick Kidger"

author_email = "contact@kidger.site"

description = "Turn SymPy expressions into trainable JAX expressions."

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/google/sympy2jax"

license = "Apache-2.0"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
]

python_requires = "~=3.7"

install_requires = ["equinox>=0.5.3", "jax>=0.3.4", "sympy>=1.7.1"]

setuptools.setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    packages=[name],
)
