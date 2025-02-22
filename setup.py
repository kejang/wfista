import setuptools
import os
from pathlib import Path

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

module_path = Path(os.path.abspath(__file__)).parent.absolute()
package_name = "wfista"

try:
    pkg_version = version(package_name)
except Exception:
    pkg_version = "0.0.1"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=package_name,
    version=pkg_version,
    author="Kwang Eun Jang",
    author_email="kejang@stanford.edu",
    description="Wavelet-FISTA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kejang/wfista",
    project_urls={
        "Bug Tracker": "https://github.com/kejang/wfista/issues",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.10",
    include_package_data=True,
    package_data={
        "wfista": [
            "thresholding/cuda/*.cu",
        ]
    },
)