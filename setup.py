import setuptools


pkg_name = "wfista"
pkg_version = "0.0.2"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=pkg_name,
    version=pkg_version,
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    package_data={
        "wfista": [
            "thresholding/cuda/*.cu",
        ]
    },
    author="Kwang Eun Jang",
    author_email="kejang@stanford.edu",
    description="Wavelet-FISTA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kejang/wfista",
    project_urls={
        "Bug Tracker": "https://github.com/kejang/wfista/issues",
    },
)
