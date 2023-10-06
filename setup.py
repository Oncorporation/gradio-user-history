from setuptools import find_packages, setup


def get_version() -> str:
    rel_path = "src/gradio_user_history/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


install_requires = [
    "gradio[oauth]>=3.44",
]

extras = {}

extras["dev"] = [
    "ruff",
    "black",
    "mypy",
]


setup(
    name="gradio_user_history",
    version=get_version(),
    author="Lucain Pouget",
    author_email="lucain@huggingface.co",
    description="A package to store user history in a gradio app.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="gradio oauth machine-learning",
    license="Apache",
    url="https://huggingface.co/spaces/Wauplin/gradio-user-history",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require=extras,
    python_requires=">=3.8.0",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
