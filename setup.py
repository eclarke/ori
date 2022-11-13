from re import M
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here/"README.md").read_text(encoding="utf-8")

setup(
    name = "ori",
    version = "0.0.1",
    description = "An interactive, CLI-based FITS file explorer",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/eclarke/fitz",
    author = "Erik Clarke",
    classifiers = [
        "Development Status :: 3 - Alpha"
    ],
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas',
        'astropy',
        'questionary',
        'tqdm'
    ],
    entry_points = {
        "console_scripts": [
            "ori=ori:main"
        ]
    },
    package_data = {
        "ori": ["*.yaml"]
    }
)