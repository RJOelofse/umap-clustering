from setuptools import setup


def readme():
    try:
        with open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Python 2.7 doesn't support encoding argument in builtin open
        import io

        with io.open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()


configuration = {
    "name": "umap-clustering",
    "version": "0.5.3",
    "description": "Uniform Manifold Approximation and Projection",
    "long_description": readme(),
    "long_description_content_type": "text/x-rst",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    "keywords": "dimension reduction t-sne manifold clustering",
    "url": "http://github.com/RJOelofse/umap-clustering",
    "maintainer": "Rudolf Oelofse",
    "maintainer_email": "rj.oelofse@gmail.com",
    "license": "BSD",
    "packages": ["umappc"],
    "install_requires": [
        "numpy >= 1.17",
        "scipy >= 1.3.1",
        "scikit-learn >= 1.2",
        "numba >= 0.51.2",
        "pynndescent >= 0.5",
        "tbb >= 2019.0",
        "pandas",
        "tqdm",
    ],
    "extras_require": {
        "plot": [
            # "pandas",
            "matplotlib",
            "datashader",
            "bokeh",
            "holoviews",
            "colorcet",
            "seaborn",
            "scikit-image",
        ],
        "parametric_umap": ["tensorflow >= 2.1", "tensorflow-probability >= 0.10"],
    },
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
