from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="verlet",
        sources=["verlet.pyx"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="verlet",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[np.get_include()],
)