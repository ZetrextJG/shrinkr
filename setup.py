from setuptools import setup, Extension

ext_modules = [
    Extension(
        "shrinkr._native",
        sources=[
            "shrinkr/bindings.c",
            "src/c_shrinkr.c",
        ],
        include_dirs=["src"],
    )
]

setup(
    packages=["shrinkr"],
    ext_modules=ext_modules,
)
