from setuptools import Extension, setup

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
