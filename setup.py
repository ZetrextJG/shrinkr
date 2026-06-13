import os
import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

C_SRC_DIR = "src"
include_dirs = [np.get_include(), C_SRC_DIR]
library_dirs = []

if sys.platform == "darwin":
    mac_includes = ["/opt/homebrew/opt/libomp/include", "/usr/local/opt/libomp/include"]
    mac_libs = ["/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib"]
    include_dirs.extend([p for p in mac_includes if os.path.exists(p)])
    library_dirs.extend([p for p in mac_libs if os.path.exists(p)])


ext_modules = [
    Extension(
        "shrinkr._native",
        sources=[
            "shrinkr/bindings.c",
            "src/c_shrinkr.c",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
    )
]


class HighPerfBuildExt(build_ext):
    """Inject compiler-specific optimization flags."""

    def build_extensions(self):
        compiler_type = self.compiler.compiler_type

        compile_args = []
        link_args = []

        if sys.platform == "win32":
            compile_args.extend(["/O2", "/openmp"])
            use_avx2 = os.environ.get("USE_AVX2", "0") == "1"
            if compiler_type == "msvc" and use_avx2:
                compile_args.append("/arch:AVX2")
            else:
                print(
                    "Notice: Compiling with default MSVC instructions. Set USE_AVX2=1 to force AVX2."
                )
        elif sys.platform == "darwin":
            compile_args.extend(["-O3", "-flto", "-Xpreprocessor", "-fopenmp"])
            link_args.extend(["-lomp", "-lm", "-flto"])
        elif sys.platform == "linux":
            compile_args.extend(["-O3", "-march=native", "-flto", "-fopenmp"])
            link_args.extend(["-flto", "-lm", "-fopenmp"])
        else:
            print("Unsupported platform detected. Contact package developer.")

        for ext in self.extensions:
            ext.extra_compile_args = compile_args
            ext.extra_link_args = link_args

        super().build_extensions()


setup(
    packages=["shrinkr"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": HighPerfBuildExt},
)
