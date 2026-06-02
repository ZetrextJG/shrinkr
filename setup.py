import os

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class HighPerfBuildExt(build_ext):
    """Inject compiler-specific optimization flags."""

    def build_extensions(self):
        compiler_type = self.compiler.compiler_type

        use_avx2 = os.environ.get("USE_AVX2", "0") == "1"

        if compiler_type == "msvc":
            c_opts = ["/O2", "/openmp"]
            if use_avx2:
                c_opts.append("/arch:AVX2")
                print("Notice: Compiling with AVX2 optimizations enabled.")
            else:
                print(
                    "Notice: Compiling with default MSVC instructions. Set USE_AVX2=1 to force AVX2."
                )
            l_opts = []
        else:
            c_opts = ["-O3", "-march=native", "-flto", "-fopenmp"]
            # Link math library (-lm) and apply Link-Time Optimization (-flto)
            l_opts = ["-flto", "-lm", "-fopenmp"]

        for ext in self.extensions:
            ext.extra_compile_args = c_opts
            ext.extra_link_args = l_opts

        super().build_extensions()


ext_modules = [
    Extension(
        "shrinkr._native",
        sources=[
            "shrinkr/bindings.c",
            "src/c_shrinkr.c",
        ],
        include_dirs=["src", numpy.get_include()],
    )
]

setup(
    packages=["shrinkr"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": HighPerfBuildExt},
)
