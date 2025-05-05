# setup.py
import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        # ensure build_temp exists
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
        )
        # build
        subprocess.check_call(
            ["cmake", "--build", "."],
            cwd=self.build_temp,
        )

setup(
    name="proxi",
    version="0.1",
    author="Siddhant Biradar",
    description="Proxi: Accelerating nearest-neighbor search for high-dimensional data!",
    ext_modules=[CMakeExtension("proxi", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
