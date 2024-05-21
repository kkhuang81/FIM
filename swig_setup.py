#!/usr/bin/env python

"""
    setup.py file for SWIG
"""
from setuptools import setup
from setuptools.command.build_ext import build_ext
from distutils.core import setup, Extension
class BuildExt(build_ext):
    def build_extensions(self):
        #self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()

test_module = Extension('_Diffusion',
                        sources=['precompute_wrap.cxx','precompute/diffusion.cpp'],
                        swig_opts=['-c++'],
                        extra_compile_args=['-std=c++11', '-O3','-pthread','-march=core2','-lcnpy','-lz'],
                        extra_link_args=['-std=c++11', '-O3','-pthread','-march=core2','-lcnpy','-lz']) 
                        
setup(name = 'Diffusion',
        version = '0.1',
        cmdclass={'build_ext': BuildExt},
        ext_modules = [test_module],
        py_modules = ['Diffusion'],)
