#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name             = "crimg",
      version          = "1.0",
      description      = "Utilities for Cosmic Ray detection algorithm",
      author           = "Ignacio Francisco Ramirez Paulino",
      author_email     = "nacho@fing.edu.uy",
      maintainer       = "nacho@fing.edu.uy",
      url              = "https://iie.fing.edu.uy/personal/nacho/",
      ext_modules      = [
          Extension(
              'crimg', ['c_src/crimg.c'],
              libraries = ['gomp'],
              extra_compile_args=["-Wall", "-fopenmp", "-O3", "-march=native", "-mtune=native"]),
     ], 
      
)
