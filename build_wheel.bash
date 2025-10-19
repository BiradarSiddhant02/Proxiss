#!/bin/bash

# Build manylinux wheels with cibuildwheel
# Skips musllinux builds

CIBW_BEFORE_ALL_LINUX="yum install -y clang++" \
CIBW_ENVIRONMENT="CMAKE_CXX_COMPILER=clang++ CMAKE_C_COMPILER=clang" \
CIBW_SKIP="*-musllinux*" \
cibuildwheel --platform linux --archs x86_64