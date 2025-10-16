#!/bin/bash

# Build manylinux wheels with cibuildwheel
# Skips musllinux builds

CIBW_BEFORE_ALL_LINUX="yum install -y gcc-c++" \
CIBW_ENVIRONMENT="CMAKE_CXX_COMPILER=g++ CMAKE_C_COMPILER=gcc" \
CIBW_SKIP="*-musllinux*" \
cibuildwheel --platform linux --archs x86_64