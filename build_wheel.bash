#!/bin/bash

# Build manylinux wheels with cibuildwheel
# Excludes musllinux and 32-bit architectures (i686)

CIBW_BEFORE_ALL_LINUX="yum install -y clang" \
CIBW_SKIP="*-musllinux* *i686*" \
CIBW_BUILD="cp39-* cp310-* cp311-* cp312-* cp313-*" \
uv run cibuildwheel --output-dir dist --platform linux --archs x86_64