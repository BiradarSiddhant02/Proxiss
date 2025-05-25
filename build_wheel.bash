#!/bin/bash
set -e

# # Build Windows wheels for x86_64
# CIBW_PLATFORM=windows
# CIBW_ARCHS="AMD64"
# cibuildwheel --output-dir dist --platform windows --arch "$CIBW_ARCHS"

# Build Linux wheels for x86_64
CIBW_PLATFORM=linux
CIBW_ARCHS="x86_64"
cibuildwheel --output-dir dist --platform linux --arch "$CIBW_ARCHS"
