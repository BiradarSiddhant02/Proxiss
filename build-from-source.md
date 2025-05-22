```bash
python -m build --wheel
auditwheel repair dist/proxiss-*.whl --plat linux_x86_64 -w dist