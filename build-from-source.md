```bash
python -m build --wheel
auditwheel repair dist/proxi-*.whl --plat linux_x86_64 -w dist