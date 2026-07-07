# HW03 Docker Image Size Report

| Repository              | Tag       | Size   |
|-------------------------|-----------|--------|
| qbc12-airbnb-serving    | naive     | 3.13GB |
| qbc12-airbnb-serving    | optimized | 1.33GB |

## Analysis
The optimized image is 1.80 GB (58%) smaller than the naive image. The naive Dockerfile uses the full `python:3.11` image which includes build tools, compilers, and development headers. The optimized Dockerfile uses `python:3.11-slim` in a multi-stage build: the builder stage installs dependencies, then the runtime stage copies only the installed site-packages and application source, leaving behind pip, setuptools, wheel, and the entire build toolchain. In production I would use the optimized image because it reduces pull time, disk usage, and attack surface — fewer installed packages means fewer potential vulnerabilities.