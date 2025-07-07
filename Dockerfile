FROM python:3.10-slim

# 禁用代理变量
ENV HTTP_PROXY=""
ENV http_proxy=""
ENV HTTPS_PROXY=""
ENV https_proxy=""
ENV NO_PROXY="*"

WORKDIR /app

# （可选）换 Debian 源
# RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 复制所有源代码
COPY . /app

# 复制 wheelhouse
COPY wheelhouse /wheelhouse

# 离线安装 tomli
RUN pip install --no-index --find-links=/wheelhouse tomli

# 离线安装 requirements.txt
RUN pip install --no-index --find-links=/wheelhouse -r requirements.txt

EXPOSE 7860

CMD ["python", "Gradio_V11.py"]
