FROM python:3.10-slim

WORKDIR /app_v2

RUN pip install flask pillow gradio requests \
    && apt-get update && apt-get install -y supervisor \
    && mkdir -p /var/log/supervisor

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY . .

EXPOSE 5000 7860

CMD ["/usr/bin/supervisord"]
