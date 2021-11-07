FROM python:3.9-slim
WORKDIR usr/src/app
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD ["python3", "server.py"]