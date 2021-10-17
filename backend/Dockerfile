FROM python:3.9-slim
WORKDIR usr/src/app
COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD ["python3", "api.py"]