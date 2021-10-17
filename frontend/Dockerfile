# BASE IMAGE
FROM python:3.9-slim

# ENVIRONMENT VARIABLES
# Zet warnings over nieuwe pip versies uit (de pip versie is onderdeel
# van het base image)
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV SERV_URL ""

# Zet the working directory (de directory van waaruit de commandos uitgevoerd
# worden)
WORKDIR /usr/src/app

# Kopieer de requirements.txt file en installeer dependencies
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

# Installeer gunicorn (production ready Python server)
RUN pip install gunicorn

# Expose deze poorten
EXPOSE 8080

# Kopieer de applicatie naar de container
COPY . .

# Run het entrypoint script
CMD ["gunicorn", "-b", "0.0.0.0:8080", "ganweb:app"]