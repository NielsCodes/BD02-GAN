# GAN-Demonstrator
Door Dimitri Batchev, Rade Grbić, Niels Kersic & Wilbert van de Westerlo.

# 1. Achtergrond
De GAN-demonstrator is een client-server applicatie, waarbij de server een getraind cGAN model bevat. Dit model wordt gebruikt om op basis van (tekst)input vanuit de client een afbeelding te genereren. Hierbij gaat het om cijfers die worden omgezet naar afbeeldingen (text-to-image), getraind op de MNIST-dataset.  

Omdat het model op afbeeldingen van individuele cijfers in de MNIST-dataset is getraind, kan deze ook alleen individuele cijfer-afbeeldingen genereren. Om ervoor te zorgen dat gebruikers toch afbeeldingen van grotere getallen kunnen genereren, wordt de inputwaarde op de server gesplists naar individuele cijfers. Per cijfer wordt het model uitgevoerd en wordt een afbeelding gegenereerd. Vervolgens worden deze afbeeldingen samengevoegd tot één afbeelding alvorens als response teruggestuurd te worden naar de client. In theorie is er geen limiet aan hoeveel cijfers het inputgetal kan bevatten, maar om te voorkomen dat de serverhardware overbelast raakt en om te voorkomen dat gebruikers lang moeten wachten op een response, is ervoor gekozen om de input tot 10 karakters (cijfers) te limiteren. 

De keuze voor een cGAN komt voornamelijk voort uit gelimiteerde hardware en tijd. Interactiviteit is een van de hoofddoelen van de demonstrator. Hoewel er meerder GAN-architecturen zijn die dit soort interactiviteit kunnen bereiken, zijn veel van deze architecturen extreem complex en/of moeilijk te trainen, met name op beperkte hardware. Een StackGAN of een GAN-CLS zouden meer gedetailleerde afbeeldingen kunnen genereren (wanneer ze zouden worden getraind op hogere kwaliteit data) en zouden meer complexe inputs kunnen verwerken, zoals hele zinnen. Echter, deze architecturen bestaan uit twee of meer complexe neurale netwerken, die niet of nauwelijks los van elkaar te testen zijn. Een cGAN is minder complex, omdat deze met enkele class labels werkt, in dit geval de cijfers 0 t/m 9 in de MNIST-dataset. 

De volledige code is te vinden op [GitHub](https://github.com/NielsKersic/BD02-GAN).

# 2. Opzet en dependencies
## 2.1 Backend
### TensorFlow
De neurale netwerken waaruit de GAN bestaat zijn gerealiseerd met TensorFlow 2.

### Numpy
Numpy wordt voor diverse taken gebruikt, zoals het genereren van de noise voor de generator.

### PyPlot
PyPlot wordt gebruikt om de gegenereerde afbeeldingen op te slaan. 

### UUID
De UUID-package wordt gebruikt om unieke IDs te genereren voor gegenereerde afbeeldingen. Op deze manier kan worden voorkomen dat afbeeldingen van meerdere server requests tegelijkertijd elkaar zouden overschrijven.

### Flask RESTful
Het Flask RESTful framework wordt gebruikt om server requests af te handelen. In dit geval is er slechts één endpoint (`/mnist`) van het type `POST`, waarbij de user input als Multipart Form wordt meegestuurd met de variabelnaam `value`.

### PIL (Python Image Library)
De Python Image Library, PIL, wordt gebruikt om meerdere individuele cijfer-afbeeldingen samen te voegen tot één outputafbeelding. De officiële library wordt niet meer onderhouden, maar een fork van PIL, genaamd Pillow, wordt actief onderhouden. Pillow is volledig compatibel met standaard PIL code. Daarom wordt de package in `server.py` geïmporteerd als `PIL`, maar wordt `Pillow` gebruikt in `requirements.txt`.

### Base64
Om de gegenereerde samengevoegde afbeelding als response terug te sturen naar de frontend, wordt de JPG-afbeelding omgezet naar een Base64 string representatie. Hiervoor wordt de base64 package gebruikt.

### Re
Om snel te verifiëren of de inputwaarde voldoet aan de eisen (enkel cijfers), wordt een Regular Expression gebruikt. De `re` package wordt hiervoor gebruikt in Python. Het controleren van deze input in de backend is niet van cruciaal belang, omdat foutieve inputs in de frontend al worden afgevangen, maar voor het geval dat de checks in de frontend falen of iemand de API direct aanroept met diens URL, wordt de input in de backend opnieuw geverifieerd. Een foutieve input, die bestaat uit meer dan 10 karakters of niet-getallen, levert een `400` error response op.

## 2.2 Frontend
### Flask
De frontend gebruikt het Flask framework om HTML-output te genereren. 

### Gunicorn
In de productieomgeving (Docker) wordt Flask uitgevoerd door een gunicorn server.

## 2.3 Algemeen
### Docker
Zowel de frontend als de backend kunnen in een Docker container worden uitgevoerd. Beiden maken gebruik van een standaard `python:3.9-slim` base image. Dit is een lichte image die snel kan worden gebuild en geen zware hardware vereist. Let wel, deze image is alleen geschikt voor de backend wanneer er gebruik wordt gemaakt van een voorgetraind model, te herkennen aan de `.h5` bestandsextensie. Wanneer dit model niet aanwezig is en de GAN eerst getraind moet worden (binnen een Docker container), kan er beter worden gekozen voor een specifieke TensorFlow base image.

# 3. De applicatie starten
## 3.1 Backend
De backend kan op meerdere manieren lokaal worden gerund. De simpelste manier is door het volgende commando in een command line in te voeren:
```
python3 server.py
```

De backend kan ook in Docker worden gestart. Hiervoor moet eerst een image worden gebuild met het bijgevoegde `Dockerfile` bestand. Natuurlijk moet Docker eerst zijn geïnstalleerd en actief zijn op de host computer. Het builden van de image kan met het volgende commando:
```
docker build -t gan-backend .
```
Deze image kan vervolgens gerund worden met het commando:
```
docker run -dp 8080:8080 gan-backend
```
De server luist naar verkeer op `localhost:8080`. Indien een andere lokale poort gewenst is, misschien omdat een andere applicatie al gebruikmaakt van poort 8080, kan `8080:8080` worden vervangen door `[gewenste poort]:8080`. Het tweede deel van deze port mapping moet wél altijd `8080` zijn, omdat de Flask RESTful server intern altijd naar poort 8080 zal luisteren. Meer informatie hierover is te vinden in de [Docker docs](https://docs.docker.com/config/containers/container-networking/). De naam van de image, in dit geval `gan-backend`, kan ook worden gewijzigd, zolang dezelfde naam wordt gebruikt tijdens het builden van de image en het starten van deze image.

## 3.2 Frontend
De frontend kan, net zoals de backend, zonder Docker worden gerund. Hiervoor moet Flask geïnstalleerd zijn. Vervolgens kan de applicatie op de volgende manier worden gestart:
```
export FLASK_APP=client
flask run
```
De frontend kan ook in Docker worden gestart. Dit kan als volgt:
```
docker build -t gan-frontend .
docker run -dp 8000:8080 gan-frontend
```
**LET OP:** zowel de frontend als de backend luisteren intern naar poort 8080. Daarom wordt in dit geval poort 8000 op de host gekoppeld aan de interne poort 8080. De frontend is vervolgens te gebruiken via `localhost:8000`.

## 3.3 Geïntegreerde applicatie
Om de applicatie als één geheel te starten, kan Docker compose worden gebruikt. De configuratie hiervoor is terug te vinden in het `docker-compose.yml` bestand. Deze configuratie maakt gebruik van de `Dockerfile` bestanden voor zowel de front- als backend. 

Met het volgende commando kan de gehele applicatie worden gestart:
```
docker compose up -d
```

Wanneer er wijzigingen aan de front- of backend zijn doorgevoerd en de Docker containers moeten worden geüpdatet, kan dat met de volgende commandos respectievelijk:
```
docker compose up -d --no-deps --build client
```
```
docker compose up -d --no-deps --build server
```

Om dit proces te ondersteunen, wordt de server URL als environment variable (`SERV_URL`) doorgegeven naar de frontend. Wanneer dit niet gebeurt, wordt een default server URL gebruikt, namelijk https://gan-dev.apis.niels.codes/. Dit is een live gehoste versie van de backend.

**LET OP:** De `SERV_URL` environment variable moet altijd eindigen met een `/` om errors te voorkomen. Wanneer de URL niet met `/` eindigt, zou de frontend bijvoorbeeld een POST requests sturen naar `localhost:8080mnist` in plaats van de correcte URL `localhost:8080/mnist`. 

# 4. Training
De repository bevat een voorgetraind model (`cgan_mnist.h5`). Dit model is getraind met 625 epochs. Wanneer een nieuw model moet worden getraind, kan dit met de functie `build_and_train_models()`. Voor de beste resultaten, dat wil zeggen de kortste trainingstijd, dient een NVIDIA videokaart in combinatie met de NVIDIA CUDA Deep Neural Network (cuDNN) library gebruikt te worden.


# 5. Vervolgstappen & advies
Mogelijke vervolgstappen voor de GAN-demonstrator hangen af van beschikbare hardware en tijd. Waar voor de huidige versie is besloten om enkel met een cGAN te werken wegens beperkte hardware, zou de GAN in de toekomst kunnen worden uitgebreid met bijvoorbeeld een GAN-CLS die afbeeldingen genereert op basis van hele zinnen. In dit geval zou het advies zijn om deze functionaliteit in een apart bestand te zetten, zoals dat nu ook het geval is met `cgan_mnist.py`. Op die manier zou de functionaliteit aan de API kunnen worden gekoppeld door de benodigde functies te importeren in `server.py` en door een nieuwe endpoint toe te voegen. De enige "grote" wijziging die zou moeten worden doorgevoerd aan bestaande code, is het veranderen van de base Docker image in de Dockerfile van de backend, van een Python image naar een TensorFlow image.