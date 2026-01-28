# EKG HRV Analyse Tool

## Features

- Laden von EKG-Signalen aus `.mat`-Dateien.
- Konfigurierbare R-Zacken-Erkennung.
- Berechnung von HRV-Metriken: HR (Mittel, Min, Max), SDNN, NN50 und SDANN.
- Analyse über mehrere, benutzerdefinierte Zeitintervalle.
- Flexible Konfiguration der Analyseparameter über eine `conf.ini`-Datei.

## Installation

Bitte sicherstellen, dass Python 3 installiert ist. Vor der Nutzung sind die Dependencies zu installieren:

``pip install -r requirements.txt
``


## Verwendung

1.  Passe die Analyseparameter in der Datei `config/conf.ini` nach Bedarf an.
2.  Die `main.py` öffnen die Variable `DATA_FILE` anpassen, um auf die EKG-Datendatei (`.mat`) zu verweisen.
3.  Bei Bedarf die zu analysierenden Zeitfenster in der `TEST_INTERVALS`-Liste in `main.py` anpassen.
4.  Dann das Analyse-Skript ausführen:

`` python main.py
``

Die Ergebnisse der Analyse werden direkt in der Konsole ausgegeben.

## Unit und System Tests

Zum Ausführen der Tests, einfach folgenden Befehl ausführen (oder in einer IDE der Wahl starten)

`` python -m pytest -q
``