#!/usr/bin/env python3
"""
Manueller Test für den ECG-Server.
Voraussetzung: Server läuft bereits (python src/ecg_server.py)
"""
import os
import requests

# Relativen Pfad zur Sample-Datei in absoluten umwandeln
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_file = os.path.join(script_dir, '..', 'samples', 'ecg1.mat')
sample_file = os.path.abspath(sample_file)

print(f"Sample-Datei: {sample_file}")
print(f"Datei existiert: {os.path.exists(sample_file)}")

# Health Check
response = requests.get('http://localhost:5000/health' )
print(f"Health: {response.json()}")

# Analyse durchführen
response = requests.post('http://localhost:5000/analyze', json={
    'file_path': sample_file,
    'intervals': [[1050, 1700]]
} )
print(f"Analyse Response: {response.json()}")

# Ergebnisse abrufen
response = requests.get('http://localhost:5000/results', params={'file': 'ecg1.mat'} )
result = response.json()
print(f"Results Response: {result}")

# Sicher auf count zugreifen
if 'count' in result:
    print(f"Ergebnisse: {result['count']} Einträge")
elif 'error' in result:
    print(f"Fehler: {result['error']}")
