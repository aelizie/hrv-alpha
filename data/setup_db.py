#!/usr/bin/env python3
"""
Setup-Skript zur Initialisierung der Datenbank.
Ausführen: python data/setup_db.py (vom Projektroot aus)
"""
import os
import sys

# Absoluten Pfad zum src-Verzeichnis berechnen, zur Sicherheit
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

from ecg_database import ECGDatabase

# Datenbank im gleichen Verzeichnis wie dieses Skript erstellen
db_path = os.path.join(script_dir, 'ecg_results.db')

db = ECGDatabase(db_path)
print(f"Datenbank initialisiert: {db_path}")
