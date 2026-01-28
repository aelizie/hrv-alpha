"""
ECG Mock Server Modul
Autor: EmSt
Datum: 28.01.2026

Dieses Modul stellt einen einfachen lokalen HTTP-Server bereit,
der Anfragen zur EKG-Analyse entgegennimmt und die Ergebnisse
zurückgibt. Es dient als Mock-Server für die Simulation einer
Client-Server-Architektur. Es wurde großteils automatisiert mit
Gemini erstellt..
"""

import os
import sys
import json
import logging
from datetime import datetime

# Flask für den HTTP-Server
from flask import Flask, request, jsonify

# Eigene Module
from ecg_hrv_analysis import ECGAnalyzer
from ecg_database import ECGDatabase, compute_config_hash

logger = logging.getLogger("ECG_Server")

# Flask App initialisieren
app = Flask(__name__)

# Globale Variablen für Analyzer und Datenbank
analyzer = None
database = None
audit_log_path = None


def init_server(config_path, db_path, audit_path=None):
    """
    Initialisiert den Server mit den notwendigen Komponenten.

    :param config_path: Pfad zur Konfigurationsdatei.
    :type config_path: str
    :param db_path: Pfad zur SQLite-Datenbank.
    :type db_path: str
    :param audit_path: Pfad zur Audit-Log-Datei (optional).
    :type audit_path: str | None
    """
    global analyzer, database, audit_log_path

    analyzer = ECGAnalyzer(config_path)
    database = ECGDatabase(db_path)
    audit_log_path = audit_path

    logger.info(f"Server initialisiert mit Config: {config_path}, DB: {db_path}")


def log_to_audit_file(event_type, details, user="SYSTEM"):
    """
    Schreibt einen Audit-Eintrag in die Log-Datei (append-only).

    :param event_type: Der Typ des Ereignisses.
    :type event_type: str
    :param details: Details zum Ereignis.
    :type details: dict
    :param user: Der verantwortliche Benutzer.
    :type user: str
    """
    if audit_log_path is None:
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "user": user,
        "details": details
    }

    with open(audit_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=str) + '\n')


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health-Check-Endpunkt.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpunkt zur Analyse einer EKG-Datei.

    Erwartet JSON-Body mit:
    - file_path: Pfad zur EKG-Datei
    - intervals: Liste von [start, end] Tupeln (optional)

    Gibt die Analyseergebnisse als JSON zurück.
    """
    if analyzer is None or database is None:
        return jsonify({"error": "Server nicht initialisiert"}), 500

    try:
        data = request.get_json()

        if not data or 'file_path' not in data:
            return jsonify({"error": "file_path ist erforderlich"}), 400

        file_path = data['file_path']
        intervals = data.get('intervals', [(0, 60)])  # Default: erste Minute

        # Validierung
        if not os.path.exists(file_path):
            log_to_audit_file("ANALYSIS_REQUEST_FAILED", {
                "file_path": file_path,
                "reason": "File not found"
            })
            return jsonify({"error": f"Datei nicht gefunden: {file_path}"}), 404

        # Intervals in Tupel umwandeln
        intervals = [tuple(i) for i in intervals]

        # Audit Log: Anfrage erhalten
        log_to_audit_file("ANALYSIS_REQUEST_RECEIVED", {
            "file_path": file_path,
            "intervals": intervals
        })

        # Analyse durchführen
        results = analyzer.analyze_file(file_path, intervals)

        if results is None:
            log_to_audit_file("ANALYSIS_FAILED", {
                "file_path": file_path,
                "reason": "Analysis returned None"
            })
            return jsonify({"error": "Analyse fehlgeschlagen"}), 500

        # Konfigurationshash berechnen
        config = {
            'sampling_rate_hz': analyzer.samplingRate_hz,
            'peak_threshold_ratio': analyzer.peakThreshold_ratio,
            'refractory_period_sec': analyzer.refractoryPeriod_sec
        }
        config_hash = compute_config_hash(config)

        # In Datenbank speichern
        saved_count = database.save_analysis_results(results, config_hash=config_hash)

        # Audit Log: Analyse abgeschlossen
        log_to_audit_file("ANALYSIS_COMPLETED", {
            "file_path": file_path,
            "intervals_analyzed": len(results['intervals']),
            "records_saved": saved_count
        })

        # NaN-Werte und numpy-Typen für JSON-Serialisierung behandeln
        import numpy as np
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(i) for i in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, float) and (obj != obj):  # NaN check
                return None
            return obj

        return jsonify({
            "status": "success",
            "results": clean_for_json(results),
            "records_saved": saved_count
        })

    except Exception as e:
        logger.error(f"Fehler bei der Analyse: {e}")
        log_to_audit_file("ANALYSIS_ERROR", {
            "error": str(e)
        })
        return jsonify({"error": str(e)}), 500


@app.route('/results', methods=['GET'])
def get_results():
    """
    Endpunkt zum Abrufen gespeicherter Ergebnisse.

    Query-Parameter:
    - file: Name der Quelldatei (optional)
    - limit: Maximale Anzahl der Ergebnisse (default: 100)
    """
    if database is None:
        return jsonify({"error": "Server nicht initialisiert"}), 500

    try:
        source_file = request.args.get('file')
        limit = int(request.args.get('limit', 100))

        if source_file:
            results = database.get_results_by_file(source_file)
        else:
            results = database.get_all_results(limit=limit)

        # Konvertiere bytes zu strings falls vorhanden
        def clean_results(obj):
            if isinstance(obj, dict):
                return {k: clean_results(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_results(i) for i in obj]
            elif isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except UnicodeDecodeError:
                    return obj.decode('latin-1')  # Fallback
            return obj

        results = clean_results(results)

        log_to_audit_file("RESULTS_RETRIEVED", {
            "source_file": source_file,
            "count": len(results)
        })

        return jsonify({
            "status": "success",
            "count": len(results),
            "results": results
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Ergebnisse: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/audit', methods=['GET'])
def get_audit():
    """
    Endpunkt zum Abrufen des Audit Trails.

    Query-Parameter:
    - limit: Maximale Anzahl der Einträge (default: 100)
    """
    if database is None:
        return jsonify({"error": "Server nicht initialisiert"}), 500

    try:
        limit = int(request.args.get('limit', 100))
        audit = database.get_audit_trail(limit=limit)

        return jsonify({
            "status": "success",
            "count": len(audit),
            "audit_trail": audit
        })

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des Audit Trails: {e}")
        return jsonify({"error": str(e)}), 500


def run_server(host='127.0.0.1', port=5000, debug=False):
    """
    Startet den Flask-Server.

    :param host: Die Host-Adresse.
    :type host: str
    :param port: Der Port.
    :type port: int
    :param debug: Debug-Modus aktivieren.
    :type debug: bool
    """
    logger.info(f"Starte Server auf {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ECG Analysis Mock Server')
    parser.add_argument('--config', default='../config/conf.ini', help='Pfad zur Konfigurationsdatei')
    parser.add_argument('--db', default='../data/ecg_results.db', help='Pfad zur Datenbank')
    parser.add_argument('--audit', default='../data/audit.log', help='Pfad zur Audit-Log-Datei')
    parser.add_argument('--host', default='127.0.0.1', help='Host-Adresse')
    parser.add_argument('--port', type=int, default=5000, help='Port')
    parser.add_argument('--debug', action='store_true', help='Debug-Modus')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout
    )

    # Server initialisieren
    init_server(args.config, args.db, args.audit)

    # Server starten
    run_server(host=args.host, port=args.port, debug=args.debug)
