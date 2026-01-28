"""
ECG Database Modul
Autor: EmSt
Datum: 28.01.2026

Dieses Modul stellt Funktionen zur Speicherung und zum Abruf von
EKG-Analyseergebnissen in einer lokalen SQLite-Datenbank bereit.
Es implementiert auch einen einfachen Audit Trail.
Es wurde großteils automatisiert mit Gemini erstellt
"""

import os
import sqlite3
import json
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger("ECG_Database")


class ECGDatabase:
    """
    Eine Klasse zur Verwaltung der lokalen SQLite-Datenbank für EKG-Analyseergebnisse.
    """

    def __init__(self, db_path="ecg_results.db"):
        """
        Initialisiert die Datenbankverbindung und erstellt die Tabellen, falls nicht vorhanden.

        :param db_path: Der Pfad zur SQLite-Datenbankdatei.
        :type db_path: str
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"ECGDatabase initialisiert mit Datenbank: {db_path}")

    def _init_database(self):
        """
        Erstellt die Datenbanktabellen, falls sie nicht existieren.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabelle für Analyseergebnisse
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_file_path TEXT,
                interval_index INTEGER NOT NULL,
                interval_start_sec REAL NOT NULL,
                interval_end_sec REAL NOT NULL,
                status TEXT NOT NULL,
                hr_mean REAL,
                hr_min REAL,
                hr_max REAL,
                sdnn REAL,
                sdann REAL,
                nn50 INTEGER,
                config_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Tabelle für Audit Trail
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.debug("Datenbanktabellen initialisiert.")

    def _get_connection(self):
        """
        Erstellt eine neue Datenbankverbindung.

        :return: Eine SQLite-Verbindung.
        :rtype: sqlite3.Connection
        """
        return sqlite3.connect(self.db_path)

    def _log_audit_event(self, event_type, details, user="SYSTEM"):
        """
        Fügt einen Eintrag zum Audit Trail hinzu.

        :param event_type: Der Typ des Ereignisses (z.B. 'RECORD_CREATED').
        :type event_type: str
        :param details: Ein Dictionary mit Details zum Ereignis.
        :type details: dict
        :param user: Der verantwortliche Benutzer.
        :type user: str
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        details_json = json.dumps(details, default=str)

        cursor.execute('''
            INSERT INTO audit_trail (timestamp, event_type, user, details)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, event_type, user, details_json))

        conn.commit()
        conn.close()
        logger.debug(f"Audit Event: {event_type} - {details}")

    def save_analysis_results(self, analysis_results, config_hash=None, user="SYSTEM"):
        """
        Speichert die Analyseergebnisse in der Datenbank.

        :param analysis_results: Das Ergebnis-Dictionary von analyze_file().
        :type analysis_results: dict
        :param config_hash: Ein Hash der verwendeten Konfigurationsparameter.
        :type config_hash: str | None
        :param user: Der verantwortliche Benutzer.
        :type user: str
        :return: Die Anzahl der gespeicherten Datensätze.
        :rtype: int
        """
        if analysis_results is None:
            logger.warning("Keine Analyseergebnisse zum Speichern vorhanden.")
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        metadata = analysis_results['metadata']
        source_file = metadata['source_file']
        source_file_path = metadata.get('source_file_path', '')

        saved_count = 0

        for interval in analysis_results['intervals']:
            metrics = interval.get('metrics') or {}

            # NaN-Werte in None umwandeln für SQLite
            sdann = metrics.get('SDANN')
            if sdann is not None and (sdann != sdann):  # NaN check
                sdann = None

            cursor.execute('''
                INSERT INTO analysis_results (
                    timestamp, source_file, source_file_path,
                    interval_index, interval_start_sec, interval_end_sec,
                    status, hr_mean, hr_min, hr_max, sdnn, sdann, nn50, config_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                source_file,
                source_file_path,
                interval['interval_index'],
                interval['start_sec'],
                interval['end_sec'],
                interval['status'],
                metrics.get('HR_Mean'),
                metrics.get('HR_Min'),
                metrics.get('HR_Max'),
                metrics.get('SDNN'),
                sdann,
                metrics.get('NN50'),
                config_hash
            ))
            saved_count += 1

        conn.commit()
        conn.close()

        # Audit Trail Eintrag
        self._log_audit_event(
            event_type="RECORDS_CREATED",
            details={
                "source_file": source_file,
                "interval_count": saved_count,
                "timestamp": timestamp,
                "config_hash": config_hash
            },
            user=user
        )

        logger.info(f"{saved_count} Analyseergebnisse in Datenbank gespeichert.")
        return saved_count

    def get_results_by_file(self, source_file):
        """
        Ruft alle Analyseergebnisse für eine bestimmte Quelldatei ab.

        :param source_file: Der Name der Quelldatei.
        :type source_file: str
        :return: Eine Liste von Dictionaries mit den Ergebnissen.
        :rtype: list[dict]
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM analysis_results
            WHERE source_file = ?
            ORDER BY timestamp DESC, interval_index ASC
        ''', (source_file,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_all_results(self, limit=100):
        """
        Ruft alle Analyseergebnisse ab (mit Limit).

        :param limit: Maximale Anzahl der zurückgegebenen Ergebnisse.
        :type limit: int
        :return: Eine Liste von Dictionaries mit den Ergebnissen.
        :rtype: list[dict]
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM analysis_results
            ORDER BY timestamp DESC, interval_index ASC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_audit_trail(self, limit=100):
        """
        Ruft die Audit Trail Einträge ab.

        :param limit: Maximale Anzahl der zurückgegebenen Einträge.
        :type limit: int
        :return: Eine Liste von Dictionaries mit den Audit-Einträgen.
        :rtype: list[dict]
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM audit_trail
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def export_to_json(self, output_path):
        """
        Exportiert alle Analyseergebnisse in eine JSON-Datei.

        :param output_path: Der Pfad zur Ausgabedatei.
        :type output_path: str
        :return: Die Anzahl der exportierten Datensätze.
        :rtype: int
        """
        results = self.get_all_results(limit=10000)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        self._log_audit_event(
            event_type="DATA_EXPORTED",
            details={
                "output_path": output_path,
                "record_count": len(results)
            }
        )

        logger.info(f"{len(results)} Datensätze nach {output_path} exportiert.")
        return len(results)


def compute_config_hash(config_dict):
    """
    Berechnet einen Hash aus den Konfigurationsparametern.

    :param config_dict: Ein Dictionary mit den Konfigurationsparametern.
    :type config_dict: dict
    :return: Ein SHA-256 Hash als Hex-String.
    :rtype: str
    """
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
