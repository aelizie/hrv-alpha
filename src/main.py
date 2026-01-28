import sys
import logging
import numpy as np
import json
import os


from ecg_hrv_analysis import ECGAnalyzer
from ecg_visualizer import plot_interval_results, plot_ecg_overview
from ecg_database import ECGDatabase, compute_config_hash


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout
)

if __name__ == "__main__":
    # 1. Pfade definieren
    CONFIG_FILE = '../config/conf.ini'
    DATA_FILE = '../samples/ecg1.mat' # Anpassbar!
    TEST_INTERVALS = [
        (0, 350), (350, 700), (700, 1050), (1050, 1700), (1700, 2600), (2600, 3500), (3500, 4400), (4400, 5300)
    ]

    # 2. Datenbank initialisieren
    os.makedirs('../data', exist_ok=True)
    db = ECGDatabase('../data/ecg_results.db')

    try:
        # 4. Analyzer instanziieren (Läd die Config) sowie Hash für DB
        analyzer = ECGAnalyzer(CONFIG_FILE)
        config = {
            'sampling_rate_hz': analyzer.samplingRate_hz,
            'peak_threshold_ratio': analyzer.peakThreshold_ratio,
            'refractory_period_sec': analyzer.refractoryPeriod_sec
        }
        config_hash = compute_config_hash(config)

        # 5. Analyse starten
        results = analyzer.analyze_file(DATA_FILE, TEST_INTERVALS)

        if results:
            # 6. Konsolenausgabe
            print("\n\n--- Strukturierte Ausgabe (JSON) ---")
            print(results)

            signal = analyzer.loadEcgSignalFromMatFile('../samples/ecg1.mat')

            # Alle erfolgreichen Intervalle visualisieren
            saved_paths = plot_interval_results(
                results,
                signal,
                analyzer.samplingRate_hz,
                output_dir='../plots'
            )

            plot_ecg_overview(
                signal,
                analyzer.samplingRate_hz,
                output_path='../plots/ecg_overview.png',
                title='EKG Signal Übersicht'
            )

            saved_count = db.save_analysis_results(results, config_hash=config_hash)
            print(f"{saved_count} Datensätze gespeichert.")

    except Exception as e:
        logging.critical(f"System Failure: {e}")