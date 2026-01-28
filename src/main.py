import sys
import logging

from ecg_hrv_analysis import ECGAnalyzer

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

    try:
        # 4. Analyzer instanziieren (Läd die Config)
        analyzer = ECGAnalyzer(CONFIG_FILE)

        # 5. Analyse starten
        analyzer.analyze_file(DATA_FILE, TEST_INTERVALS)

    except Exception as e:
        logging.critical(f"System Failure: {e}")