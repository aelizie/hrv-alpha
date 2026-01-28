"""
RAMS Übung 2 - Teil 1: EKG Analyse Tool
Autor: EmSt
Datum: 03.01.2026
"""

import os
import logging
import configparser
import numpy as np
import scipy.io

# Logger konfigurieren (aber keine Handler hinzufügen, das macht das aufrufende Skript)
logger = logging.getLogger("HRV_Analyzer")


class ECGAnalyzer:
    def __init__(self, config_path):
        """
        Initialisiert den Analyzer und lädt die Config.
        """
        self._load_config(config_path)
        logger.info(f"ECGAnalyzer initialisiert mit Config: {config_path}")

    def _load_config(self, path):
        """
        Lädt und parst eine Config Datei, die im INI-Format bereitgestellt wird. Abschnitte und
        Schlüssel aus der Datei werden auf ein flaches Dictionary abgebildet oder
        spezifischen Attributes zugewiesen. Parsing Fehler oder Fälle, in denen die Datei fehlt
        werden mit entsprechenden Exceptions behandelt.

        :param path: Der Dateipfad zu der zu ladenden Config.
        :type path: str
        :return: Ein flaches Dictionary, das die Parameter enthält.
        :rtype: dict
        :raises FileNotFoundError: Wenn der angegebene Dateipfad nicht gefunden werden kann.
        :raises ValueError: Wenn die Datei unvollständig ist oder Fehler enthält.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        parser = configparser.ConfigParser()
        parser.read(path)

        # Wir holen die COnfig damit wir wieder 'camelCase_unit' auf class Ebene direkt zugreifen können.
        cfg = {}
        try:
            # Helper lambda für cleanes Casting
            get_fl = lambda sec, key: float(parser.get(sec, key))

            # Mapping der INI Sections auf class ebene
            self.samplingRate_hz = get_fl('AnalysisParameters', 'samplingRate_hz')

            self.peakThreshold_ratio = get_fl('PeakDetection', 'peakThreshold_ratio')
            self.refractoryPeriod_sec = get_fl('PeakDetection', 'refractoryPeriod_sec')
            self.searchWindow_sec = get_fl('PeakDetection', 'searchWindow_sec')

            self.nn50Threshold_ms = get_fl('HRVMetrics', 'nn50Threshold_ms')
            self.sdannWindow_sec = get_fl('HRVMetrics', 'sdannWindow_sec')
            self.minValidHr_bpm = get_fl('HRVMetrics', 'minValidHr_bpm')
            self.maxValidHr_bpm = get_fl('HRVMetrics', 'maxValidHr_bpm')

            self.outlierLowCut_perc = get_fl('Preprocessing', 'outlierLowCut_perc')
            self.outlierHighCut_perc = get_fl('Preprocessing', 'outlierHighCut_perc')

        except Exception as e:
            logger.error(f"Fehler beim Parsen der Config: {e}")
            raise ValueError("Config Datei ist fehlerhaft oder unvollständig")

        return cfg

    def loadEcgSignalFromMatFile(self, file_path):
        """
        Lädt ein EKG-Signal aus einer .mat Datei
        Es sucht und extrahiert die erste gültige Datenvariable, die nicht Metadaten entspricht
        (z.B Variable, deren Schlüssel nicht mit '__' beginnt).
        Das gefundene Signal wird als flaches Numpy-Array zurückgegeben.

        :param file_path: Der vollständige Pfad zur .mat Datei.
        :type file_path: str
        :return: Das extrahierte Signal als Numpy-Array, falls erfolgreich; andernfalls None.
        :rtype: Optional[np.ndarray]
        """
        if not os.path.exists(file_path):
            logger.error(f"Datei nicht gefunden: {file_path}")
            return None

        try:
            mat_data = scipy.io.loadmat(file_path)
            # Suche nach der Variable, die keine Metadaten ('__') ist
            for key, val in mat_data.items():
                if not key.startswith('__'):
                    # Konvertierung zu flachem np Array für einfache Verarbeitung
                    signal_mvA = np.array(val).flatten()
                    logger.info(f"Signal geladen aus Variable '{key}'. Länge: {len(signal_mvA)} Samples.")
                    return signal_mvA

            logger.error("Keine gültige Datenvariable in der .mat Datei gefunden.")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Laden der Datei: {e}")
            return None

    def clipOutliersFromSignal(self, signal_mvA):
        """
        CLippt Ausreißer im gegebenen Signal basierend auf den bereitgestellten
        Perzentil Schwellenwerten. Werte im Signal, die unter dem unteren Perzentil Schwellenwert
        liegen, werden auf den unteren Schwellenwert gesetzt, und Werte, die über dem oberen
        Schwellenwert liegen, werden auf den oberen Schwellenwert begrenzt.

        :param signal_mvA: Eigangssignal, das Outlier Clipping benötigt.
        :type signal_mvA: numpy.ndarray
        :return: Das Signal, bei dem Ausreißer auf die definierten Schwellenwerte begrenzt wurden.
        :rtype: numpy.ndarray
        """
        if len(signal_mvA) == 0:
            return signal_mvA

        lowestValue_mv = np.percentile(signal_mvA, self.outlierLowCut_perc) # Alles unter diesem Wert wird auf diesen Wert geclippt.
        highestValue_mv = np.percentile(signal_mvA, self.outlierHighCut_perc) # Alles über diesem Wert wird auf diesen Wert geclippt.

        logger.debug(f"Outlier Intervall ist [{lowestValue_mv:.2f}, {highestValue_mv:.2f}]")
        # Signale werden geclipped, daher alles was kleiner als lower oder mehr als upper ist, wird auf diese Werte reduziert.
        return np.clip(signal_mvA, lowestValue_mv, highestValue_mv)

    def detectPeaksFromSignal(self, signal_mvA):
        """
        Detektiert R-Zacken im gegebenen EKG-Signal.

        Diese Funktion identifiziert die Indizes von R-Zacken in einem EKG-Signal mithilfe einer schwellenwertbasierten
        Algorithmus. Zuerst wird ein Schwellenwert basierend auf der maximalen Amplitude des Signals bestimmt.
        Alle Samples über dem Schwellenwert werden als potenzielle R-Zacken betrachtet. Um eine solide
        Peak-Detektion sicherzustellen, wird eine Refraktärzeit angewendet, die die Erfassung mehrerer Peaks innerhalb
        eines minimalen Zeitraums verhindert. Die genaue Position der R-Zacken wird durch die Suche nach lokaler
        Maximum innerhalb eines definierten Suchfensters verfeinert

        :param signal_mvA: Das EKG-Signal in Millivolt als 1D-Array.
        :type signal_mvA: numpy.ndarray
        :return: Ein 1D Array, das die Indizes der detektierten R-Zacken im Eingangssignal enthält.
        :rtype: numpy.ndarray
        """
        if len(signal_mvA) == 0:
            return np.array([])

        # Bestimmung des Schwellwerts, basierend auf der höchsten Amplitude, davon ausgehend eine Threshold
        # an der die R Zacken simpelst bestimmt werden können.
        highestAmplitudeInSignal_mv = np.percentile(signal_mvA, 99)
        thresholdForPeakDetection_mv = highestAmplitudeInSignal_mv * self.peakThreshold_ratio

        # Finden aller Punkte über der Schwelle
        aboveThreshold_idxA = np.where(signal_mvA > thresholdForPeakDetection_mv)[0]

        if len(aboveThreshold_idxA) == 0:
            logger.warning("Keine Peaks über Threshold gefunden. Fehlerhaftes Signal?")
            return np.array([])

        peaks_idxA = []
        lastDetectedPeak_idx = -1
        # Definition der Distanz bzw Menge an Indizes/Samples für die Refraktärzeit bzw des Suchfensters,
        # basierend auf Umwandlung von sampling rate und Zeit Konfiguration
        minimumDistance_samples = int(self.refractoryPeriod_sec * self.samplingRate_hz)
        searchWindow_samples = int(self.searchWindow_sec * self.samplingRate_hz)

        # Iteration und Refraktärzeit-Filterung
        for idx in aboveThreshold_idxA:
            if (lastDetectedPeak_idx == -1 or
                (idx - lastDetectedPeak_idx) > minimumDistance_samples
            ):
                # Neuer potenzieller QRS-Komplex gefunden
                # Um den genauen Peak zu finden, suchen wir das lokale Maximum
                # im Bereich [idx, idx + Fenster], um nicht auf der Seite zu landen.
                searchEnd_idx = min(idx + searchWindow_samples, len(signal_mvA))

                # Lokales Suchenster Definieren
                localWindow_mvA = signal_mvA[idx:searchEnd_idx]

                if len(localWindow_mvA) > 0:
                    # Maximum im Fenster finden, und an peaks array anhängen
                    highestLocalPeak_idx = np.argmax(localWindow_mvA)
                    peak_idx = idx + highestLocalPeak_idx
                    peaks_idxA.append(peak_idx)
                    lastDetectedPeak_idx = peak_idx

        logger.info(f"{len(peaks_idxA)} R-Zacken detektiert.")
        return np.unique(peaks_idxA)  # Indizes der R-Zacken

    def calculateHrvMetrics(self, nnTimeIntervals_msA, duration_sec):
        """
        Berechnet Metriken der HRV aus einer Sequenz von NN Intervallen und der Messdauer.

        Diese Funktion berechnet verschiedene HRV-Metriken wie die mittlere, minimale und
        maximale Herzfrequenz (HR), die Standardabweichung der NN-Intervalle (SDNN), die
        Standardabweichung des Durchschnitts der NN-Intervalle über einen spezifische Zeitraum (SDANN)
        und den NN50-Wert, der die Anzahl aufeinanderfolgender NN-Intervalle darstellt,
        die sich um mehr als 50ms unterscheiden.
        Sie verwendet Gültigkeitsgrenzen für Herzfrequenzen, um potenziell unphysiologische HR-Werte
        zu identifizieren.

        :param nnTimeIntervals_msA: Liste von NN-Intervallen in Millisekunden.
        :type nnTimeIntervals_msA: list[float]
        :param duration_sec: Gesamtdauer der Messung in Sekunden.
        :type duration_sec: float

        :return: Dictionary, das die HRV-Metriken enthält:
            - 'HR_Mean': Mittlere Herzfrequenz in Schlägen pro Minute (bpm).
            - 'HR_Min': Minimale Herzfrequenz in bpm.
            - 'HR_Max': Maximale Herzfrequenz in bpm.
            - 'SDNN': Standardabweichung der NN-Intervalle, in Millisekunden.
            - 'SDANN': Standardabweichung des Durchschnitts der NN-Intervalle in einem Zeitraum, in Millisekunden.
            - 'NN50': Anzahl aufeinanderfolgender NN-Intervalle, die sich um mehr als 50ms unterscheiden.
        :rtype: dict[str, float]

        """
        if len(nnTimeIntervals_msA) < 1:
            return None

        hrValues_bpmA = 60000.0 / nnTimeIntervals_msA # Instantes HR pro Schlag
        hr_mean = 60000.0 / np.mean(nnTimeIntervals_msA) # HR als Mittel von NN
        hr_min = 60000.0 / np.max(nnTimeIntervals_msA)  # längstes NN -> niedrigste NN
        hr_max = 60000.0 / np.min(nnTimeIntervals_msA)  # kürzestes NN -> höchste NN

        # Fürs Debugging ob die Messung generell schlecht ist, da am Limit
        low = 60000.0 / self.maxValidHr_bpm
        high = 60000.0 / self.minValidHr_bpm

        logger.debug(f"NN < low: {np.sum(nnTimeIntervals_msA < low)}, NN > high: {np.sum(nnTimeIntervals_msA > high)}")

        # Artefakt-Check (Plausibilität)
        if hr_max > self.maxValidHr_bpm or hr_min < self.minValidHr_bpm:
            logger.warning(f"Unphysiologische HR: {hr_min:.0f}-{hr_max:.0f} bpm")

        # --- SDNN ---
        # Standardabweichung aller NN Intervalle (ddof 1 für Stichprobe)
        sdnn = np.std(nnTimeIntervals_msA, ddof=1)

        # --- NN50 Count ---
        # Differenzen zwischen benachbarten Intervallen
        diffs = np.abs(np.diff(nnTimeIntervals_msA))
        # Menge der Intervalle Päärchen die eine Differenz höher al 50ms haben
        nn50 = np.sum(diffs > self.nn50Threshold_ms)

        # --- SDANN ---
        # Benötigt Segmente in einer bestimmten Mindestlänge (Standard 5 Minuten)
        sdann = np.nan
        segmentWindow_ms = self.sdannWindow_sec * 1000.0

        # Erste optimistische Vorarbprüfung
        if duration_sec >= self.sdannWindow_sec:

            # Erstellen von "Timestamps" in ms, vom Start des Intervalls, um genauer segmentieren zu können
            timestampsBasedOnCumulativeTime_msA = np.cumsum(nnTimeIntervals_msA)

            # Nochmaliges Überprüfen ob tatsächlich ausreichend Zeitdaten vorhanden sind
            if len(timestampsBasedOnCumulativeTime_msA) > 0 and timestampsBasedOnCumulativeTime_msA[-1] >= segmentWindow_ms:
                # Wieviele volle Zeitsegmente sind verfügbar (truncating vom Rest)
                fullSegments_num = int(timestampsBasedOnCumulativeTime_msA[-1] / segmentWindow_ms)
                segmentMeans_ms = []

                for i in range(fullSegments_num):
                    # Definiere die Timestamps für das Segment
                    segmentWindowBegin_ms, segmentWindowEnd_ms = i * segmentWindow_ms, (i + 1) * segmentWindow_ms
                    # Erstelle eine numpy boolean index mask um alle Werte die in das Segment fallen zu erhalten
                    maskOfRelevantNN_boolA = (timestampsBasedOnCumulativeTime_msA >= segmentWindowBegin_ms) & (timestampsBasedOnCumulativeTime_msA < segmentWindowEnd_ms)
                    if np.any(maskOfRelevantNN_boolA):
                        #Berechne nun den Durchschnitt der NNs im maskierten Zeitraum (also dem aktuellen Segment)
                        segmentMeans_ms.append(
                            np.mean(
                                nnTimeIntervals_msA[maskOfRelevantNN_boolA]
                            )
                        )

                if len(segmentMeans_ms) > 1:
                    sdann = np.std(segmentMeans_ms, ddof=1)
                elif len(segmentMeans_ms) == 1:
                    sdann = 0.0 # Keine Abweichung bei nur einem Segment
        else:
            logger.info("Intervall < 5 Minuten. SDANN kann nicht berechnet werden.")

        return {
            'HR_Mean': hr_mean, 'HR_Min': hr_min, 'HR_Max': hr_max,
            'SDNN': sdnn, 'SDANN': sdann, 'NN50': nn50
        }

    def analyze_file(self, file_path, analysisIntervals_secT):
        """
        Analysiert eine EKG-Signaldatei, um Metriken der HRV über pezifizierte
        Intervalle zu berechnen. Die Funktion lädt das EKG Signal, validiert
        und verarbeitet die festgelegten Intervalle, extrahiert R-Zacken und berechnet,
        sofern durchführbar, HRV-Metriken. Die Analyseergebnisse werden ausgegeben und
        beinhalten optional zusammenfassende Statistiken über alle validen Intervalle,
        wenn mehr als 7 Intervalle verarbeitet wurden

        :param file_path: Pfad zur Datei, die das EKG-Signal enthält.
        :type file_path: str
        :param analysisIntervals_secT: Liste von Tupeln, die Start- und Endzeiten für
            die zu analysierenden Intervalle (in Sekunden) definieren.
        :type analysisIntervals_secT: list[tuple[float, float]]
        :return: Ein Dictionary mit den Analyseergebnissen, Metadaten und dem Rohsignal.
                 Gibt None zurück, wenn das Signal nicht geladen werden konnte.
        :rtype: dict | None
        """
        logger.info(f"Starte Analyse: {file_path}")

        rawSignal_mvA = self.loadEcgSignalFromMatFile(file_path)
        if rawSignal_mvA is None:
            logger.error("Kein Signal erhalten. Breche Analyse ab.")
            return
        logger.info(f"Signal ist {len(rawSignal_mvA)} Samples lang.")
        totalSignalDuration_sec = len(rawSignal_mvA) / self.samplingRate_hz

        # Ergebnisstruktur initialisieren
        results = {
            'metadata': {
                'source_file': os.path.basename(file_path),
                'source_file_path': file_path,
                'total_signal_duration_sec': totalSignalDuration_sec,
                'sampling_rate_hz': self.samplingRate_hz,
            },
            'intervals': [],
            'summary': None
        }

        metrics_dictA = []

        print(f"\n{'=' * 60}")
        print(f"BERICHT: HRV Analyse für {os.path.basename(file_path)}")
        print(f"{'=' * 60}")

        for idx, (start_sec, end_sec) in enumerate(analysisIntervals_secT):
            print(f"\n--- Intervall {idx + 1}: {start_sec}s - {end_sec}s ---")

            interval_result = {
                'interval_index': idx + 1,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'status': 'skipped',
                'metrics': None,
                'r_peaks_indices': None,
                'nn_intervals_ms': None
            }

            # Validierung der Intervallgrenzen
            if start_sec < 0 or end_sec > totalSignalDuration_sec or start_sec >= end_sec:
                logger.warning(f"Ungültiges Intervall [{start_sec}, {end_sec}]. Abbruch.")
                interval_result['status'] = 'invalid_bounds'
                continue

            # Slicing auf dem RAW Signal
            start_idx = int(start_sec * self.samplingRate_hz)
            end_idx = int(end_sec * self.samplingRate_hz)

            rawSignalInterval_mvA = rawSignal_mvA[start_idx:end_idx]

            # Lokales Preprocessing, daher nur dieses spezifische Stück wird bereinigt.
            intervalSignal_mvA = self.clipOutliersFromSignal(rawSignalInterval_mvA)

            # Check, ob beim Preprocessing was schief ging (z.B leeres Array zurück)
            if intervalSignal_mvA is None or len(intervalSignal_mvA) == 0:
                logger.warning("Preprocessing hat leeres Signal geliefert. Abbruch.")
                interval_result['status'] = 'preprocessing_failed'
                results['intervals'].append(interval_result)
                continue

            # 5. Detektion auf dem bereinigten Slice
            rPeaksInInterval_idxA = self.detectPeaksFromSignal(intervalSignal_mvA)

            if len(rPeaksInInterval_idxA) < 2:
                logger.warning("Zu wenige Peaks in diesem Intervall. Abbruch.")
                interval_result['status'] = 'too_few_peaks'
                results['intervals'].append(interval_result)
                continue

            # Umrechnung von Indizes zu Zeit (Sekunden) - relativ zum Intervall-Start
            rPeaksInInterval_secA = rPeaksInInterval_idxA / self.samplingRate_hz

            # HRV Metrics, angefangen mit dem NN, als Differenz zwischen den Peak in Millisekunden
            nnTimeIntervals_msA = np.diff(rPeaksInInterval_secA) * 1000.0

            metrics_dict = self.calculateHrvMetrics(nnTimeIntervals_msA, end_sec - start_sec)

            if metrics_dict:
                metrics_dictA.append(metrics_dict)
                interval_result['status'] = 'success'
                interval_result['metrics'] = metrics_dict
                interval_result['r_peaks_indices'] = rPeaksInInterval_idxA.tolist()
                interval_result['nn_intervals_ms'] = nnTimeIntervals_msA.tolist()

                sdann_str = f"{metrics_dict['SDANN']:.2f} ms" if not np.isnan(metrics_dict['SDANN']) else "N/A"
                print(f"  HR: {metrics_dict['HR_Mean']:.1f} bpm (Min/Max: {metrics_dict['HR_Min']:.0f}/{metrics_dict['HR_Max']:.0f})")
                print(f"  SDNN: {metrics_dict['SDNN']:.2f} ms | NN50: {metrics_dict['NN50']} | SDANN: {sdann_str}")

            results['intervals'].append(interval_result)
        # Statistik über Intervalle (Requirement: > 7 Intervalle)
        if len(metrics_dictA) > 7:
            print(f"\n{'-' * 60}\nZUSAMMENFASSUNG (> 7 Intervalle)\n{'-' * 60}")
            hr_means = [r['HR_Mean'] for r in metrics_dictA]
            sdnns = [r['SDNN'] for r in metrics_dictA]

            summary = {
                'hr_mean_avg': np.mean(hr_means),
                'hr_mean_std': np.std(hr_means, ddof=1),
                'sdnn_avg': np.mean(sdnns),
                'sdnn_std': np.std(sdnns, ddof=1),
                'valid_interval_count': len(metrics_dictA)
            }
            results['summary'] = summary

            print(f"HR Mean:   {np.mean(hr_means):.2f} ± {np.std(hr_means, ddof=1):.2f} bpm")
            print(f"SDNN Mean: {np.mean(sdnns):.2f} ± {np.std(sdnns, ddof=1):.2f} ms")
        return results
