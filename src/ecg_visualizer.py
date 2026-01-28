"""
ECG Visualizer Modul
Autor: EmSt
Datum: 28.01.2026

Dieses Modul stellt Funktionen zur Visualisierung von EKG-Signalen und
detektierten R-Zacken bereit. Es ist als separates Modul konzipiert,
um die Modularität des Projekts zu wahren.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Nicht-interaktives Backend für Server-Umgebungen
import matplotlib.pyplot as plt

logger = logging.getLogger("ECG_Visualizer")


def _draw_core(time_sec, signal, r_peaks_time=None, r_peaks_amp=None, title="", output_path=None):
    """Interne Hilfsfunktion für das Rendering der Matplotlib-Grafik, aufgehübscht mit Gemini"""
    fig, ax = plt.subplots(figsize=(14, 5))

    # EKG Signal plotten
    ax.plot(time_sec, signal, 'b-', linewidth=0.5, label='EKG Signal')

    # R-Zacken markieren
    if r_peaks_time is not None and len(r_peaks_time) > 0:
        ax.scatter(r_peaks_time, r_peaks_amp, c='red', marker='o', s=30,
                   label=f'R-Zacken (n={len(r_peaks_time)})', zorder=5)

    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Speichern, falls Pfad angegeben
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grafik gespeichert unter: {output_path}")

    plt.close(fig)
    return output_path

def plot_ecg_with_peaks(signal_mvA, r_peaks_indices, sampling_rate_hz,
                        title="EKG Signal mit R-Zacken",
                        output_path=None,
                        time_range_sec=None):
    """
    Erstellt eine Visualisierung eines EKG-Signals mit markierten R-Zacken.

    :param signal_mvA: Das EKG-Signal als numpy Array (in mV).
    :type signal_mvA: numpy.ndarray
    :param r_peaks_indices: Die Indizes der detektierten R-Zacken im Signal.
    :type r_peaks_indices: list[int] | numpy.ndarray
    :param sampling_rate_hz: Die Abtastrate des Signals in Hz.
    :type sampling_rate_hz: float
    :param title: Der Titel der Grafik.
    :type title: str
    :param output_path: Der Pfad, unter dem die Grafik gespeichert werden soll.
                        Wenn None, wird die Grafik nicht gespeichert.
    :type output_path: str | None
    :param time_range_sec: Ein Tupel (start_sec, end_sec), um nur einen Ausschnitt
                           des Signals darzustellen. Wenn None, wird das gesamte Signal gezeigt.
    :type time_range_sec: tuple[float, float] | None
    :return: Der Pfad zur gespeicherten Grafik oder None, wenn nicht gespeichert.
    :rtype: str | None
    """
    if signal_mvA is None or len(signal_mvA) == 0:
        logger.warning("Kein Signal zum Visualisieren vorhanden.")
        return None

        # Zeitachse und Slicing vorbereiten
    start_idx, end_idx = 0, len(signal_mvA)
    if time_range_sec:
        start_idx = max(0, int(time_range_sec[0] * sampling_rate_hz))
        end_idx = min(len(signal_mvA), int(time_range_sec[1] * sampling_rate_hz))

    plot_signal = signal_mvA[start_idx:end_idx]
    time_sec = np.arange(len(plot_signal)) / sampling_rate_hz + (start_idx / sampling_rate_hz)

    # R-Peaks filtern, die im Ausschnitt liegen
    r_indices = np.array(r_peaks_indices)
    mask = (r_indices >= start_idx) & (r_indices < end_idx)
    valid_indices = r_indices[mask]

    return _draw_core(
        time_sec, plot_signal,
        r_peaks_time=valid_indices / sampling_rate_hz,
        r_peaks_amp=signal_mvA[valid_indices.astype(int)],
        title=title, output_path=output_path
    )


def plot_interval_results(analysis_results, signal_mvA, sampling_rate_hz,
                          output_dir="./plots",
                          interval_index=None):
    """
    Erstellt Visualisierungen für ein oder mehrere Analyseintervalle.

    :param analysis_results: Das Ergebnis-Dictionary von analyze_file().
    :type analysis_results: dict
    :param signal_mvA: Das vollständige EKG-Signal.
    :type signal_mvA: numpy.ndarray
    :param sampling_rate_hz: Die Abtastrate des Signals.
    :type sampling_rate_hz: float
    :param output_dir: Das Verzeichnis, in dem die Grafiken gespeichert werden.
    :type output_dir: str
    :param interval_index: Der Index des zu visualisierenden Intervalls (1-basiert).
                           Wenn None, werden alle erfolgreichen Intervalle visualisiert.
    :type interval_index: int | None
    :return: Eine Liste der Pfade zu den erstellten Grafiken.
    :rtype: list[str]
    """
    if analysis_results is None:
        logger.warning("Keine Analyseergebnisse zum Visualisieren vorhanden.")
        return []

    saved_paths = []
    source_file = analysis_results['metadata']['source_file']
    intervals_to_plot = analysis_results['intervals']

    if interval_index is not None:
        intervals_to_plot = [i for i in intervals_to_plot if i['interval_index'] == interval_index]

    for interval in intervals_to_plot:
        if interval['status'] != 'success':
            logger.info(f"Überspringe Intervall {interval['interval_index']} (Status: {interval['status']})")
            continue

        # Da r_peaks_indices im Dictionary relativ zum Intervallstart sind,
        # rechnen wir sie hier in absolute Indizes um, damit plot_ecg_with_peaks sie korrekt sliced.
        t_start = interval['start_sec']
        abs_peaks = np.array(interval['r_peaks_indices']) + int(t_start * sampling_rate_hz)

        path = os.path.join(output_dir, f"{source_file}_intervall_{interval['interval_index']}.png")
        title = f"{source_file} - Intervall {interval['interval_index']} ({t_start}s-{interval['end_sec']}s)\nHR: {interval['metrics']['HR_Mean']:.1f} bpm"

        saved_path = plot_ecg_with_peaks(
            signal_mvA, abs_peaks, sampling_rate_hz,
            title=title, output_path=path, time_range_sec=(t_start, interval['end_sec'])
        )
        if saved_path: saved_paths.append(saved_path)

    return saved_paths


def plot_ecg_overview(signal_mvA, sampling_rate_hz, output_path=None,
                      title="EKG Signal Übersicht"):
    """
    Erstellt eine Übersichtsgrafik des gesamten EKG-Signals.

    :param signal_mvA: Das EKG-Signal als numpy Array.
    :type signal_mvA: numpy.ndarray
    :param sampling_rate_hz: Die Abtastrate des Signals.
    :type sampling_rate_hz: float
    :param output_path: Der Pfad zum Speichern der Grafik.
    :type output_path: str | None
    :param title: Der Titel der Grafik.
    :type title: str
    :return: Der Pfad zur gespeicherten Grafik oder None.
    :rtype: str | None
    """
    if signal_mvA is None or len(signal_mvA) == 0:
        logger.warning("Kein Signal zum Visualisieren vorhanden.")
        return None

    duration_min = len(signal_mvA) / sampling_rate_hz / 60
    time_sec = np.arange(len(signal_mvA)) / sampling_rate_hz

    return _draw_core(
        time_sec, signal_mvA,
        title=f"{title} (Dauer: {duration_min:.1f} min)",
        output_path=output_path
    )
