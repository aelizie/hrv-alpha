"""
Unit Tests für ECGAnalyzer (pytest)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from ecg_hrv_analysis import ECGAnalyzer  # noqa


@pytest.fixture(scope="session")
def paths():
    return {
        "root": ROOT,
        "config": ROOT / "config" / "conf.ini",
        "samples": ROOT / "samples",
        "ecg_file": ROOT / "samples" / "ecg1.mat",
    }


@pytest.fixture()
def analyzer(paths):
    return ECGAnalyzer(str(paths["config"]))


class TestConfigLoading:
    def test_ut_lc_01_valid_config_loads_and_sets_attributes(self, paths):
        """UT_LC_01 - Laden einer validen Konfigurationsdatei"""
        a = ECGAnalyzer(str(paths["config"]))

        for attr in [
            "samplingRate_hz",
            "peakThreshold_ratio",
            "refractoryPeriod_sec",
            "nn50Threshold_ms",
            "sdannWindow_sec",
        ]:
            assert hasattr(a, attr), f"Missing attribute: {attr}"

        assert a.samplingRate_hz == 200.0

    def test_ut_lc_02_missing_config_raises_filenotfound(self):
        """UT_LC_02 - Laden einer nicht existierenden Konfigurationsdatei"""
        with pytest.raises(FileNotFoundError):
            ECGAnalyzer("/nonexistent/path/conf.ini")

    def test_ut_lc_03_invalid_config_value_raises_valueerror(self, tmp_path):
        """UT_LC_03 - Laden einer fehlerhaften Konfigurationsdatei"""
        bad = tmp_path / "bad.ini"
        bad.write_text("[AnalysisParameters]\nsamplingRate_hz = not_a_number\n", encoding="utf-8")

        with pytest.raises(ValueError):
            ECGAnalyzer(str(bad))


class TestMatLoading:
    def test_ut_lms_01_load_valid_mat_returns_array(self, analyzer, paths):
        """UT_LMS_01 - Laden eines validen EKG-Signals"""
        signal = analyzer.loadEcgSignalFromMatFile(str(paths["samples"] / "ecg1.mat"))
        assert signal is not None
        assert isinstance(signal, np.ndarray)
        assert signal.size > 0

    def test_ut_lms_02_load_missing_mat_returns_none(self, analyzer):
        """UT_LMS_02 - Laden einer nicht existierenden .mat-Datei"""
        signal = analyzer.loadEcgSignalFromMatFile("/nonexistent/file.mat")
        assert signal is None


class TestSignalProcessing:
    def test_ut_cos_01_clip_outliers_normal_signal(self, analyzer):
        """UT_COS_01 - Outlier-Clipping mit normalem Signal"""
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 10.0, -10.0, 0.3, 0.4])
        clipped = analyzer.clipOutliersFromSignal(x)

        assert clipped.max() < 10.0
        assert clipped.min() > -10.0

    def test_ut_cos_02_clip_outliers_empty_array(self, analyzer):
        """UT_COS_02 - Outlier-Clipping mit leerem Array"""
        x = np.array([])
        clipped = analyzer.clipOutliersFromSignal(x)
        assert clipped.size == 0

    def test_ut_dps_01_detect_peaks_synthetic(self, analyzer):
        """UT_DPS_01 - Peak-Detektion mit synthetischem Signal"""
        signal_length = 2000
        s = np.zeros(signal_length)

        peak_positions = [250, 270, 400, 600, 880, 1200, 1460, 1500, 1600, 1890] # zwei mal sehr nahe peaks
        for pos in peak_positions:
            s[pos] = 1.0
            if 5 < pos < signal_length - 5: #Erzeuge künstliche Peak Rampe um Peak herum
                s[pos - 2 : pos] = [0.2, 0.5]
                s[pos + 1 : pos + 3] = [0.5, 0.2]

        peaks = analyzer.detectPeaksFromSignal(s)

        assert  len(peaks) == 8

    def test_ut_dps_02_detect_peaks_empty(self, analyzer):
        """UT_DPS_02 - Peak-Detektion mit leerem Signal"""
        peaks = analyzer.detectPeaksFromSignal(np.array([]))
        assert len(peaks) == 0

    def test_ut_dps_03_detect_peaks_constant_signal(self, analyzer):
        """UT_DPS_03 - Peak-Detektion mit konstantem Signal"""
        s = np.ones(1000) * 0.5
        peaks = analyzer.detectPeaksFromSignal(s)
        assert len(peaks) <= 2


class TestHrvMetrics:
    def test_ut_chm_01_metrics_known_nn(self, analyzer):
        """UT_CHM_01 - HRV-Metriken mit bekannten NN-Intervallen"""
        nn = np.array([800.0, 850.0, 900.0, 850.0, 800.0])

        expected_hr_mean = 60000.0 / np.mean(nn)
        expected_sdnn = np.std(nn, ddof=1)

        m = analyzer.calculateHrvMetrics(nn, 60)
        assert m is not None

        assert m["HR_Mean"] == pytest.approx(expected_hr_mean, abs=0.1)
        assert m["SDNN"] == pytest.approx(expected_sdnn, abs=0.1)

    def test_ut_chm_02_metrics_empty_returns_none(self, analyzer):
        """UT_CHM_02 - HRV-Metriken mit leerem Array"""
        m = analyzer.calculateHrvMetrics(np.array([]), 60)
        assert m is None

    def test_ut_chm_03_nn50(self, analyzer):
        """UT_CHM_03 - NN50-Berechnung"""
        nn = np.array([800.0, 900.0, 850.0, 950.0, 800.0])
        m = analyzer.calculateHrvMetrics(nn, 60)
        assert m is not None
        assert m["NN50"] == 3

    def test_ut_chm_04_sdann_short_duration_is_nan(self, analyzer):
        """UT_CHM_04 - SDANN mit kurzer Dauer (<5 Min)"""
        nn = np.array([800.0] * 100)
        m = analyzer.calculateHrvMetrics(nn, 60)
        assert m is not None
        assert np.isnan(m["SDANN"])

    def test_ut_chm_05_hr_min_max(self, analyzer):
        """UT_CHM_05 - HR Min/Max Berechnung"""
        nn = np.array([500.0, 1000.0, 750.0])
        m = analyzer.calculateHrvMetrics(nn, 60)
        assert m is not None

        assert m["HR_Min"] == pytest.approx(60.0, abs=0.1)
        assert m["HR_Max"] == pytest.approx(120.0, abs=0.1)



"""
System Tests für ECGAnalyzer (pytest)
Testet das Zusammenspiel aller Komponenten und die End-to-End-Funktionalität.
"""

def run_and_capture_stdout(capsys, fn, *args, **kwargs) -> str:
    # Quick created with AI, after some redundant code in following tests
    """Run function and return captured stdout."""
    fn(*args, **kwargs)
    out = capsys.readouterr().out
    return out


def extract_first_hr_bpm(output: str) -> float | None:
    # Quick created with AI, after some redundant code in following tests
    """
    Extrahiert den ersten HR-Wert aus Ausgabe:
    erwartet z.B. 'HR: 55.9 bpm'
    """
    m = re.search(r"HR:\s*([\d.]+)\s*bpm", output)
    return float(m.group(1)) if m else None


def assert_contains_metrics_for_intervals(output: str, expected_interval_count: int):
    for i in range(1, expected_interval_count + 1):
        assert f"Intervall {i}" in output, f"Missing 'Intervall {i}' in output"

    assert ("HR:" in output or "bpm" in output), "No HR / bpm found"
    assert "SDNN:" in output, "No SDNN found"
    assert "NN50:" in output, "No NN50 found"


class TestSystemAnalyzeFile:
    def test_st_af_01_valid_intervals_contains_metrics(self, analyzer, paths, capsys):
        """ST_AF_01 - Analyse mit validen Intervallen"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        intervals = [(0, 350), (350, 700)]
        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)

        assert_contains_metrics_for_intervals(out, expected_interval_count=2)

    def test_st_af_02_more_than_7_intervals_shows_summary(self, analyzer, paths, capsys):
        """ST_AF_02 - Analyse mit mehr als 7 Intervallen (Statistik-Trigger)"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        intervals = [
            (0, 350), (350, 700), (700, 1050), (1050, 1400),
            (1400, 1750), (1750, 2100), (2100, 2450), (2450, 2800),
        ]
        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)

        assert "Intervall 8" in out
        assert "ZUSAMMENFASSUNG" in out
        assert "HR Mean:" in out
        assert "SDNN Mean:" in out

    def test_st_af_03_exactly_7_intervals_has_no_summary(self, analyzer, paths, capsys):
        """ST_AF_03 - Analyse mit genau 7 Intervallen (keine Statistik)"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        intervals = [
            (0, 350), (350, 700), (700, 1050), (1050, 1400),
            (1400, 1750), (1750, 2100), (2100, 2450),
        ]
        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)

        assert "Intervall 7" in out
        assert "ZUSAMMENFASSUNG" not in out

    def test_st_af_04_invalid_intervals_are_skipped_but_valid_processed(self, analyzer, paths, capsys):
        """ST_AF_04 - Analyse mit ungültigen Intervallen"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        intervals = [
            (500, 400),      # Start > End
            (-10, 100),      # negative
            (0, 99999999),   # end huuuge
            (100, 200),      # valid
        ]
        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)

        assert ("HR:" in out or "bpm" in out), "Expected at least one HR result for valid interval"

    def test_st_af_05_very_short_interval_does_not_crash(self, analyzer, paths, capsys):
        """ST_AF_05 - Analyse mit sehr kurzem Intervall"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), [(0, 1)])

        assert isinstance(out, str)

    def test_st_af_06_long_interval_should_output_sdann_not_na(self, analyzer, paths, capsys):
        """ST_AF_06 - Analyse mit langem Intervall (>5 Min für SDANN)"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), [(0, 400)])

        assert "SDANN:" in out, "Expected SDANN line"
        assert "SDANN: N/A" not in out, "SDANN is N/A despite interval > 5 min (check implementation/data)"

    def test_st_af_07_missing_file_is_handled(self, analyzer, capsys):
        """ST_AF_07 - Analyse mit nicht existierender Datei"""
        out = run_and_capture_stdout(capsys, analyzer.analyze_file, "/nonexistent/file.mat", [(0, 100)])

        assert isinstance(out, str)

    def test_st_af_08_empty_intervals_is_handled(self, analyzer, paths, capsys):
        """ST_AF_08 - Analyse mit leerer Intervallliste"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), [])
        assert isinstance(out, str)

    def test_st_af_09_hr_plausibility_range(self, analyzer, paths, capsys):
        """ST_AF_09 - Plausibilitätsprüfung der HR-Werte"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), [(0, 60)])
        hr = extract_first_hr_bpm(out)

        assert hr is not None, "Could not parse HR from output"
        assert 30 <= hr <= 220, f"HR out of plausible range: {hr}"

    def test_st_af_10_repeatability_same_hr(self, analyzer, paths, capsys):
        """ST_AF_10 - Konsistenz über mehre Durchläufe"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        intervals = [(0, 100)]

        out1 = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)
        hr1 = extract_first_hr_bpm(out1)
        assert hr1 is not None

        out2 = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)
        hr2 = extract_first_hr_bpm(out2)
        assert hr2 is not None

        assert hr1 == pytest.approx(hr2, abs=0.01)

    def test_st_int_01_main_py_intervals_workflow(self, analyzer, paths, capsys):
        """ST_INT_01 - Integration: Vollständiger Workflow mit main.py Intervallen"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden: {paths['ecg_file']}")

        intervals = [
            (0, 350), (350, 700), (700, 1050), (1050, 1700),
            (1700, 2600), (2600, 3500), (3500, 4400), (4400, 5300),
        ]
        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)

        for i in range(1, 9):
            assert f"Intervall {i}" in out

        assert "ZUSAMMENFASSUNG" in out
        assert "BERICHT" in out


class TestSystemEdgeCasesMore:
    @pytest.mark.parametrize(
        "intervals",
        [
            [(0.0, 60.0)],               # floats
            [(0, 60), (10, 70)],         # overlaps
            [(0, 0)],                    # zero length
        ],
    )
    def test_more_edge_cases_do_not_crash(self, analyzer, paths, capsys, intervals):
        """Extra: ein paar fiese Interval-Edgecases"""
        if not paths["ecg_file"].exists():
            pytest.skip(f"Sample file nicht gefunden:{paths['ecg_file']}")

        out = run_and_capture_stdout(capsys, analyzer.analyze_file, str(paths["ecg_file"]), intervals)
        assert isinstance(out, str)
