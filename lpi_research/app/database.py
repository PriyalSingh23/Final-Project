"""SQLite storage for LPI transmission history and EW benchmarks.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "lpi_experiments.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables for transmissions and benchmark runs."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transmissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        plaintext TEXT,
        ciphertext_hex TEXT,
        snr_db REAL,
        cfo_hz REAL,
        delay_samples INTEGER,
        recovered_text TEXT,
        success INTEGER,
        rs_errors_corrected INTEGER,
        crc_passed INTEGER,
        stealth_score REAL,
        stealth_grade TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        num_frames INTEGER,
        ks_p REAL,
        kurtosis REAL,
        spectral_entropy REAL,
        iq_circularity REAL,
        papr_db REAL,
        csfa_ratio REAL,
        c42 REAL,
        wvd_ratio REAL,
        composite_score REAL,
        stealth_grade TEXT,
        adversary_cnn_acc REAL,
        all_passed INTEGER
    )
    """)

    conn.commit()
    conn.close()


def log_transmission(record: Dict[str, Any]):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO transmissions (
        timestamp, plaintext, ciphertext_hex, snr_db, cfo_hz, delay_samples,
        recovered_text, success, rs_errors_corrected, crc_passed,
        stealth_score, stealth_grade
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        record.get("plaintext", ""),
        record.get("ciphertext_hex", ""),
        record.get("snr_db", 0.0),
        record.get("cfo_hz", 0.0),
        record.get("delay_samples", 0),
        record.get("recovered_text", ""),
        1 if record.get("success", False) else 0,
        record.get("rs_errors_corrected", 0),
        1 if record.get("crc_passed", False) else 0,
        record.get("stealth_score", 0.0),
        record.get("stealth_grade", "Unknown")
    ))
    conn.commit()
    conn.close()


def log_benchmark(bench: Dict[str, Any]):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO benchmarks (
        timestamp, num_frames, ks_p, kurtosis, spectral_entropy, iq_circularity,
        papr_db, csfa_ratio, c42, wvd_ratio, composite_score, stealth_grade,
        adversary_cnn_acc, all_passed
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        bench.get("num_frames", 500),
        bench.get("ks_p", 1.0),
        bench.get("kurtosis", 3.0),
        bench.get("spectral_entropy", 1.0),
        bench.get("iq_circularity", 0.0),
        bench.get("papr_db", 0.0),
        bench.get("csfa_ratio", 1.0),
        bench.get("c42", 0.0),
        bench.get("wvd_ratio", 1.0),
        bench.get("composite_score", 0.0),
        bench.get("stealth_grade", "Excellent"),
        bench.get("adversary_cnn_acc", 50.0),
        1 if bench.get("all_passed", True) else 0
    ))
    conn.commit()
    conn.close()


def get_recent_transmissions(limit: int = 25) -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transmissions ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
