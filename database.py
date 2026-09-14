import sqlite3

DB_NAME = "agrishield.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnosis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mrl_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pesticide TEXT,
            application_date TEXT,
            harvest_date TEXT,
            estimated_residue REAL,
            safe_harvest INTEGER,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_diagnosis(image_hash, prediction, confidence, timestamp):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO diagnosis_logs
        (image_hash, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (image_hash, prediction, confidence, timestamp))

    conn.commit()
    conn.close()
def log_mrl(
    pesticide,
    application_date,
    harvest_date,
    estimated_residue,
    safe_harvest,
    timestamp
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO mrl_logs
        (pesticide, application_date, harvest_date,
         estimated_residue, safe_harvest, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        pesticide,
        application_date,
        harvest_date,
        estimated_residue,
        safe_harvest,
        timestamp
    ))

    conn.commit()
    conn.close()