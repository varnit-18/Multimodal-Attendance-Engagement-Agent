"""
Database Management Module
SQLite Persistent Storage for Users, Biometric Encodings, Attendance Logs, and Anomaly Events.
"""

import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from .config import DATABASE_PATH, OFFICE_START_TIME, LATE_GRACE_MINUTES, REPEAT_PUNCH_COOLDOWN_MINUTES


class DatabaseManager:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = str(db_path)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Create tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    role TEXT NOT NULL,
                    email TEXT,
                    face_embedding TEXT,
                    voice_embedding TEXT,
                    photo_path TEXT,
                    voice_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                )
            """)

            # Attendance Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    face_score REAL,
                    voice_score REAL,
                    liveness_score REAL,
                    total_score REAL NOT NULL,
                    verification_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT,
                    anomaly_flag INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # Anomalies and Security Flags
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    details TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    # ---------------- User Management ---------------- #

    def register_user(
        self,
        user_id: str,
        name: str,
        department: str,
        role: str,
        email: str = "",
        face_embedding: Optional[np.ndarray] = None,
        voice_embedding: Optional[np.ndarray] = None,
        photo_path: str = "",
        voice_path: str = ""
    ) -> bool:
        """Register or update a user with their biometric embeddings."""
        face_json = json.dumps(face_embedding.tolist()) if face_embedding is not None else None
        voice_json = json.dumps(voice_embedding.tolist()) if voice_embedding is not None else None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (user_id, name, department, role, email, face_embedding, voice_embedding, photo_path, voice_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    department = excluded.department,
                    role = excluded.role,
                    email = excluded.email,
                    face_embedding = COALESCE(excluded.face_embedding, users.face_embedding),
                    voice_embedding = COALESCE(excluded.voice_embedding, users.voice_embedding),
                    photo_path = COALESCE(excluded.photo_path, users.photo_path),
                    voice_path = COALESCE(excluded.voice_path, users.voice_path)
            """, (user_id, name, department, role, email, face_json, voice_json, photo_path, voice_path))
            conn.commit()
            return True

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ? AND is_active = 1", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("face_embedding"):
                data["face_embedding"] = np.array(json.loads(data["face_embedding"]))
            if data.get("voice_embedding"):
                data["voice_embedding"] = np.array(json.loads(data["voice_embedding"]))
            return data

    def get_all_users(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE is_active = 1 ORDER BY name ASC")
            rows = cursor.fetchall()
            users = []
            for r in rows:
                item = dict(r)
                if item.get("face_embedding"):
                    item["face_embedding"] = np.array(json.loads(item["face_embedding"]))
                if item.get("voice_embedding"):
                    item["voice_embedding"] = np.array(json.loads(item["voice_embedding"]))
                users.append(item)
            return users

    def delete_user(self, user_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET is_active = 0 WHERE user_id = ?", (user_id,))
            conn.commit()
            return True

    # ---------------- Attendance Logging ---------------- #

    def can_record_punch(self, user_id: str) -> tuple[bool, str]:
        """Check duplicate punch within cooldown period."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp FROM attendance_logs 
                WHERE user_id = ? AND status IN ('PRESENT', 'LATE')
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id,))
            row = cursor.fetchone()
            if row:
                last_time = datetime.datetime.fromisoformat(row["timestamp"])
                now = datetime.datetime.now()
                diff_minutes = (now - last_time).total_seconds() / 60.0
                if diff_minutes < REPEAT_PUNCH_COOLDOWN_MINUTES:
                    return False, f"Recent punch already registered {int(diff_minutes)} min ago. Cooldown is {REPEAT_PUNCH_COOLDOWN_MINUTES} min."
        return True, "OK"

    def record_attendance(
        self,
        user_id: str,
        face_score: float,
        voice_score: float,
        liveness_score: float,
        total_score: float,
        verification_mode: str,
        status: str,
        notes: str = "",
        anomaly_flag: int = 0
    ) -> int:
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Determine Late Status if not rejected
        if status == "PRESENT":
            office_start = datetime.datetime.strptime(OFFICE_START_TIME, "%H:%M:%S").time()
            grace_delta = datetime.timedelta(minutes=LATE_GRACE_MINUTES)
            cutoff = (datetime.datetime.combine(datetime.date.today(), office_start) + grace_delta).time()
            if now.time() > cutoff:
                status = "LATE"
                notes += f" [Arrived after grace cutoff {cutoff.strftime('%H:%M')}]"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO attendance_logs 
                (user_id, timestamp, date, time, face_score, voice_score, liveness_score, total_score, verification_mode, status, notes, anomaly_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, now.isoformat(), date_str, time_str,
                face_score, voice_score, liveness_score, total_score,
                verification_mode, status, notes, anomaly_flag
            ))
            conn.commit()
            return cursor.lastrowid

    def get_attendance_logs(
        self,
        date: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT a.*, u.name, u.department, u.role
            FROM attendance_logs a
            LEFT JOIN users u ON a.user_id = u.user_id
            WHERE 1=1
        """
        params: List[Any] = []
        if date:
            query += " AND a.date = ?"
            params.append(date)
        if user_id:
            query += " AND a.user_id = ?"
            params.append(user_id)

        query += " ORDER BY a.timestamp DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    # ---------------- Anomaly & Security Tracking ---------------- #

    def record_anomaly(self, user_id: Optional[str], anomaly_type: str, severity: str, details: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO anomalies (user_id, anomaly_type, severity, details)
                VALUES (?, ?, ?, ?)
            """, (user_id, anomaly_type, severity, details))
            conn.commit()

    def get_anomalies(self, resolved: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM anomalies"
        params = []
        if resolved is not None:
            query += " WHERE resolved = ?"
            params.append(resolved)
        query += " ORDER BY timestamp DESC"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def resolve_anomaly(self, anomaly_id: int):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE anomalies SET resolved = 1 WHERE id = ?", (anomaly_id,))
            conn.commit()

    # ---------------- Analytics & Summary Stats ---------------- #

    def get_summary_stats(self) -> Dict[str, Any]:
        today = datetime.date.today().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as total_enrolled FROM users WHERE is_active = 1")
            total_enrolled = cursor.fetchone()["total_enrolled"]

            cursor.execute("SELECT COUNT(DISTINCT user_id) as present_today FROM attendance_logs WHERE date = ? AND status IN ('PRESENT', 'LATE')", (today,))
            present_today = cursor.fetchone()["present_today"]

            cursor.execute("SELECT COUNT(*) as late_today FROM attendance_logs WHERE date = ? AND status = 'LATE'", (today,))
            late_today = cursor.fetchone()["late_today"]

            cursor.execute("SELECT COUNT(*) as total_anomalies FROM anomalies WHERE resolved = 0")
            unresolved_anomalies = cursor.fetchone()["total_anomalies"]

            attendance_rate = (present_today / total_enrolled * 100) if total_enrolled > 0 else 0.0

            return {
                "total_enrolled": total_enrolled,
                "present_today": present_today,
                "late_today": late_today,
                "attendance_rate": round(attendance_rate, 1),
                "unresolved_anomalies": unresolved_anomalies
            }

    def seed_default_data(self):
        """Populate initial mock users and attendance logs if database is empty."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as c FROM users")
            count = cursor.fetchone()["c"]
            if count > 0:
                return  # Already populated

        # Seed realistic users
        sample_users = [
            ("EMP001", "Aarav Sharma", "AI & Robotics", "Senior Research Engineer", "aarav.sharma@futureskills.ai"),
            ("EMP002", "Priya Nair", "Computer Vision", "Lead CV Scientist", "priya.nair@futureskills.ai"),
            ("EMP003", "Rahul Verma", "Data Science", "ML Ops Architect", "rahul.verma@futureskills.ai"),
            ("EMP004", "Ananya Iyer", "Audio Engineering", "Speech Processing Specialist", "ananya.iyer@futureskills.ai"),
            ("EMP005", "Kavya Patel", "Product Management", "AI Product Lead", "kavya.patel@futureskills.ai"),
        ]

        np.random.seed(42)
        for uid, name, dept, role, email in sample_users:
            # Deterministic synthetic biometric feature vectors for instant testing
            face_emb = np.random.randn(128).astype(np.float32)
            face_emb /= np.linalg.norm(face_emb)

            voice_emb = np.random.randn(20).astype(np.float32)
            voice_emb /= np.linalg.norm(voice_emb)

            self.register_user(
                user_id=uid,
                name=name,
                department=dept,
                role=role,
                email=email,
                face_embedding=face_emb,
                voice_embedding=voice_emb
            )

        # Seed sample attendance history for the past 3 days
        today = datetime.date.today()
        for day_offset in range(3, -1, -1):
            date_val = (today - datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")
            for uid, name, dept, role, email in sample_users:
                # 90% attendance chance
                if np.random.rand() > 0.15:
                    is_late = np.random.rand() > 0.8
                    status = "LATE" if is_late else "PRESENT"
                    h = 9 if not is_late else 10
                    m = np.random.randint(15, 30) if not is_late else np.random.randint(0, 45)
                    s = np.random.randint(10, 59)
                    time_val = f"{h:02d}:{m:02d}:{s:02d}"
                    ts = f"{date_val}T{time_val}"

                    f_score = float(np.random.uniform(0.78, 0.96))
                    v_score = float(np.random.uniform(0.72, 0.94))
                    l_score = float(np.random.uniform(0.85, 0.98))
                    tot = 0.5 * f_score + 0.35 * v_score + 0.15 * l_score

                    with self.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO attendance_logs
                            (user_id, timestamp, date, time, face_score, voice_score, liveness_score, total_score, verification_mode, status, notes, anomaly_flag)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            uid, ts, date_val, time_val,
                            round(f_score, 3), round(v_score, 3), round(l_score, 3), round(tot, 3),
                            "MULTIMODAL", status, "Automatic System Verification", 0
                        ))
                        conn.commit()

        # Seed sample anomaly
        self.record_anomaly(
            user_id="EMP003",
            anomaly_type="SPOOF_ATTEMPT",
            severity="MEDIUM",
            details="Static image presentation detected. Zero eye blink variance across 15 frames."
        )
