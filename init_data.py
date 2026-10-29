"""
Initialization & Data Seeding Script
Generates sample assets and pre-populates SQLite database for offline evaluation.
"""

from src.database import DatabaseManager
from src.utils import generate_synthetic_assets_if_missing

if __name__ == "__main__":
    print("Initializing synthetic demo assets (faces and audio clips)...")
    generate_synthetic_assets_if_missing()
    print("Initializing SQLite database and seeding default users...")
    db = DatabaseManager()
    db.seed_default_data()
    print("Initialization completed successfully!")
