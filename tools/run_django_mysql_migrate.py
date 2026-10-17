"""
Run Django migrations against the authoritative Servy MySQL database.
Loads credentials from .env.mysql_credentials and invokes Django call_command('migrate').
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load MySQL credentials
with open(".env.mysql_credentials", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

os.environ["SERVY_DB_ENGINE"] = "mysql"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

import django
django.setup()

from django.core.management import call_command
from django.db import connection

print("Active Database Engine:", connection.settings_dict["ENGINE"])
print("Active Database Name:", connection.settings_dict["NAME"])
print("Active Host/Port:", connection.settings_dict["HOST"], connection.settings_dict["PORT"])

print("\n--- Running Django Migrations on MySQL ---")
call_command("migrate", interactive=False)
print("\nDjango migrations on MySQL completed successfully!")
