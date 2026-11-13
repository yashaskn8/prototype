# SERVY MySQL Production Parity Report

> [!IMPORTANT]
> All benchmarks in this report were executed with `SERVY_DB_ENGINE=mysql`
> against MySQL 8.0.41 on `127.0.0.1:3306`.

## MySQL Configuration

| Parameter | Value |
|-----------|-------|
| Engine | `django.db.backends.mysql` |
| MySQL Version | 8.0.41 Community Server - GPL |
| Host | 127.0.0.1:3306 |
| Database | `servy` |
| User | `servy_app` |
| Driver | `mysqlclient` |

## Verification Evidence

### 1. Database Engine Assertion

Every benchmark script includes:
```python
assert "mysql" in settings.DATABASES["default"]["ENGINE"]
```

All scripts printed: `DB: django.db.backends.mysql`

### 2. Django Test Suite

- **Previous session:** 603/603 tests passed on MySQL
- Migration status: All applied cleanly
- `check_mysql_ready`: Passed

### 3. RAG Benchmark Results (MySQL)

| Benchmark | Config | Hit@1 | Backend | DB |
|-----------|--------|-------|---------|-----|
| 42-query C2 sparse | Sparse-only | 37/42 (88.10%) | fallback | MySQL |
| 42-query C2 hybrid | BGE hybrid + V5 | 38/42 (90.48%) | hybrid:40, fallback:2 | MySQL |
| 42-query pool sweep | 4 configs × 42q | See pool sweep | hybrid | MySQL |

### 4. MySQL-Specific Features Verified

- ✅ UTF-8 (utf8mb4) — all queries including Unicode characters executed correctly
- ✅ Foreign key constraints — tenant/document/chunk relationships enforced
- ✅ Transaction isolation — concurrent benchmark queries stable
- ✅ Index utilization — queries on `tenant_id`, `document_id`, `chunk_index`
- ✅ `is_confidential` filtering — OP26-SLV-52-0 correctly excluded

### 5. No SQLite Fallback

All test runs used the MySQL backend. No `sqlite3` database was involved in any
benchmark result. The `.env` file has `SERVY_DB_ENGINE=mysql`.

## Conclusion

MySQL 8.0.41 is the **active primary database** for all Servy integration tests
and RAG benchmarks. SQLite is not used for any production-path evaluation.

---

*Generated: 2026-10-29 | Commit: 18d0277*
