"""Build an exact, resumable public-record lookup for the full GPU score census."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3
import time
import pyarrow.parquet as pq
from barbet_pipeline.common import read_json, save_new, sha_file, digest
from scripts.run_full_v4_gpu_quality import download

STATE = Path('/root/barbet-context-sft-data/reports/full-corpus-v4')
GPU = STATE / 'gpu-quality'


def main():
    final = GPU / 'provenance.sqlite3'
    if final.exists():
        receipt = read_json(GPU / 'provenance-index.json')
        assert sha_file(final) == receipt['database_sha256']
        return
    scratch = GPU / 'provenance-downloads'
    scratch.mkdir(exist_ok=True)
    database = GPU / 'provenance.inprogress.sqlite3'
    connection = sqlite3.connect(database)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=FULL')
    connection.execute('CREATE TABLE IF NOT EXISTS records (record_id BLOB PRIMARY KEY, parent_record_id TEXT NOT NULL, source TEXT NOT NULL, split TEXT NOT NULL, status TEXT NOT NULL, root_task_id TEXT NOT NULL, query_hash TEXT NOT NULL, verification_scope TEXT NOT NULL) WITHOUT ROWID')
    connection.execute('CREATE TABLE IF NOT EXISTS imported (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, rows INTEGER NOT NULL, exported INTEGER NOT NULL) WITHOUT ROWID')

    def fetch(item):
        path = scratch / (digest(item['remote']) + '.parquet')
        if path.exists():
            assert sha_file(path) == item['sha256']
        else:
            download(item, path)
        return item, path

    while True:
        items = {}
        for p in sorted((STATE / 'publication/groups').glob('*.json')):
            group = read_json(p)
            assert group['all_anonymous_hashes_match']
            for f in group['files']:
                if '/dispositions/' in f['remote']:
                    assert f['remote'] not in items
                    items[f['remote']] = {**f, 'revision': group['head']}
        imported = {r[0]: r[1:] for r in connection.execute('SELECT path,sha256,rows,exported FROM imported')}
        for name, values in imported.items():
            assert name in items and values[0] == items[name]['sha256']
        pending = [items[k] for k in sorted(set(items) - set(imported))]
        # Submit only a bounded window; never download the whole corpus at once.
        for offset in range(0, len(pending), 4):
            with ThreadPoolExecutor(4) as pool:
                downloaded = list(pool.map(fetch, pending[offset:offset + 4]))
            for item, path in downloaded:
                exported = 0
                count = 0
                with connection:
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                        values = []
                        for row in batch.to_pylist():
                            count += 1
                            if row['record_id'] is None:
                                assert row['status'] == 'quarantined'
                                continue
                            assert row['status'] in ['verified', 'source_reference', 'reference_only']
                            record_id = bytes.fromhex(row['record_id'])
                            assert len(record_id) == 32
                            values.append((record_id, row['parent_record_id'], row['source'], row['split'], row['status'], row['root_task_id'], row['query_hash'], row['scope']))
                        connection.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?)', values)
                        exported += len(values)
                    assert count == item['rows']
                    connection.execute('INSERT INTO imported VALUES (?,?,?,?)', (item['remote'], item['sha256'], count, exported))
                path.unlink()  # Only this disposable verified download, never sources.
            print(json.dumps({'indexed_disposition_files': connection.execute('SELECT COUNT(*) FROM imported').fetchone()[0],
                              'indexed_rows': connection.execute('SELECT SUM(exported) FROM imported').fetchone()[0]}), flush=True)
        if (GPU / 'full-score-plan.json').exists():
            processing = read_json(STATE / 'processing-summary.json')
            total = connection.execute('SELECT SUM(rows), SUM(exported), COUNT(*) FROM imported').fetchone()
            if total == (processing['counts']['active_rows_processed'], processing['counts']['exported'], processing['source_shards']):
                assert connection.execute('SELECT COUNT(*) FROM records').fetchone()[0] == processing['counts']['exported']
                connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                connection.execute('PRAGMA journal_mode=DELETE')
                assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                connection.close()
                database.rename(final)
                save_new(GPU / 'provenance-index.json', {'complete': True, 'rows': total[1], 'parents': total[0], 'source_shards': total[2],
                    'database_sha256': sha_file(final), 'processing_sha256': sha_file(STATE / 'processing-summary.json'),
                    'indexer_sha256': sha_file(__file__), 'source': 'all anonymously byte-verified v4 disposition files',
                    'private_oracles_used': False})
                return
        time.sleep(5)


if __name__ == '__main__':
    main()
