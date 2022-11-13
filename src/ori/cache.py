from pathlib import Path
import sqlite3

class Cache(object):

    def _mktables(con, reset=False):
        with con:
            if reset:
                con.execute("drop table if exists headers")
                con.execute("drop table if exists skipped")
            con.execute(f"create table if not exists headers(file varchar, key varchar, value varchar)")
            con.execute(f"create table if not exists skipped(file varchar)")

    def __init__(self, target_dir:str, cache_name:str = ".fitz_cache", reset:bool = False):
        self._cache_file = Path(target_dir, cache_name)
        con = sqlite3.connect(self._cache_file)
        Cache._mktables(con, reset)

    def _connect(self):
        return sqlite3.connect(self._cache_file)

    def exists(self):
        self._cache_file.exists()

    def get(self):
        headers = list()
        skipped = list()
        con = self._connect()
        try:
            with con:
                cur = con.execute("select * from headers")
                headers = cur.fetchall()
                cur = cur.execute("select * from skipped")
                skipped = cur.fetchall()
        finally:
            con.close()
        return headers, skipped

    def set(self, headers, skipped):
        con = self._connect()
        try:
            Cache._mktables(con, reset=True)
            with con:
                con.executemany("insert into headers(file, key, value) values (?, ?, ?)", headers)
                con.executemany("insert into skipped(file) values (?)", [(v,) for v in skipped])
        finally:
            con.close()

    def reset(self):
        con = self._connect()        
        try:
            Cache._mktables(con, reset=True)
        finally:
            con.close()