import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime 
import subprocess

from astropy.io import fits

def is_valid(f: Path, allow_siril: bool):
    if allow_siril:
        return not (f.is_symlink())
    else:
        return not (f.is_symlink() or f.name.startswith('r_') or f.name.startswith('pp_'))

def find_fits(target_dir: Path, pattern = "*.fit*", days = None, allow_siril = False):
    if days is not None:
        cmd = subprocess.run(
            ['find', str(target_dir), '-mtime', f'-{days}', '-name', pattern], 
            stdout=subprocess.PIPE, text=True
        )
        return [str(f) for f in cmd.stdout.splitlines() if is_valid(Path(str(f)), allow_siril)]
    else:
        return [str(f) for f in tqdm(target_dir.rglob(pattern), unit = ' files') if is_valid(f, allow_siril)]   

"""
Returns a tidy key:value dictionary of a FITS header, along with an id 
computed from a hash of the non-comment/history fields.
"""
def parse_header(header: fits.Header) -> dict:
    comment = '\n'.join(header.get('COMMENT', []))
    history = '\n'.join(header.get('HISTORY', []))
    date_obs = datetime.fromisoformat(header.get('DATE-OBS'))
    id = hash(frozenset(header.values()))
    return {
        'id':id, 
        'comment': comment, 
        'history': history, 
        'date_obs':date_obs, 
        **{k.replace('-', '_').lower(): v for k,v in header.items() if k not in ['HISTORY', 'COMMENT', 'DATE-OBS']}}
    
"""
Parses a list of files into a list of dictionaries from the parsed headers and file info. 
Files whose headers could not be parsed are returned in a separate list.
"""
def parse_files(files: list[str]) -> tuple[list[dict], list[str]]:
    rows, skipped = list(), list()
    for file in tqdm(files, unit = ' files', desc = 'Reading headers'):
        file = Path(file)
        try:
            header = fits.getheader(file)
            rows.append({**parse_header(header), 'name': file.name, 'path': str(file.parent)})
        except OSError as e:
            logging.info(f"Could not parse header for {file}: {e}")
            skipped.append(str(file))
    return rows, skipped