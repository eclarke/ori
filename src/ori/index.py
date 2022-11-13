from __future__ import annotations
from pathlib import Path
import logging
import shutil
from datetime import datetime
import warnings
import os

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning

from .cache import Cache
from .config import get_config

tqdm.pandas()

def headers_to_df(headers: list(tuple[str, str, str])) -> pd.DataFrame:
    df = pd.DataFrame(headers, columns = ['file', 'attr', 'value'])
    df = df[~df.attr.isin(['COMMENT','HISTORY','NOTE'])]
    df = df.pivot(index='file', columns='attr', values = 'value').reset_index()
    return df

class Index(object):

    _required_keys = ['INSTRUME', 'IMAGETYP', 'CCD-TEMP', 'DATE-OBS', 'EXPTIME', 'GAIN', 'XBINNING', 'YBINNING', 'OFFSET']

    def __init__(self, headers, target_dir, index_key='file', missing_label = '(Unknown)'):

        def normalize_temps(t):
            if pd.isna(t):
                return t
            normal_temps = [0, -10, -15, -20]
            if round(t)+1 in normal_temps: return round(t) + 1
            if round(t)-1 in normal_temps: return round(t) - 1
            return round(t)
        
        self.config = get_config()

        base_columns = self.config['required'] + self.config['mutable']
        tz = self.config.get('timezone', 'UTC')

        df = headers_to_df(headers) 
        df = df.reindex(columns = list(set(df.columns.to_list() + base_columns)))
        df['_CCDTEMP'] = df['CCD-TEMP'].astype('float').map(normalize_temps).map("{:g}C".format)
        df['_GAIN'] = df['GAIN'].astype('float').map("{:1.0f}".format)
        df['_OFFSET'] = df['OFFSET'].astype('float').map("{:1.0f}".format)
        df['_DEC'] = df['DEC'].astype('float').round(2)
        df['_RA'] = df['RA'].astype('float').round(2)
        df['_LOCALDT'] = pd.to_datetime(df['DATE-OBS'], utc=True).dt.tz_convert(tz)
        df['_LOCALDATE'] = df['_LOCALDT'].dt.date
        df['_LOCALTIME'] = df['_LOCALDT'].dt.timetz
        df['_NIGHT'] = df._LOCALDT.round('D').dt.date.astype('string')
        df['_BINNING'] = df[['XBINNING', 'YBINNING']].astype('str').agg('x'.join, axis=1) 
        df['_IMAGETYP'] = df['IMAGETYP'].str.replace(' Frame', '').str.lower()
        df['_CHANGED'] = False        
        df['_NAME'] = df.file.apply(lambda f: str(Path(f).name))
        df['_PATH'] = df.file.apply(lambda f: str(Path(f).parent))
        df['_newname'] = pd.NA
        df['_newpath'] = pd.NA

        missing_session_info = df.SESSION.isna()
        if any(missing_session_info):
            df.loc[df.SESSION.isna(), ['SESSION']] = df.loc[df.SESSION.isna(), ['_NIGHT']]
            df['_CHANGED'] = missing_session_info

        self._df = df.set_index(index_key)
        self._missing_label = missing_label
        self._rowmask = [True for _ in self.df.index]
        self._selected_attrs = []
        self.selected_attrs = self.config['defaults']
        self._root_dir = str(target_dir)
        self._cache = Cache(target_dir)
        self._index = index_key
        self._cache_valid = True
        self._prev_rowmasks = []
 

    #---- Df Methods
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, newdf: pd.DataFrame):
        self._df = newdf.reset_index().set_index(self._index)


    #---- Rowmask Methods
    @property
    def rowmask(self):
        return self._rowmask

    @rowmask.setter
    def rowmask(self, new_mask):
        assert len(new_mask) == len(self._df)
        if not any(new_mask):
            logging.warn("No rows selected")
        self._rowmask = new_mask

    def reset_rowmask(self):
        self._rowmask = [True for _ in self.df.index]

    def stash_rowmask(self):
        self._prev_rowmasks.append(self.rowmask)

    #---- Attribute selection Methods
    @property
    def selected_attrs(self):
        return self._selected_attrs

    @selected_attrs.setter
    def selected_attrs(self, new_attrs):
        _selected_attrs = [attr for attr in list(dict.fromkeys(new_attrs)) if attr in self.df.columns]
        if len(_selected_attrs) < len(set(new_attrs)):
            logging.warn(f"Some attrs skipped as they were not present in any headers: {[_ for _ in new_attrs if _ not in _selected_attrs]}")
        if len(_selected_attrs) == 0:
            logging.warn("No attrs selected")
        self._selected_attrs = _selected_attrs


    def all_attrs(self, defaults = list()):
        all_attrs = sorted(self.df.columns.to_list())
        _defaults = [_ for _ in defaults if _ in all_attrs]
        return _defaults + [_ for _ in all_attrs if _ not in _defaults]


    def _invalidate_cache(self):
        self._cache.reset()


    #---- Attribute values

    def values_for_attr(self, attr):
        selection = self.df.loc[self.rowmask].copy()
        values = selection[attr].fillna(self._missing_label)
        for value in values.unique():
            if value == self._missing_label:
                _df = selection[selection[attr].isna()].reset_index()
            else:
                _df = selection[selection[attr] == value].reset_index()
            
            nfiles = _df.file.nunique()
            ncombos = _df.drop(columns = 'file').drop_duplicates().shape[0]

            yield {
                'value': value,
                'ncombos': ncombos,
                'nfiles': nfiles
            }

    @property
    def selection(self) -> pd.DataFrame:
        return self.df.loc[self._rowmask, self._selected_attrs]

    @selection.setter
    def selection(self, new_selection: pd.DataFrame):
        self.rowmask = self.df.index.isin(new_selection.index)
        self.selected_attrs = new_selection.columns
        return self

    def summary(self, missing_label = ' ∅'):
        selection = self.df[self.rowmask].copy()
        selection.loc[selection.IMAGETYP.isin(['Bias Frame','Dark Frame']), ['OBJECT', 'FILTER']] = np.nan
        selection.loc[selection.IMAGETYP == 'Flat Frame', ['CCD-TEMP', 'OBJECT']] = np.nan
        selection = selection.fillna(missing_label)
        if len(self.selected_attrs) == 0:
            print("No attributes selected, using default attributes")
            group_keys = [_ for _ in self.config['defaults'] if _ not in ['_PATH', 'EXPTIME']]
        else:
            group_keys = self.selected_attrs
        if 'EXPTIME' not in group_keys:
            _group_exp = list(group_keys + ['EXPTIME'])
            group = selection[_group_exp].groupby(group_keys)
            return (
                    group.agg(
                        SumExpTime = pd.NamedAgg('EXPTIME', lambda x: x.astype('float').sum()),
                        NFiles = pd.NamedAgg('EXPTIME', lambda x: x.size))
                    .sort_values(by = ['NFiles', 'SumExpTime'], ascending=False)
                    .assign(SumExpTime = lambda df: df.SumExpTime.map('{:,g}s'.format))
                    .rename(columns = {'NFiles':'Files', 'SumExpTime':'𝚺 Exp. Time'})
            )
        else:
            return (
                selection.value_counts(subset = group_keys).to_frame('# files')
            )


    def problem_files(self):
        selection = self.df[self.rowmask].copy()
        calib_attrs = self.config['calibration']
        catalog_attrs = self.config['catalog']

        def core_attrs(imgtype, exceptions = ['_GAIN_OFFSET', 'IMAGETYP', '_IMAGETYP', 'DATE-OBS']):
            attrs = calib_attrs.get(imgtype, {}).get('attrs', []) + catalog_attrs.get(imgtype, {}).get('path', []) + catalog_attrs.get(imgtype, {}).get('name', [])
            return list(dict.fromkeys(a for a in attrs if a not in exceptions))
        
        for imgtype, _df in selection.groupby(['_IMAGETYP'], as_index=False):
            attrs = core_attrs(imgtype)
            _df = _df[attrs + ['_PATH']]
            missing = _df.apply(lambda row: any(row.isna()), axis=1)
            yield imgtype, attrs + ['_PATH', '_IMAGETYP'], _df[missing].fillna('<!>')


    def files(self):
        return self.selection.index.tolist()

    def _update_changed_vec(self, mask = None, files = None):
        if mask is not None:
            assert len(mask) == len(self.df)    
        elif files is not None:
            mask = self.df.index.isin(files)
        else:
            raise ValueError("At least one of mask or files needs to be specified")
        self.df['_CHANGED'] = mask | self.df['_CHANGED']            
        

    def _change_attr(self, attr, value):
        if attr in self.df.columns:
            changed_values = (self.df[attr] != value) & (self.rowmask)
        else:
            self.df[attr] = pd.NA
            changed_values = self.rowmask

        self.df.loc[self.rowmask, [attr]] = value
        self._update_changed_vec(mask = changed_values)
        if attr not in self.selected_attrs:
            self.selected_attrs += [attr]


    def changed(self) -> pd.DataFrame:
        return self.df[self.df['_CHANGED']]


    def _calibration_(self, IMAGETYP):
        imgtype = IMAGETYP.replace(' Frame', '').lower()
        return self.config['calibration'][imgtype]


    def calibration_frames(self, c_type):
        c_type = c_type.replace(' Frame', '').lower()
        c_data = self.config['calibration'][c_type]
        c_attrs = c_data['attrs']
        c_targets = c_data['targets']
        calib_files = self.df[self.df._IMAGETYP == c_type].reset_index()[c_attrs + [self.df.index.name]]
        df = self.df[self.rowmask].copy()
        target_files = df[df._IMAGETYP.isin(c_targets)].reset_index()[c_attrs].drop_duplicates().dropna()
        #target_files = self.df[self.rowmask].query(f"_IMAGETYP in @target_types").reset_index()[c_attrs].drop_duplicates().dropna()
        calib_files = pd.merge(target_files, calib_files, how = 'outer', indicator = True).query('_merge.isin(["both", "left_only"])')
        calib_files = calib_files.assign(_STATUS = lambda x: [{'both':'Available', 'left_only':'Missing'}[_] for _ in x._merge]).drop(columns = '_merge').set_index(self._index)
        return calib_files

    def calibration_frames2(self, c_type, exp_tolerance = 2, temp_tolerance = 0):
        c_type = c_type.replace(' Frame', '').lower()
        c_data = self.config['calibration'][c_type]
        c_attrs = c_data['attrs']
        c_targets = c_data['targets']
        df = self.df.copy()
        df = df.assign(
            EXPTIME = lambda df: df.EXPTIME.astype('float').round(exp_tolerance),
            _CCDTEMP = lambda df: df['CCD-TEMP'].astype('float').round(temp_tolerance)
        )
        c_files = df.loc[df._IMAGETYP == c_type, c_attrs].reset_index()
        t_files = df.loc[self.rowmask & df._IMAGETYP.isin(c_targets)].reset_index()[c_attrs].drop_duplicates().dropna()
        c_files = pd.merge(t_files, c_files, how = 'outer', indicator = True).query('_merge.isin(["both", "left_only"])')
        c_files = c_files.assign(_STATUS = lambda x: [{'both':'Available', 'left_only':'Missing'}[_] for _ in x._merge]).drop(columns = '_merge').set_index(self._index)
        return c_files



    """
    Syncs updated attributes and moves/copies files as defined by the _newpath and _newname 
    virtual attributes.
    - op: if 'copy', write changes to a copy whose path is determined by _newpath/_newname. The updated
        file is not added to the Index. If 'move', write changes to original file and move it. The updated 
        file replaces the previous one in the Index. If 'symlink', create a symlink from the original file 
        to the new destination. The symlink is not added to the Index.
    - overwrite_existing: If True, replace any existing files that exist in _newpath/_newname.
    """         
    def sync(self, op = True, overwrite_existing = False, save_oldname = False):

        def _move_copy_or_symlink(op, file, newfile: Path):
            if not newfile.parent.exists():
                newfile.parent.mkdir(parents = True, exist_ok=True)
            if op == 'copy':
                newfile = shutil.copy(file, newfile)
            elif op == 'move':
                newfile = shutil.move(file, newfile)
            elif op == 'symlink':
                os.symlink(file, newfile)
            return newfile

        def _add_oldname(header, oldname, newname):
            header.add_history(f'({datetime.now().date()}) filename: {oldname} -> {newname}')
            return header

        def _update_header(row, header: fits.Header, to_update = self.config['mutable']):
            new_values = row.filter(items = to_update).dropna().to_dict()
            for key, new_value in new_values.copy().items():
                old_value = header.get(key)
                if str(old_value) != str(new_value):
                    header.add_history(f'({datetime.now().date()}) {key}: {old_value} -> {new_value}')
                if key == 'SESSION':
                    new_values[key] = str(new_value)
                
            header.update(new_values)
            return header

        def _sync_row(row):
            oldpath = row._PATH
            oldname = row._NAME
            newpath = row._newpath if not pd.isna(row._newpath) else oldpath
            newname = row._newname if not pd.isna(row._newname) else oldname
            newfile = Path(newpath, newname)
            do_rename = (newpath != oldpath) or (newname != oldname)
            if do_rename:
                if newfile.exists() and not overwrite_existing:
                    print(f"Error: desired filename already exists ({newfile})")
                    return
                newfile = _move_copy_or_symlink(op, row.file, newfile)
                self.df.loc[self.df.index == row.file, ['_newpath', '_newname']] = pd.NA
        
            if not op == 'symlink':
                if ((newname != oldname) & save_oldname) or row._CHANGED:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', AstropyUserWarning)
                        with fits.open(newfile, mode='update', output_verify = 'silentfix+ignore' ) as fitsfile:
                            header = fitsfile[0].header
                            if newname != oldname:
                                header = _add_oldname(header, oldname, newname)
                            if row._CHANGED:
                                header = _update_header(row, header)
                            fitsfile[0].header = header
                            fitsfile.flush(output_verify='silentfix+ignore')

                self.df.loc[self.df.index == row.file, ['_CHANGED']] = False
                if op == 'move':
                    self.df.loc[self.df.index == row.file, ['_path']] = newpath
                    self.df.rename(index = {row.file:newfile}, inplace = True)
            
                if row._CHANGED and not do_rename:
                    if not do_rename:
                        # invalidate cache since the headers have changed but the paths have not
                        self._cache_valid = False

        self.changed().reset_index().progress_apply(_sync_row, axis=1)
        
        if not self._cache_valid:
            self._invalidate_cache()


    def change_subtree(self, destination):
        df = self.df.reset_index()
        
        def _new_subtree(file):
            try:
                path = str(Path(destination, Path(file).relative_to(self._root_dir).parent))
                return path
            except ValueError as e:
                logging.warn(f"Skipping {file} as it is not in original target directory")
                return pd.NA

        df.loc[self.rowmask, ['_newpath']] = df.loc[self.rowmask].apply(lambda row: _new_subtree(row.file), axis=1)
        df.loc[self.rowmask, ['_CHANGED']] = True
        self.df = df
        

    def organize(self, root_dir, structure_key = 'catalog'):
        catalog_attrs = self.config[structure_key]
        base_path = catalog_attrs['_base_path']
        base_name = catalog_attrs['_base_name']
        attr_structure = {
            imgtype: {
                'path': base_path + catalog_attrs[imgtype].get('path', []),
                'name': base_name + catalog_attrs[imgtype].get('name', [])
                
            } for imgtype in catalog_attrs if imgtype not in ['_base_path', '_base_name']
        }

        def _required_attrs(imgtype, ignored = []):
            attrs = attr_structure[imgtype]
            attrs = set(attrs['path'] + attrs['name'] + ['DATE-OBS', '_IMAGETYP'])
            return [_ for _ in attrs if _ not in ignored]

        def _get_new_name(row):
            def namepart(attr):
                part = str(row[attr]).replace(' ', '').replace(':', '-').replace('_', '-')
                if attr in ['_GAIN', '_OFFSET']:
                    part = attr.strip('_').lower() + part
                return part
            attrs = attr_structure[row._IMAGETYP]['name'] 
            if len(attrs) == 0:
                return pd.NA
            else:
                attrs = attrs + ['_SEQNO']
                return '_'.join(namepart(attr) for attr in attrs) + '.fits'
            

        def _get_new_path(row):
            def subdir(attr):
                value = row[attr]
                key = attr.lower().strip('_')
                if pd.isna(value) or value == 'nan' or value == self._missing_label:
                    value = f'unknown {key}'    
                if key in ['binning', 'gain', 'offset']:
                    value = key + '-' + value.replace(' ', '')
                return value
            attrs = attr_structure[row._IMAGETYP]['path']
            return str(Path(root_dir, Path(*[subdir(attr) for attr in attrs])))
            # return str(Path(root_dir, Path(*[str(row[s]).replace(' ', '_') for s in attr_structure[row._IMAGETYP]['path']])))

        img_groups = self.df[self.rowmask].groupby('_IMAGETYP', as_index=False)
        for imgtype, _df in img_groups:
            base_attrs = _required_attrs(imgtype)
            group_attrs = [_ for _ in attr_structure[imgtype]['name'] if _ not in ['DATE-OBS', '_SEQNO']]

            df = self.df.loc[self.df.index.isin(_df.index), base_attrs]
            
            if 'EXPTIME' in base_attrs:
                df['EXPTIME'] = df.EXPTIME.astype('float').map('{:g}s'.format)

            if len(group_attrs) > 0:
                df['_SEQ'] = df.groupby(group_attrs)['DATE-OBS'].rank('dense')
                df['_SEQNO'] = df._SEQ.map('{:1.0f}'.format).str.zfill(3)
            
            df['_newname'] = df.apply(_get_new_name, axis=1)
            df['_newpath'] = df.apply(_get_new_path, axis=1)

            if '_newname' not in self.df.columns:
                self.df['_newname'] = pd.NA
            if '_newpath' not in self.df.columns:
                self.df['_newpath'] = pd.NA

            df = df.reset_index().set_index(self._index)

            self.df.loc[self.df.index.isin(df.index), ['_newname', '_newpath']] = df[['_newname', '_newpath']]
            self.df.loc[self.df.index.isin(df.index), ['_CHANGED']] = True



    def create_symlinks(self, destination):
        def _symlink(src_file, destination):
            dst_file = Path(destination, Path(src_file).name)
            os.symlink(src_file, dst_file)
        self.df.index.map()

    def fill_missing_sessions(self):
        missing = self.df.SESSION.isna()
        self.df['SESSION'] = self.df['_NIGHT']
        self._update_changed_vec(mask=missing)

