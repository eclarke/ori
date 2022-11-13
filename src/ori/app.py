from __future__ import annotations
from pathlib import Path
import textwrap
import subprocess
import logging
import argparse

from tqdm import tqdm
import pandas as pd
import questionary as q
from astropy.io import fits

from .cache import Cache
from .index import Index

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


def parse_fits_headers(files):
    rows, skipped = list(), list()
    for f in tqdm(files, unit = " files", desc="Parsing headers"):
        try:
            header = fits.getheader(f)
            for k, v in header.items():
                rows.append((str(f), str(k), str(v)))
        except OSError as e:
            skipped.append(f)
    return rows, skipped

def prompt_destination(default, msg = 'Destination'):
    dest = q.path(msg, default=default, only_directories=True).ask()
    if not Path(dest).exists():
        if q.confirm(f"{dest} does not exist, create it?", default=False).ask():
            Path(dest).mkdir(parents=True, exist_ok=True)
        else:
            print("Task aborted.")
            return None
    return dest


class App(Index):  

    def attrs_as_choices(self, defaults = list()):
        return [q.Choice(attr, checked = attr in defaults) for attr in self.all_attrs(defaults)]


    def attr_values_as_choices(self, attr):
        return [
            values | {'choice': q.Choice(
                title = "{value} ({ncombos} combinations of {nfiles} files)".format(**values), 
                value = values['value']
            )} for values in self.values_for_attr(attr)
        ]        

    """
    Prompts user to select the set of attributes that will be used to browse the index
    """
    def iselect_attrs(self):
        choices = self.attrs_as_choices(defaults = self.selected_attrs)
        self.selected_attrs = q.checkbox("Select attributes:", choices = choices).ask()
        return self

    """
    Prompts user to refine selection (i.e. select narrower combinations of values for attributes with multiple values selected)
    or clear selected values and start over. 
    """
    def iselect_values(self):
        if not all(self.rowmask):

            option = q.select(
                "What would you like to do with existing selection?", 
                choices=['Refine', 'Invert', 'Discard'], 
                default='Refine'
            ).ask()

            if option == 'Discard':
                self.reset_rowmask()
            elif option == 'Invert':
                self.rowmask = [not _ for _ in self.rowmask]
            elif option is None:
                return
        
        self.stash_rowmask()
        
        for attr in self.selected_attrs:
            choices = self.attr_values_as_choices(attr)
            if len(choices) == 0:
                raise ValueError(f"{values}, {attr}")

            # Sort by the number of distinct combos, unless it's the date attribute
            if attr in ['SESSION']:
                choices = sorted(choices, key = lambda x: x['value'], reverse = True)
            else:
                choices = sorted(choices, key = lambda x: x['ncombos'], reverse = True)

            # Auto-select the first choice
            choices[0]['choice'].checked = True

            prompt = q.checkbox(f"  {attr}:", choices = [x['choice'] for x in choices])
            values = [x['value'] for x in choices]
            selected_values = prompt.skip_if(len(values) == 1, default = values).ask()
            while len(selected_values) == 0:
                print("Error: at least one option needs to be selected with [space]")
                selected_values = prompt.ask()
            value_mask = [value in selected_values for value in self.selection[attr]]
            if self._missing_label in selected_values:
                value_mask = value_mask | self.selection[attr].isna()
            self.selection = self.selection[value_mask]
        return self        


    """
    Shows a submenu with different ways of showing info about the selection.
    """
    def info_submenu(self):
        
        counts = lambda: print(self.summary(missing_label='~'))
        files = lambda: [print(file) for file in self.files()]
        changed = lambda: [print(file) for file in self.changed().index]

        while True:
            task = q.select("Show:", choices = [
                q.Choice("Summary of selected attributes", value=counts),
                q.Choice("Paths of selected files", value=files),
                q.Choice("Paths of files with unsaved changes", value=changed),
                q.Choice("< Go back", value = 'quit')
            ]).ask()

            if task is None or task == 'quit':
                break
            else:
                with pd.option_context('display.max_rows', None):
                    task()

    def ifind_problem_files(self):
        problem_files = []
        attr_list = []
        for imgtype, attrs, problem_df in self.problem_files():
            if len(problem_df) > 0:
                with pd.option_context('display.max_rows', None):
                    print(f"{imgtype.upper()} FRAMES: ")
                    print(problem_df.value_counts().to_frame("Files"))
                    problem_files = problem_files + problem_df.index.tolist()
                    attr_list = attr_list + attrs
        if len(problem_files) > 0:
            print(f"{len(problem_files)} problem files found in current selection.")
            if q.confirm("Refine current selection to only these files?").ask():
                self.rowmask = self.df.index.isin(problem_files)
                self.selected_attrs = attr_list

    """
    Identifies calibration frames that are needed for the selected files. 
    If calibration frames with those characteristics are found in the index, the user can
    add or replace the selection with those files.
    """
    def ifind_calibration_frames(self):
        calib_data = self.config['calibration']
        calib_type = q.select("Calibration type:", choices = calib_data.keys()).ask()
        calib_frames = self.calibration_frames(calib_type)
        c_attrs = calib_data[calib_type]['attrs']
        print(calib_frames.value_counts(subset = c_attrs + ['_STATUS']))
        available = calib_frames.query("_STATUS == 'Available'")
        def _add(): 
            self.stash_rowmask()
            self.rowmask = self.rowmask | self.df.index.isin(available.index)
        def _replace(): 
            self.stash_rowmask()
            self.rowmask = self.df.index.isin(available.index)
        def _nothing(): pass
        if len(available) > 0:
            choice = q.select("Viable calibration frames found. What do you want to do with them?", choices = [
                q.Choice("Add to selection", value = _add),
                q.Choice("Replace selection", value = _replace),
                q.Choice("Do nothing", value = _nothing)
            ]).ask()
            choice()
        return self

    def ifind_calib_frames_fuzzy(self):
        calib_data = self.config['calibration']
        calib_type = q.select("Calibration type:", choices = calib_data.keys()).ask()
        c_attrs = calib_data[calib_type]['attrs']
        exp_tolerance = 2
        temp_tolerance = 1
        def isint(x):
            try:
                int(x)
                return True
            except:
                return False
        int_validator = q.Validator.from_callable(isint, error_message="Not a valid integer")
        if 'EXPTIME' in c_attrs:
            exp_tolerance = q.text(
                "Match EXPTIME to this many digits  [1 = 0.1, 0 = 1, -1 = 10, etc]:", 
                default="2", validate=int_validator).ask()
        if '_CCDTEMP' in c_attrs:
            temp_tolerance = q.text(
                "Match CCD-TEMP to this many digits [1 = 0.1, 0 = 1, -1 = 10, etc]:", 
                default="0", validate=int_validator).ask()
        calib_frames = self.calibration_frames2(calib_type, int(exp_tolerance), int(temp_tolerance))
        counts = calib_frames.value_counts(subset = c_attrs + ['_STATUS']).to_frame("files").reset_index()
        counts.loc[counts._STATUS == "Missing", ['files']] = pd.NA
        with pd.option_context('display.max_rows', None):
            print(counts.set_index(c_attrs + ['_STATUS']).fillna('~'))

        available = calib_frames.query("_STATUS == 'Available'")
        def _add(): 
            self.stash_rowmask()
            self.rowmask = self.rowmask | self.df.index.isin(available.index)
        def _replace(): 
            self.stash_rowmask()
            self.rowmask = self.df.index.isin(available.index)
        def _nothing(): pass
        if len(available) > 0:
            choice = q.select("Viable calibration frames found. What do you want to do with them?", choices = [
                q.Choice("Add to selection", value = _add),
                q.Choice("Replace selection", value = _replace),
                q.Choice("Do nothing", value = _nothing)
            ]).ask()
            choice()
    """
    Allows the user to change the values of specific attributes
    """
    def ichange_attr(self):
        choices = self.attrs_as_choices()
        choices = [c for c in choices if c.value in self.config['mutable']]        
        selected_attr = q.select("Select an attribute to modify:", choices = choices).ask()
        choices = [c['choice'] for c in self.attr_values_as_choices(selected_attr)]
        for i, c in enumerate(choices):
            if c.value == self._missing_label:
                c.disabled = "N/A"
                choices[i] = c
        if len(choices) == 1:
            choices[0].disabled = "Existing value for all files"
        choices.append(q.Choice("[Set New Value]", value = '_new'))
        chosen_value = q.select("Coalesce around existing value or set new value:", choices=choices).ask()
        if chosen_value == '_new':
            validator = lambda val: len(val) <= 70
            choices = self.df[selected_attr].dropna().unique().tolist()
            msg = "Enter new value (max 70 chars), or leave empty to set to NA:"
            if len(choices) > 0:
                print(f"Values found in other files: {choices}")
                question = q.autocomplete(msg, choices = choices, validate=validator)
            else:
                question = q.text(msg, validate=validator)
            chosen_value = question.ask()
            while chosen_value not in choices and chosen_value is not None:
                if chosen_value == "":
                    chosen_value = pd.NA
                    confirm = q.confirm("Are you sure you want to remove all the values for this attribute?", default=False)
                else:
                    confirm = q.confirm("Entered value does not match any values found in other files, proceed anyway?", default=False)
                if confirm.ask():
                    break
                else:
                    chosen_value = question.ask()
        if chosen_value is not None:
            self._change_attr(selected_attr, chosen_value)
        return self
    

    def ichange_folders(self):
        if not all(pd.isna(self.df._newpath)):
            print("Error: previous file move/copy requested but not synced. Save changes and try again.")
            return
        destination = prompt_destination(
            default = str(Path(self._root_dir).parent),
            msg = "New top-level directory:"
        )

        if destination is not None:
            self.change_subtree(destination)
            self.selected_attrs = self.selected_attrs + ['_newpath']


    def iorganize(self):
        if not all(pd.isna(self.df._newpath)):
            print("Error: previous file move/copy requested but not synced. Save changes and try again.")
            return
        _root = prompt_destination(default = str(Path(self._root_dir).parent), msg = "New top-level directory:")
        if _root is not None:
            _root = Path(_root).resolve()
            self.organize(_root)
            self.selected_attrs = self.selected_attrs + ['_newname', '_newpath']
        print("New directory structure and filenames prepared. Save changes to move/copy files on disk.")
            

    def iorganize_siril(self):
        if not all(pd.isna(self.df._newpath)):
            print("Error: previous file move/copy requested but not synced. Save changes and try again.")
            return
        print("Warning: the directory structure for Siril assumes you've already standardized the binning, gain, offset and ccd-temp for your selection. These attributes will not be reflected in the directory structure or file names.")
        _root = prompt_destination(default = str(Path(self._root_dir).parent), msg = "New top-level directory:")
        if _root is not None:
            _root = Path(_root).resolve()
            self.organize(_root, structure_key='siril')
            self.selected_attrs = self.selected_attrs + ['_newname', '_newpath']
        print("New directory structure and filenames prepared. Save changes to move/copy files on disk.")


    def ifill_sessions(self):
        if q.confirm("Some headers do not have a SESSION keyword. Would you like to fill it in with an autocomputed value?").ask():
            self.fill_missing_sessions()

    def iuse_prev_rowmask(self):
        n_prev = sum(self._prev_rowmasks[-1])
        if q.confirm(f"Use prior selection of {n_prev} files? Current selection will be lost.").ask():
            self._rowmask = self._prev_rowmasks.pop()
            


    def isave_changes(self):

        if any(~pd.isna(self.df._newpath)) or any(~pd.isna(self.df._newname)):
            overwrite = False
            save_oldname = False
            op = q.select("Paths have changed. How do you want to save these changes?", choices = [
                q.Choice("[Copy] files to the new location, then apply any changes to the copy", value = 'copy'),
                q.Choice("[Move] files to the new location, then apply any changes", value = 'move'),
                q.Choice("[Symlink] files to new location. NOTE: Any attribute changes will not be applied unless separately saved to the original file.", value='symlink')
            ]).ask()
            if op in ['move', 'copy']:
                overwrite = q.confirm("Overwrite existing files, if they exist?", default=False).ask()
                if any(~pd.isna(self.df._newname)):
                    save_oldname = q.confirm("Save record of old filenames as a HISTORY entry?", default=False).ask()
            elif op is None:
                print("Save canceled")
                return

            self.sync(op, overwrite, save_oldname)
        else:
            print("\n{:#^70}".format(" WARNING "))
            print(textwrap.fill(
                "File attributes have been changed, but paths have not. "
                "If you save now, you will permanently modify the original files. "
                "If this is undesired, abort and choose [Move...] or [Organize...] from the main menu."
            ))
            print("{:#^70}\n".format(""))
            if q.confirm(
                "{: ^70}".format("Are you sure you want to modify the original files?"), 
                default=False, qmark='').ask():
                
                self.sync()


    def loop(self):

        while True:
            n_selected = len(self.selection)
            n_changed = len(self.changed())
            has_prev_rowmask = len(self._prev_rowmasks) > 0

            task_choices = [
                q.Separator('{:-<30}'.format('')), 
                {'name': 'View selection', 'value': self.info_submenu},               
                q.Separator('{:-<30}'.format('')),
                {'name': 'Find calibration frames', 'value': self.ifind_calibration_frames},
                {'name': 'Fuzzy-find calibration frames', 'value': self.ifind_calib_frames_fuzzy},                
                {'name': 'Find problem files', 'value': self.ifind_problem_files},
                {'name': 'Select different files', 'value': self.iselect_values},
                {'name': 'Select different attributes', 'value': self.iselect_attrs},                
                q.Separator('{:-<30}'.format('')),
                {'name': 'Modify an attribute', 'value': self.ichange_attr},
                {'name': 'Move selected files to new directory', 'value': self.ichange_folders},
                {'name': 'Organize folders/filenames', 'value': self.iorganize},
                {'name': 'Organize into Siril-friendly structure', 'value': self.iorganize_siril},
                {'name': 'Fill missing SESSION values', 'value': self.ifill_sessions, 'disabled': 'no missing values' if not any(self.df.SESSION.isna()) else None},
                q.Separator('{:-<30}'.format('')),
                {'name': 'Use prior selection', 'value': self.iuse_prev_rowmask, 'disabled': 'no prior selection' if not has_prev_rowmask else None},
                {'name': 'Save changes', 'value': self.isave_changes, 'disabled': 'no unsaved changes' if n_changed == 0 else None},
                {'name': 'Exit', 'value': exit}
            ]

            if n_changed > 0:
                message = f"{n_selected} files selected. {n_changed} files' attributes have unsaved changes.\nNext task:"
            else:
                message = f"{n_selected} files selected.\nNext task:"
            if n_selected == 0:
                self.rowmask = [True for _ in self.df.index]
                self.iselect_values()
            else:
                task_select = [{
                    'type': 'select',
                    'message': message,
                    'name': 'task',
                    'use_shortcuts': True,
                    'choices': task_choices
                }]

                task = q.prompt(task_select)
                if task is None:
                    break
                task['task']()


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'target_dir',
        help="Directory to recursively search for *.fit* files"
    )

    parser.add_argument(
        '-d', '--days_old', metavar="N",
        help="Only look for files created/modified in the last N days",
        type=int
    )

    parser.add_argument(
        "--reset",
        help = "Resets cache for this directory", 
        action="store_true"
    )

    parser.add_argument(
        "--allow_siril",
        help = "Don't exclude Siril preprocessing prefixes (pp_*, r_*)",
        action = "store_true"
    )

    parser.add_argument(
        "--skip_parsing", "-s",
        help = "Don't re-scan directory, just use the db",
        action = "store_true"
    )

    return parser.parse_args()


def run_app(target_dir: str, days_old: int, reset_cache: bool, allow_siril: bool):

    target_dir = Path(target_dir).resolve()
    cache = Cache(target_dir, reset = reset_cache)
    print("Scanning for FITS files...")
    files = find_fits(target_dir, days = days_old, allow_siril = allow_siril)

    _headers, _skipped = cache.get()

    cached_files = set([f for f, _, _ in _headers] + [f for f, in _skipped])
    files_not_in_cache = [f for f in files if f not in cached_files]
    if len(files_not_in_cache) == 0:
        print("Cache hit")
        headers = [h for h in _headers if h[0] in files]
    else:
        print("Cache miss")
        headers, skipped = parse_fits_headers(files)
        
        print("Caching headers")
        cache.set(headers, skipped)

    if len(headers) == 0:
        logging.error("No valid FITS files found")
        exit()

    fidx = App(headers, target_dir)
    fidx.iselect_attrs().iselect_values()
    if any(fidx.df._CHANGED):
        print("Default values for some attributes (e.g., SESSION) have been autofilled.")
    fidx.loop()

