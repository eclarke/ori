from pathlib import Path
from datetime import datetime
import logging

from sqlalchemy import (
    create_engine, 
    Integer,
    Float, 
    String, 
    Column, 
    ForeignKey, 
    DateTime,
    Table,
    select,
    func
)
from sqlalchemy.orm import (
    declarative_base, 
    Session,  
    column_property,
    declared_attr,
    deferred
)
from sqlalchemy.ext.declarative import DeferredReflection

import pandas as pd

from ori.config import get_config
from ori.app import parse_cli_args
from ori.utils import find_fits, parse_files

Base = declarative_base()

class Reflected(DeferredReflection):
    __abstract__ = True


class Header(Reflected, Base):
    __tablename__ = "header"
    id = Column(Integer, primary_key = True)
    date_obs = Column(DateTime)

    @declared_attr
    def localdate(cls):
        return column_property(select(func.date(cls.date_obs)))

class Change(Base):
    __tablename__ = "change"
    id = Column(Integer, primary_key = True)
    source_id = Column(ForeignKey("header.id"))
    key = Column(String(10), nullable = False)
    oldvalue = Column(String)
    newvalue = Column(String)


class SqlIndex(object):

    base = Base

    @staticmethod
    def df_to_sql(df: pd.DataFrame, tbl_name, engine, if_exists):
        df.to_sql(name = tbl_name, con = engine, if_exists = if_exists, index=False)
    
    """
    Create a new, SQLite-based FITS index.
    :param: target_dir the root directory
    :param: headers_df a dataframe of headers parsed from the target_dir
    """
    def __init__(self, target_dir, **kwargs):
        
        self.engine = create_engine(f"sqlite+pysqlite:///{Path(target_dir, 'index.db')}", future=True)
        self.header_tbl = None
        self.change_tbl = None
        self.missing_label = kwargs.get('missing_label', '(Unknown)')
        self.target_dir = target_dir

        header_df = kwargs.get('header_df')
        if header_df is not None:
            print("inserting rows into db")
            if_exists = kwargs.get('if_exists', 'fail')
            print(Header.__table__.name)
            SqlIndex.df_to_sql(df = header_df, tbl_name = Header.__table__.name, engine = self.engine, if_exists=if_exists)

        Reflected.prepare(self.engine)




if __name__ == "__main__":
    args = parse_cli_args()
    target_dir = Path(args.target_dir).resolve()
    if args.skip_parsing:
        index = SqlIndex(target_dir)
    else:
        files = find_fits(target_dir, days = args.days_old, allow_siril=args.allow_siril)
        header_dict, skipped = parse_files(files)
        header_df = pd.DataFrame(header_dict)
        index = SqlIndex(target_dir, header_df = header_df)

    with Session(index.engine) as session:
        stmt = select(Header).where(Header.localdate == func.min(Header.localdate)).group_by()
        header, = session.execute(stmt).fetchone()
        print(f'{header.id}: {header.name}')