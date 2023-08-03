import logging
import sys
import time
from ast import literal_eval

import pandas
import requests
import requests_cache
from requests_cache import NEVER_EXPIRE, CachedSession


PUBMED_TRIAL_PROTOCOL = "(clinicaltrial[Filter] NOT editorial)"

SENSITIVE_TRIAL_PROTOCOL = (
    "(randomized controlled trial [pt] OR "
    "controlled clinical trial [pt] OR "
    "randomized [tiab] OR placebo [tiab] OR "
    "drug therapy [sh] OR randomly [tiab] OR "
    "trial [tiab] OR groups [tiab]) "
    "NOT (animals [mh] NOT humans [mh])"
)

SPECIFIC_TRIAL_PROTOCOL = (
    "(randomized controlled trial [pt] OR "
    "controlled clinical trial [pt] OR "
    "randomized [tiab] OR placebo [tiab] OR "
    "clinical trials as topic [mesh: noexp] OR randomly [tiab] OR "
    "trial [ti]) "
    "NOT (animals [mh] NOT humans [mh])"
)


def load_trials(name, unique_id, drop_duplicates=True, require_source=True):
    # NOTE: the name field can be either a sponsor or a site
    if require_source:
        necessary_columns = unique_id + ["source"]
    else:
        necessary_columns = unique_id
    try:
        trials = pandas.read_csv(name)
    except FileNotFoundError:
        return pandas.DataFrame(columns=necessary_columns)
    missing_columns = set(necessary_columns) - set(trials.columns)
    if len(missing_columns) > 0:
        print(",".join(missing_columns) + " must be columns in the input file")
        sys.exit(1)
    # Drop any unnamed columns
    # We do this rather than having to know whether the file contains an index on load
    trials = trials.drop(
        trials.columns[trials.columns.str.contains("unnamed", case=False)], axis=1
    )
    return trials


def filter_unindexed(a, b, unique_id):
    """
    Given two unindexed dataframes, remove rows in b from a, as matched by unique_id
    """
    a_indexed = a.set_index(unique_id)
    remaining = a_indexed.loc[
        (a_indexed.index).difference(b.set_index(unique_id).index)
    ]
    return remaining.reset_index()


def create_session(name="general_cache", use_cache=False):
    if use_cache:
        requests_cache.install_cache(name, backend="sqlite")
        # Do not expire the cache
        session = CachedSession(expire_after=NEVER_EXPIRE)
    else:
        session = requests.Session()
    return session


def query(url, session, params={}, retries=2):
    """
    Given a name, query the ror api
    """
    if retries > 0:
        try:
            response = session.get(url, params=params)
            logging.info(f"{url} {response}")
            response.raise_for_status()
            print(response.__dict__)
            return response.json()
        except Exception as e:
            logging.error(f"Failed to download {e}, retries left {retries}")
            time.sleep(60)
            return query(
                url,
                session,
                retries=retries - 1,
            )
    else:
        raise requests.exceptions.RequestException("Failed to download after retries")


def expand_sites(df):
    # One row per site
    assert not df.empty
    df["sites"] = df.apply(lambda x: literal_eval(x["sites"]), axis=1)
    exploded = df.explode("sites")
    if isinstance(exploded.sites.iloc[0], dict):
        sites = pandas.json_normalize(exploded.sites)
        return exploded[["trial_id"]].join(sites)
    return exploded


def preprocess_trial_file(filepath, source):
    output_dir = filepath.parent
    prefix = filepath.stem

    df = pandas.read_csv(filepath, index_col=[0])
    if source == "cris":
        df = df.rename(columns={"cris_sites": "sites"})
    elif source == "ctri":
        # Raw data, unhelpful while unprocessed
        df = df.drop("spon_address", axis=1)
    elif source == "isrctn":
        df = df.rename(
            columns={
                "rec_sites": "sites",
                "spon_city": "city",
                "spon_country": "country",
            }
        )
        # Captured in the site dictionary
        df = df.drop("rec_countries", axis=1)
    elif source == "pactr":
        df = df.rename(columns={"spon_country": "country"})
        # Raw data, unhelpful while unprocessed
        df = df.drop("spon_city", axis=1)

    df["source"] = source
    df = df.rename(columns={"sponsor": "name"})
    # Save the sponsor file
    df.drop("sites", axis=1).to_csv(output_dir / f"{prefix}_sponsor.csv")
    sites = expand_sites(df[["trial_id", "sites"]])
    sites["source"] = source
    sites.to_csv(output_dir / f"{prefix}_sites.csv")


"""
Cleaning registry data
Extract India address state (capitalized): sites.address.str.extract(r'([A-Z ]+$)')
Make sure they are both strings
updated = sites.apply(lambda x: x["address"].replace(x["state"], ""), axis=1)
Extract India address city: updated.str.extract(r'([A-Z][A-Za-z]*)$')[0].str.extract("([A-Z][^A-Z]*)")
OR
Get list of India cities and extract the last one; can do this for sponsor do
Could maybe do this for pactr address
"""
