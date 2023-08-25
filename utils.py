import difflib
import json
import logging
import re
import sys
import time
from ast import literal_eval

import country_converter as coco
import numpy
import pandas
import requests
import requests_cache
from requests_cache import NEVER_EXPIRE, CachedSession
from unidecode import unidecode


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


def latin_to_utf8(string):
    try:
        fixed = string.encode("latin-1").decode("utf-8")
        if fixed != string:
            return fixed
        else:
            return None
    except Exception:
        return None


def convert_country(country_column, to="ISO2"):
    """
    Standardise to a country code
    """
    cc = coco.CountryConverter()
    # Initial conversion
    out = cc.pandas_convert(country_column, to=to, not_found=numpy.nan)
    missing = country_column[
        out.isnull() & country_column.notnull() & (country_column != "")
    ]
    logging.info(f"Country conversion level 1: {len(missing)} missing")

    # Try utf 8 encoding (TÃ¼rkiye)
    fixed = cc.pandas_convert(
        missing.apply(lambda x: latin_to_utf8(x)).dropna(),
        to=to,
        not_found=numpy.nan,
    )
    out.loc[fixed.index] = fixed
    missing = country_column[
        out.isnull() & country_column.notnull() & (country_column != "")
    ]
    logging.info(f"Country conversion level 2: {len(missing)} missing")

    # Use provided mapper (Wales --> UK; KSA --> Saudia Arabia)
    fixed2 = cc.pandas_convert(
        map_country(missing).dropna(), to=to, not_found=numpy.nan
    )
    out.loc[fixed2.index] = fixed2
    missing = country_column[
        out.isnull() & country_column.notnull() & (country_column != "")
    ]
    logging.info(f"Country conversion level 3: {len(missing)} missing")

    # Allow for letter switching and double letters i.e. Thailannd, Taiwian
    fixed3 = cc.pandas_convert(
        missing.apply(lambda x: find_close_country(x)).dropna(),
        to=to,
        not_found=numpy.nan,
    )
    out.loc[fixed3.index] = fixed3
    missing = country_column[
        out.isnull() & country_column.notnull() & (country_column != "")
    ]
    logging.info(f"Country conversion level 4: {len(missing)} missing")
    logging.warn(f"Failed to convert the countries: {','.join(missing.values)}")
    # Replace non-null values that failed to convert i.e. European Union
    out.loc[missing.index] = missing
    return out


def extract_before_regex_ignorecase(line, criteria):
    """
    Extract the string before any of the criteria, ignoring case
    """
    loc = re.search(criteria, line, flags=re.IGNORECASE)
    if loc:
        return line[0 : loc.start()].strip()  # noqa: E203

    else:
        return line


def remove_noninformative(name):
    """
    The following strings were commonly the entire name, in which case
    we do not want to try to match. If they are part of the string they may
    hamper the ror affiliation matching.

    We do not use split because we want to use a regex and ingnorecase
    We do not use name.str.extract because we are trying to extract text
    BEFORE any of the criteria (.*?)(criteria) would give us more than one
    match group

    There were instances of both Investigative Site and investigative Site
    """
    non_informative = [
        "study center",
        "research site",
        "study site",
        "clinical trial site",
        "clinical research site",
        "local institution",
        "clinical study site",
        "investigator site",
        "investigative site",
        "investigational site",
        "clinical site",
        "medical facility",
        "\\(site",
        "/id#",
    ]
    criteria = "(" + "|".join(non_informative) + ")"
    name = name.apply(
        lambda x: extract_before_regex_ignorecase(str(x), criteria) if x == x else x
    ).replace("", None)
    return name


def clean_trials(trials):
    if "name" in trials.columns:
        # Remove non-informative
        trials["name"] = remove_noninformative(trials.name)

    # Use ISO2 as per the ror schema
    # https://ror.readme.io/docs/all-ror-fields-and-sub-fields
    if "country" in trials.columns:
        trials.loc[:, "country"] = convert_country(trials.country)

    # Drop duplicates AFTER we have done some cleaning
    trials = trials.drop_duplicates(
        subset=trials.columns.difference(["site_country_list"])
    )

    return trials


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
    # We do this rather than having to know whether the file contains an index
    # on load
    trials = trials.drop(
        trials.columns[trials.columns.str.contains("unnamed", case=False)],
        axis=1,
    )
    return trials


def filter_unindexed(a, b, unique_id):
    """
    Given two unindexed dataframes
    remove rows in b from a, as matched by unique_id
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
            return response.json()
        except Exception as e:
            if e.response.status_code == 500:
                # Pass back to caller
                raise e
            logging.error(f"Failed to download {e}, retries left {retries}")
            time.sleep(20)
            return query(
                url,
                session,
                retries=retries - 1,
            )
    else:
        raise requests.exceptions.RequestException("Failed to download after retries")


def split_country_list(site_country_list):
    non_null = site_country_list.dropna()
    try:
        non_null = non_null.apply(
            lambda x: literal_eval(x).split(";") if isinstance(x, str) else x
        )
    except Exception:
        pass

    exploded = non_null.explode().reset_index()
    countries = convert_country(exploded["site_country_list"])
    exploded["site_country_list"] = countries
    grouped = exploded.groupby("index").site_country_list.apply(list)
    site_country_list.loc[grouped.index] = grouped
    site_country_list.loc[site_country_list.isnull()] = site_country_list.loc[
        site_country_list.isnull()
    ].apply(lambda x: [])
    return site_country_list


def expand_sites(df, set_country=None, site_column_name="sites"):
    # One row per site
    assert not df.empty
    df.loc[:, "sites"] = df.apply(
        lambda x: literal_eval(x["sites"])
        if isinstance(x["sites"], str)
        else x["sites"],
        axis=1,
    )
    if isinstance(df["sites"].iloc[0], list):
        df = df.explode("sites").reset_index(drop=True)
    if isinstance(df.sites.iloc[0], dict):
        normalized = pandas.json_normalize(df.sites)
        df = df[["trial_id"]].join(normalized)

    if "site_country_list" in df.columns:
        df["site_country_list"] = split_country_list(df.site_country_list)
        df["country"] = df.site_country_list.apply(
            lambda x: x[0] if len(x) == 1 else None
        )

    if set_country:
        df["country"] = df.apply(lambda x: set_country if x["name"] else "", axis=1)
    return df


def find_city_country(
    df, column, country_name, remove_filename=None, use_alternatives=True
):
    """
    Given the country, find the city
    """

    # Add space before capital letter (so we can ignore case)
    df[column] = df[column].apply(
        lambda x: re.sub(r"(?<=[a-z0-9\-])(?=[A-Z])", " ", str(x))
    )

    iso2 = coco.convert(country_name, to="ISO2")
    all_cities = pandas.read_csv(
        "geonames-all-cities-with-a-population-1000.csv", delimiter=";"
    )
    country_cities = all_cities[all_cities["Country Code"] == iso2]
    names = country_cities["Name"].apply(lambda x: unidecode(str(x))).drop_duplicates()

    if use_alternatives:
        alt = country_cities.dropna(subset=["Alternate Names"])
        # The series has comma-delimited rows, join into a single series
        alt.loc[:, "alt_name"] = alt["Alternate Names"].str.split(",")
        exploded = alt.explode("alt_name")
        # Lower case the map key so we can do case-insensitive mapping
        exploded["alt_name"] = (
            exploded["alt_name"].apply(lambda x: unidecode(str(x))).str.lower()
        )
        exploded["Name"] = exploded["Name"].apply(lambda x: unidecode(str(x)))
        # Because we are unidecoding everything
        # we have duplicates (index and value)
        exploded = exploded.drop_duplicates(subset=["Name", "alt_name"])
        # And potentially empty strings after decoding
        exploded = exploded[exploded.alt_name != ""]

        alt_map = pandas.Series(exploded["Name"].values, index=exploded["alt_name"])

        all_names = list(names) + list(alt_map.keys())
    else:
        all_names = list(names)
    criteria = r"\b(" + "|".join([re.escape(x) for x in all_names]) + r")\b"
    # assert("" not in criteria)

    # We need to combine the main list and alternatives
    # otherwise we will not know which one was last
    # But this makes things much slower
    all_matches = (
        df[column]
        .apply(lambda x: unidecode(str(x)))
        .str.extractall(criteria, flags=re.IGNORECASE)
    )
    if remove_filename:
        with open(remove_filename) as f:
            to_remove = f.read().splitlines()
        all_matches = all_matches[
            ~all_matches[0].str.lower().isin([item.lower() for item in to_remove])
        ]
    matched = (
        all_matches.groupby(level=0)
        .tail(1)
        .reset_index()
        .drop("match", axis=1)
        .set_index("level_0")
    )
    df["city"] = matched[0].str.title()
    if use_alternatives:
        df["city"].update(
            matched[~matched[0].str.lower().isin(names.str.lower())][0]
            .str.lower()
            .map(alt_map.to_dict())
        )

    df.loc[(df.city.notnull()), "country"] = country_name
    return df


def find_close_country(name):
    with open("countries.json") as f:
        countries = json.load(f)
    country_list = list(set(countries.values()))
    matching_countries = difflib.get_close_matches(name, country_list)
    if matching_countries:
        confidence = difflib.SequenceMatcher(None, matching_countries[0], name).ratio()
        if confidence > 0.8:
            return matching_countries[0]
    else:
        return None


def map_country(country_column):
    with open("countries.json") as f:
        countries = json.load(f)
    country_map = pandas.Series(countries)

    return country_column.map(country_map)


def find_country(df, other_column="address"):
    """
    other_column is a column that has more information i.e. an address
    """
    missing_country = df[df.country.isnull()]
    with open("countries.json") as f:
        countries = json.load(f)
    country_list = (
        r"\b("
        + "|".join(
            [
                rf"{re.escape(x)}"
                for x in sorted(list(countries.keys()), key=len, reverse=True)
            ]
        )
        + r")\b"
    )
    mapped = (
        missing_country[other_column]
        .apply(lambda x: unidecode(str(x)))
        .str.extract(country_list, flags=re.IGNORECASE)[0]
        .map(countries)
    )
    mapped = convert_country(mapped, to="ISO2")
    df.loc[mapped.index, "country"] = mapped
    return df


def preprocess_trial_file(args):
    filepath = args.input_file
    source = args.source
    output_dir = filepath.parent

    set_country = None
    site_column_name = "sites"

    df = pandas.read_csv(filepath, index_col=[0])
    if source == "cris":
        df = df.rename(columns={"cris_sites": "sites"})
        set_country = "KR"
    elif source == "ctri":
        df = df.rename(columns={"spon_address": "address", "spon_type": "type"})
        df = find_city_country(df, "address", "IN", "india_states.txt")
        df = find_country(df)
    elif source == "drks":
        df = df.rename(
            columns={
                "spon_country": "country",
                "spon_city": "city",
                "Countries": "site_country_list",
            }
        )
        # TODO: site data prefixed with "Medical center"
        # "University medical center" "Doctor's office", or "Other"
        # Pull out the type
    elif source == "isrctn":
        df = df.rename(
            columns={
                "rec_sites": "sites",
                "spon_city": "city",
                "spon_country": "country",
                "ror_id": "ror",
            }
        )
        # Any row that has a ror id, set the name resolved
        # Captured in the site dictionary
        df = df.drop("rec_countries", axis=1)
    elif source == "pactr":
        df = df.rename(columns={"spon_country": "country", "spon_type": "type"})
        # Raw data, unhelpful while unprocessed
        df = df.drop("spon_city", axis=1)
    elif source == "lbctr":
        df = df.rename(
            columns={
                "sponsor_country": "country",
                "Countries": "site_country_list",
            }
        )
        # There are more countries than sites; all listed sites are in Lebanon
        set_country = "LB"
    elif source == "anzctr":
        df = df.rename(
            columns={
                "spon_name": "sponsor",
                "spon_country": "country",
                "spon_type": "type",
                "aus_sites": "sites",
                "Countries": "site_country_list",
            }
        )
        # There are more countries than sites; all listed sites are AUS or NZ
        set_country = "AU"
    elif source == "irct":
        df = df.rename(columns={"spon_city": "city"})
        set_country = "IR"
    elif source == "tctr":
        pass
    elif source == "repec":
        site_column_name = "institution"
        set_country = "PE"
    elif source == "rpcec":
        df = df.rename(columns={"countries": "site_country_list"})
    elif source == "euctr":
        # We need to expand the "{name country type}" or "{sponsor address}
        # dictionaries into dataframe rows
        df["sponsor"] = df["sponsor"].apply(lambda x: literal_eval(x))
        expanded = df["sponsor"].apply(pandas.Series)
        # Take name or sponsor
        expanded["sponsor"] = expanded.sponsor.fillna(expanded.name)
        expanded = expanded.drop("name", axis=1)
        # We need to try to find the city/country from the address
        with_country = find_country(expanded)
        df = df.drop("sponsor", axis=1).join(with_country)
        df["sites"] = df.countries.apply(
            lambda x: [{"country": c} for c in literal_eval(x)]
        )
    elif source == "ctgov":
        df = df.rename(
            columns={"LeadSponsorName": "sponsor", "LeadSponsorClass": "type"}
        )
        df = df.rename(
            columns={
                "NCTId": "trial_id",
                "LocationFacility": "sites",
                "LocationCity": "city",
                "LocationCountry": "country",
            }
        )

    df["source"] = source
    if "sponsor" in df.columns:
        df = df.rename(columns={"sponsor": "name"})
        # The columns we want to keep for the sponsor
        subset = df[
            df.columns[
                df.columns.isin(
                    ["trial_id", "name", "type", "country", "city", "source", "ror"]
                )
            ]
        ]
        subset = clean_trials(subset)
        subset.to_csv(output_dir / f"{source}_sponsor.csv")

    # The columns we want to keep for the sites
    if "sites" in df.columns:
        try:
            sites = expand_sites(
                df[
                    df.columns[
                        df.columns.isin(["trial_id", "sites", "site_country_list"])
                    ]
                ],
                set_country=set_country,
                site_column_name=site_column_name,
            )
        except Exception:
            logging.info("Sites failed to expand, assuming already expanded")
            sites = df
        sites = sites.rename(columns={site_column_name: "name"})
        sites["source"] = source
        if source == "ctri":
            sites = find_city_country(sites, "address", "IN")
            sites = find_country(sites)
            sites.loc[sites.country.isnull(), "country"] = "IN"
        elif source == "drks":
            sites = find_city_country(sites, "name", "DE")

        sites = clean_trials(sites)
        sites = sites[sites.name.notnull()].reset_index(drop=True)
        sites[
            sites.columns[
                sites.columns.isin(
                    [
                        "trial_id",
                        "name",
                        "type",
                        "country",
                        "city",
                        "site_country_list",
                        "source",
                    ]
                )
            ]
        ].to_csv(output_dir / f"{source}_sites.csv")
