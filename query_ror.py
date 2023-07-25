import argparse
import logging
import math
import pathlib
import re
import sys
import time
import urllib

import country_converter as coco
import numpy
import pandas
import requests
import requests_cache
from requests_cache import NEVER_EXPIRE, CachedSession


SPECIAL_CHARS_REGEX = r'[\+\-\=\|\>\<\!\(\)\\\{\}\[\]\^"\~\*\?\:\/\.\,\;]'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s [%(levelname)-9s] %(message)s [%(module)s]",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(handler)


def filter_unindexed(a, b, unique_id):
    """
    Given two unindexed dataframes, remove rows in b from a, as matched by unique_id
    """
    a_indexed = a.set_index(unique_id)
    remaining = a_indexed.loc[
        (a_indexed.index).difference(b.set_index(unique_id).index)
    ]
    return remaining.reset_index()


def set_index(trials, unique_id, include_id=True):
    cols = unique_id
    if include_id:
        cols = ["trial_id"] + cols
    return trials.set_index(cols)


def merge_drop_dups(left, right, on, how="left"):
    if on == "index":
        merged = left.merge(
            right,
            right_index=True,
            left_index=True,
            how=how,
            suffixes=("", "_y"),
            indicator=True,
        )
    else:
        merged = left.merge(right, on=on, how=how, suffixes=("", "_y"), indicator=True)
    merged = merged.drop(merged.filter(regex="_y$").columns, axis=1)
    return merged


def create_session(use_cache=False):
    if use_cache:
        requests_cache.install_cache("ror_cache", backend="sqlite")
        # Do not expire the cache
        session = CachedSession(expire_after=NEVER_EXPIRE)
    else:
        session = requests.Session()
    return session


def load_mapping_dict(mapping_dict_path, compare_id):
    mapping_dict = pandas.read_csv(mapping_dict_path, index_col=compare_id)
    # If the index was created with city and country, there may be duplicated entries
    # TODO: what to do with entries that do not have city and country?
    # TODO: maybe do not allow them to be added?
    return mapping_dict.loc[~mapping_dict.index.duplicated()]


def filter_already_mapped(mapping_dict_file, df, output_file, compare_id, unique_id):
    mapping_dict = load_mapping_dict(mapping_dict_file, compare_id)

    merged = merge_drop_dups(df, mapping_dict, compare_id, how="left")

    already_mapped = merged[merged._merge == "both"].drop("_merge", axis=1)
    remaining = merged[merged._merge == "left_only"].drop("_merge", axis=1)

    assert (len(remaining) + len(already_mapped)) == len(df)
    # Nothing that is mapped should have a null ror or name_ror
    assert already_mapped.ror.notnull().all()
    assert already_mapped.name_ror.notnull().all()

    # TODO: make sure this has the same columns as chunked outputs
    already_mapped["method"] = "dict"
    already_mapped["count_non_matches"] = 0
    already_mapped["method_ror"] = None
    already_mapped["non_matches"] = None
    # Sort the columns alphabetically
    already_mapped = already_mapped.sort_index(axis=1)
    already_mapped.to_csv(
        output_file, mode="a", header=not output_file.exists(), index=False
    )

    return remaining


def write_to_mapping_dict(df, mapping_dict_file):
    matches_map = df[df.ror.notnull()][
        ["name", "name_ror", "ror", "city", "country"]
    ].drop_duplicates()
    matches_map = set_index(matches_map, include_id=False)
    if mapping_dict_file.exists():
        mapping_dict = load_mapping_dict(mapping_dict_file)
        joined = pandas.concat([mapping_dict, matches_map])
        new_matches = joined.loc[joined.index.difference(mapping_dict.index)]
    else:
        new_matches = matches_map
    if not new_matches.empty:
        new_matches = new_matches.reset_index()
        assert (
            len(new_matches[(new_matches.ror.notnull()) & (new_matches.name.isnull())])
            == 0
        )
        new_matches.to_csv(
            mapping_dict_file,
            mode="a",
            header=not mapping_dict_file.exists(),
            index=False,
        )


def remove_special_chars(name, replace=" "):
    return re.sub(SPECIAL_CHARS_REGEX, replace, name)


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
        lambda x: extract_before_regex_ignorecase(x, criteria) if x == x else x
    ).replace("", numpy.nan)
    return name


def convert_country(country_column, to="ISO2"):
    """
    Standardise to a country code
    Use ISO2 as per the ror schema
    https://ror.readme.io/docs/all-ror-fields-and-sub-fields
    """
    cc = coco.CountryConverter()
    return cc.pandas_convert(country_column, to=to, not_found=numpy.nan)


def load_trials(name, unique_id, drop_duplicates=True):
    # NOTE: the name field can be either a sponsor or a site
    necessary_columns = unique_id + ["source"]
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

    # Remove non-informative
    trials["name"] = remove_noninformative(trials.name)

    # Convert country to ISO-2
    trials.loc[:, "country"] = convert_country(trials.country)

    # Drop duplicates AFTER we have done some cleaning
    trials = trials.drop_duplicates()

    return trials


def get_empty_results():
    return {
        "name_ror": None,
        "ror": None,
        "method": None,
        "method_ror": None,
        "count_non_matches": 0,
        "non_matches": [],
    }


def process_response(response, results, method="first", method_ror="chosen"):
    results["name_ror"] = response["organization"]["name"]
    results["ror"] = response["organization"]["id"]
    results["method"] = method
    results["method_ror"] = method_ror
    return results


def process_ror_json(name, session, results, extra_data):
    """
    There are two techniques for choosing a match from ror
    1. Use the ror "chosen field"
    2. The highest score above 0.8 among the ror objects with matching
       city/country
    """
    resp = query(name, session)
    # TODO: there could be two chosen
    for item in resp["items"]:
        if item["chosen"]:
            return process_response(item, results)
    potential_matches = []
    if extra_data.get("country") and extra_data.get("city"):
        for item in resp["items"]:
            if (
                item["organization"]["country"]["country_code"]
                == extra_data.get("country")
            ) and (
                item["organization"]["addresses"][0]["city"] == extra_data.get("city")
            ):
                potential_matches.append(item)
    if len(potential_matches) > 0 and potential_matches[0]["score"] > 0.8:
        return process_response(potential_matches[0], results, method="city")
    else:
        # TODO: append these together
        results["count_non_matches"] = len(potential_matches)
        results["non_matches"] = [
            {match["organization"]["name"]: match["score"]}
            for match in potential_matches
        ]
    return results


def query(name, session, retries=2):
    """
    Given a name, query the ror api
    """
    url = (
        f"https://api.ror.org/organizations?affiliation={urllib.parse.quote_plus(name)}"
    )
    if retries > 0:
        try:
            response = session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if e.response.status_code == 403:
                logger.error(f"Failed to download, retries left {retries}")
                time.sleep(60)
                return query(
                    name,
                    session,
                    retries=retries - 1,
                )
            elif e.response.status_code == 500:
                logger.error("Server Error, trying to strip quotations")
                name = name.replace('"', "").replace("'", "")
                return query(name, session, retries=retries)
        else:
            # NOTE: we should not get here, debug
            logger.error("Query failed")
            import code

            code.interact(local=locals())
            return


def apply_ror(data, session):
    """
    Try different techniques to resolve with ror
    Such as adding the country and city directly into the query string
    and removing special characters so ror treats it as one string
    """
    results = get_empty_results()
    name = data["name"]
    if name != name:
        logger.debug("skipping null name")
        return results
    no_special = remove_special_chars(name, replace="")
    if no_special.isnumeric():
        logger.debug(f"skipping numeric name {name}")
        return results
    results = process_ror_json(name, session, results, extra_data=data)
    # Try adding the country and city into the query
    if not results["name_ror"] and data.get("country") and data.get("city"):
        logger.debug("Adding in city and country")
        results["method"] = "city/country"
        results = process_ror_json(
            name + f",{data.get('city')}, {data.get('country')}",
            session,
            results,
            extra_data=data,
        )
    if not results["name_ror"]:
        if no_special != name:
            logger.debug("Checking removing special chars")
            # ROR splits on special chars, remove special chars and try again
            # https://github.com/ror-community/ror-api/blob/master/rorapi/matching.py#L28C1-L28C20
            # i.e. Federal State Budgetary Scientific Institution
            # ""Federal Research Centre of Nutrition, Biotechnology""
            results["method"] = "remove_special"
            results = process_ror_json(
                remove_special_chars(name),
                session,
                results,
                extra_data=data,
            )
    return results


def process_file(args):
    trial_file = args.trial_file
    output_name = args.output_name
    data_type = args.data_type
    mapping_dict_file = args.mapping_dict_file
    use_cache = args.use_cache
    keep_dups = args.keep_duplicates
    add_to_mapping_dict = args.add_to_mapping_dict
    chunk_size = args.chunk_size

    output_dir = trial_file.parent
    output_file = output_dir / output_name

    session = create_session(use_cache)

    if data_type == "site":
        compare_id = ["name", "city", "country"]
    else:
        compare_id = ["name"]
    unique_id = ["trial_id"] + compare_id

    trials = load_trials(trial_file, unique_id, (not keep_dups))
    already_processed = load_trials(output_file, unique_id)

    # Filter out entries already present in the output file
    remaining = filter_unindexed(trials, already_processed, unique_id)

    # If a harcoded map is provided, resolve those first
    if mapping_dict_file and mapping_dict_file.exists():
        remaining = filter_already_mapped(
            mapping_dict_file, remaining, output_file, compare_id, unique_id
        )

    all_results = {}
    query_num = 0
    total_chunks = math.ceil(len(remaining) / chunk_size)
    for key, data in remaining.groupby(compare_id):
        logger.debug(f"chunk: ({query_num // chunk_size}/{total_chunks})")
        results = apply_ror(dict(zip(compare_id, key)), session)
        logger.debug(
            f"{query_num % chunk_size}/{chunk_size} chunk {query_num // chunk_size}/{total_chunks}: {key[0]} {results}"
        )
        for index in data.index:
            all_results[index] = results
        if query_num % chunk_size == 0:
            # Intermediate save
            logging.debug("Saving...")
            results_df = pandas.DataFrame.from_dict(all_results).T
            combined = (
                merge_drop_dups(results_df, remaining, on="index")
                .drop("_merge", axis=1)
                .sort_index(axis=1)
            )
            if mapping_dict_file and add_to_mapping_dict:
                write_to_mapping_dict(combined, mapping_dict_file)
            combined.to_csv(
                output_file, mode="a", header=not output_file.exists(), index=False
            )
            all_results = {}
        query_num += 1
    results_df = pandas.DataFrame.from_dict(all_results).T
    combined = (
        merge_drop_dups(results_df, remaining, on="index")
        .drop("_merge", axis=1)
        .sort_index(axis=1)
    )
    assert len(combined[(combined.ror.notnull()) & (combined.name.isnull())]) == 0
    if mapping_dict_file and add_to_mapping_dict:
        write_to_mapping_dict(combined, mapping_dict_file)
    combined.to_csv(output_file, mode="a", header=not output_file.exists(), index=False)


# TODO: MAKE THIS GENERALIZABLE
def make_table(args):
    processed_file = args.processed_file
    output_dir = processed_file.parent

    df = pandas.read_csv(processed_file)

    if "country" in df.columns:
        df["country"] = convert_country(df["country"], to="name")

        # % ROR by country
        table = df.groupby(["country"])[["ror", "trial_id"]].agg(
            {"ror": "count", "trial_id": "count"}
        )
        table.loc["Total"] = table.sum()
        table.columns = ["Has ROR", "Total Trials"]
        table["% Has ROR"] = (100 * table["Has ROR"] / table["Total Trials"]).astype(
            int
        )
        table = table.sort_values(["% Has ROR", "Total Trials"], ascending=False)
        total = table.T.pop("Total")
        table = pandas.concat([total.to_frame().T, table], axis=0)
        table.to_csv(output_dir / f"{processed_file.stem}_table_counts.csv")

        # Among those with ROR, count of papers by site
        ror = df[df.ror.notnull()]
        counts = (
            ror.groupby(["name_ror", "city", "country"])
            .agg({"trial_id": "count", "name": pandas.Series.nunique})
            .sort_values(by="trial_id", ascending=False)
        )
        counts.index = counts.index.set_names(["Site Name", "City", "Country"])
        counts.columns = ["Total Trials", "Unique Names"]
        counts.to_csv(output_dir / f"{processed_file.stem}_table_only_ror.csv")

    else:
        # % ROR
        table = pandas.DataFrame(df.agg({"ror": "count", "trial_id": "count"})).T
        table.columns = ["Has ROR", "Total Trials"]
        table["% Has ROR"] = (100 * table["Has ROR"] / table["Total Trials"]).astype(
            int
        )
        table.to_csv(output_dir / f"{processed_file.stem}_table_counts.csv")

        # Among those with ROR, count of papers by site
        ror = df[df.ror.notnull()]
        counts = (
            ror.groupby(["name_ror"])
            .agg({"trial_id": "count", "name": pandas.Series.nunique})
            .sort_values(by="trial_id", ascending=False)
        )
        counts.index = counts.index.set_names(["Site Name"])
        counts.columns = ["Total Trials", "Unique Names"]
        counts.to_csv(output_dir / f"{processed_file.stem}_table_only_ror.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # For the query command
    query_parser = subparsers.add_parser("query")
    query_parser.set_defaults(func=process_file)
    query_parser.add_argument(
        "--trial-file",
        required=True,
        type=pathlib.Path,
        help="File with the following columns: trial_id, name (site/sponsor), source (ctgov etc) and optionally city, country",
    )
    query_parser.add_argument(
        "--output-name", type=str, required=True, help="Name of output file"
    )
    query_parser.add_argument(
        "--data-type",
        choices=["site", "sponsor"],
        required=True,
        help="Site data assumes city/country columns, sponsor does not",
    )
    query_parser.add_argument(
        "--mapping-dict-file",
        type=pathlib.Path,
        help="Path to dictionary with definted matches",
    )
    query_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Cache queries and/or use cache",
    )
    query_parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not drop completely duplicated rows",
    )
    query_parser.add_argument(
        "--add-to-mapping-dict",
        action="store_true",
        help="Add newly resolved ids to the mapping dict",
    )
    query_parser.add_argument(
        "--chunk-size", type=int, help="Save the output every n queries", default=1000
    )

    # For the table command
    table_parser = subparsers.add_parser("table")
    table_parser.set_defaults(func=make_table)
    table_parser.add_argument(
        "--processed-file", required=True, type=pathlib.Path, help="File with ROR data"
    )

    args = parser.parse_args()
    args.func(args)
