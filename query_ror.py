import argparse
import logging
import math
import pathlib
import re
import urllib

import country_converter as coco
import numpy
import pandas

from setup import get_base_parser, get_full_parser, setup_logger
from utils import create_session, filter_unindexed, load_trials, query


SPECIAL_CHARS_REGEX = r"[\+\-\=\|\>\<\!\(\)\\\{\}\[\]\^\"\~\*\?\:\/\.\,\;]"


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


def load_mapping_dict(mapping_dict_path):
    mapping_dict = pandas.read_csv(mapping_dict_path)
    # If the index was created with city and country, there may be duplicated entries
    # TODO: what to do with entries that do not have city and country?
    return mapping_dict


def filter_already_mapped(mapping_dict_file, df, output_file, compare_id, unique_id):
    mapping_dict = load_mapping_dict(mapping_dict_file)

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


def write_to_mapping_dict(df, mapping_dict_file, compare_id):
    matches_map = (
        df[df.ror.notnull()][
            ["name", "name_ror", "ror", "organization_type", "city", "country"]
        ]
        .drop_duplicates()
        .sort_index(axis=1)
    )
    if mapping_dict_file.exists():
        mapping_dict = load_mapping_dict(mapping_dict_file)
        new_matches = filter_unindexed(matches_map, mapping_dict, compare_id)
    else:
        new_matches = matches_map
    if not new_matches.empty:
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


def clean_trials(trials):
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
        "organization_type": None,
        "method": None,
        "method_ror": None,
        "count_non_matches": 0,
        "non_matches": [],
    }


def process_response(response, results, method="first", method_ror="chosen"):
    results["name_ror"] = response["organization"]["name"]
    results["ror"] = response["organization"]["id"]
    results["organization_type"] = "_".join(response["organization"]["types"])
    results["method"] = method
    results["method_ror"] = method_ror
    return results


def process_ror_json(name, session, results, extra_data):
    """
    There are three techniques for choosing a match from ror
    1. The highest score above 0.8 with matching
       city/country
    2. The highest score above 0.8 without country/city match, if industry
    3. Chosen (look for false positives)
    """
    resp = query_ror(name, session)
    if not resp:
        return results
    potential_matches = []
    chosen = None
    for item in resp["items"]:
        if item["chosen"]:
            chosen = item
        if item["score"] >= 0.8:
            potential_matches.append(item)
    if extra_data.get("country") and extra_data.get("city"):
        for item in potential_matches:
            if (
                item["organization"]["country"]["country_code"]
                == extra_data.get("country")
            ) and (
                item["organization"]["addresses"][0]["city"] == extra_data.get("city")
            ):
                return process_response(item, results, method="city/country")
    for item in potential_matches:
        if "Company" in item["organization"]["types"]:
            return process_response(item, results, method="industry")
    if chosen:
        return process_response(chosen, results, method="chosen")
    else:
        # TODO: append these together
        results["count_non_matches"] = len(potential_matches)
        results["non_matches"] = [
            {match["organization"]["name"]: match["score"]}
            for match in potential_matches
        ]
    return results


def query_ror(name, session):
    def make_url(name):
        return f"https://api.ror.org/organizations?affiliation={urllib.parse.quote_plus(name)}"

    url = make_url(name)
    logging.error(f"{url}")
    try:
        return query(url, session)
    except Exception as e:
        if e.response.status_code == 500:
            logging.error(f"Server Error, trying to strip quotations {name}")
            name = name.replace('"', "").replace("'", "")
            url = make_url(name)
            return query(name, session, retries=1)


def apply_ror(data, session):
    """
    Try different techniques to resolve with ror
    Such as adding the country and city directly into the query string
    and removing special characters so ror treats it as one string
    """
    results = get_empty_results()
    name = data["name"]
    if name != name:
        logging.debug("skipping null name")
        return results
    no_special = remove_special_chars(name, replace="")
    if no_special.isnumeric() or len(no_special) == 0:
        logging.debug(f"skipping numeric/special char name {name}")
        return results
    results = process_ror_json(name, session, results, extra_data=data)
    # Try adding the country and city into the query
    if not results["name_ror"] and data.get("country") and data.get("city"):
        logging.debug("Adding in city and country")
        results["method"] = "city/country"
        results = process_ror_json(
            name + f",{data.get('city')}, {data.get('country')}",
            session,
            results,
            extra_data=data,
        )
    if not results["name_ror"]:
        if no_special != name:
            logging.debug("Checking removing special chars")
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
    trial_file = args.input_file
    output_file = args.output_file
    data_type = args.data_type
    mapping_dict_file = args.mapping_dict_file
    use_cache = args.use_cache
    keep_dups = args.keep_duplicates
    add_to_mapping_dict = args.add_to_mapping_dict
    chunk_size = args.chunk_size

    session = create_session("ror_cache", use_cache)

    if data_type == "site":
        compare_id = ["name", "city", "country"]
    else:
        compare_id = ["name"]
    unique_id = ["trial_id"] + compare_id

    trials = load_trials(trial_file, unique_id, (not keep_dups))
    trials = clean_trials(trials)
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
        logging.debug(f"chunk: ({query_num // chunk_size}/{total_chunks})")
        results = apply_ror(dict(zip(compare_id, key)), session)
        logging.debug(
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
                write_to_mapping_dict(combined, mapping_dict_file, compare_id)
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


def pcnt_ror_table(df, groupby):
    # % ROR
    table = df.groupby(groupby).agg({"ror": "count", "trial_id": "count"})
    table.loc["Total"] = table.sum()
    table.columns = ["Has ROR", "Total Trials"]
    table["% Has ROR"] = (100 * table["Has ROR"] / table["Total Trials"]).astype(int)
    table = table.sort_values(["% Has ROR", "Total Trials"], ascending=False)
    total = table.T.pop("Total")
    table = pandas.concat([total.to_frame().T, table.drop("Total")], axis=0)
    return table


def write_table(df, output_dir, input_file, suffix):
    path = output_dir / f"{input_file.stem}_table_{suffix}.csv"
    df.to_csv(path)


def papers_by_site_table(df, groupby):
    # Among those with ROR, count of papers by site
    ror = df[df.ror.notnull()]
    table = (
        ror.groupby(groupby)
        .agg({"trial_id": "count", "name": pandas.Series.nunique})
        .sort_values(by="trial_id", ascending=False)
    )
    # table.index = counts.index.set_names(["Site Name", "City", "Country"])
    table.columns = ["Total Trials", "Unique Names"]
    return table


def make_tables(args):
    processed_file = args.input_file
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = processed_file.parent

    df = pandas.read_csv(processed_file)

    if "country" in df.columns:
        df["country"] = convert_country(df["country"], to="name")
        df["continent"] = convert_country(df["country"], to="continent")

        write_table(
            pcnt_ror_table(df, ["country"]), output_dir, processed_file, "by_country"
        )
        write_table(
            pcnt_ror_table(df, ["continent"]),
            output_dir,
            processed_file,
            "by_continent",
        )

        write_table(
            papers_by_site_table(df, ["name_ror", "city", "country"]),
            output_dir,
            processed_file,
            "only_ror",
        )
        africa = df[df.continent == "Africa"]
        write_table(
            papers_by_site_table(africa, ["name_ror", "city", "country"]),
            output_dir,
            processed_file,
            "only_ror_africa",
        )

    else:
        write_table(
            pcnt_ror_table(df, ["organization_type"]),
            output_dir,
            processed_file,
            "counts",
        )
        write_table(
            papers_by_site_table(df, ["name_ror"]),
            output_dir,
            processed_file,
            "only_ror",
        )


if __name__ == "__main__":
    parent = get_full_parser()
    ror_parser = argparse.ArgumentParser()
    subparsers = ror_parser.add_subparsers()

    query_parser = subparsers.add_parser("query", parents=[parent])
    query_parser.set_defaults(func=process_file)
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

    base = get_base_parser()
    table_parser = subparsers.add_parser("tables", parents=[base])
    table_parser.set_defaults(func=make_tables)
    table_parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Alternate directory to store tables",
    )

    args = ror_parser.parse_args()
    if hasattr(args, "func"):
        setup_logger(args.verbosity)
        args.func(args)
    else:
        ror_parser.print_help()
