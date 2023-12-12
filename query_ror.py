import argparse
import logging
import math
import pathlib
import re
import sys
import time
import urllib

import matplotlib.pyplot as plt
import pandas
import plotly.graph_objects as go
import schemdraw
from schemdraw import flow

from setup import (
    get_base_parser,
    get_full_parser,
    get_results_parser,
    get_verbosity_parser,
    setup_logger,
)
from utils import (
    add_suffix,
    append_safe,
    create_session,
    filter_unindexed,
    load_glob,
    load_trials,
    map_who,
    match_paths,
    preprocess_trial_file,
    query,
    region_map,
    region_pie,
    remove_surrounding_double_quotes,
    world_map,
)


SPECIAL_CHARS_REGEX = r"[\+\-\=\|\>\<\!\(\)\\\{\}\[\]\^\"\~\*\?\:\/\.\,\;]"


def merge_and_update(left, right, on, indicator=False):
    assert isinstance(on, list)
    overlapping_columns = list(left.columns.intersection(right.columns))
    for col in on:
        overlapping_columns.remove(col)
    merged = (
        left.reset_index()
        .merge(right, on=on, how="left", indicator=indicator)
        .set_index("index")
    )
    for col in overlapping_columns:
        merged[col] = merged[f"{col}_y"].combine_first(merged[f"{col}_x"])
    # Drop the redundant columns
    for col in overlapping_columns:
        merged.drop([f"{col}_x", f"{col}_y"], axis=1, inplace=True)
    return merged


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


def check_mapping_dict(mapping_dict, compare_id):
    errors = mapping_dict.groupby(compare_id, dropna=False).filter(
        lambda group: group["ror"].nunique() > 1
    )
    if len(errors) > 1:
        logging.error(
            "The following rows in the mapping dictionary may have errors and need to be fixed manually"
        )
        print(errors[compare_id])
        sys.exit(1)


def load_mapping_dict(mapping_dict_path):
    mapping_dict = pandas.read_csv(mapping_dict_path)
    # If the index was created with city and country, there may be duplicated entries
    # TODO: what to do with entries that do not have city and country?
    return mapping_dict


def filter_already_mapped(mapping_dict_file, df, output_file, compare_id, unique_id):
    mapping_dict = load_mapping_dict(mapping_dict_file)

    # If the map has more info (city/country) than df we could have multiple matches
    # Only allow those with one match, throw the rest back to ror
    mapping_dict = mapping_dict[
        mapping_dict.groupby(compare_id)["ror"].transform("count").eq(1)
    ]

    merged = merge_and_update(df, mapping_dict, compare_id, indicator=True)

    already_mapped = merged[merged._merge == "both"].drop("_merge", axis=1)
    remaining = merged[merged._merge == "left_only"].drop("_merge", axis=1)

    assert (len(remaining) + len(already_mapped)) == len(df)
    # Nothing that is mapped should have a null name_resolved
    assert (already_mapped.name_resolved.notnull() | already_mapped.ror.notnull()).all()

    # Make sure what we write will match the following output
    ror_columns = get_empty_results().keys()
    necessary_columns = list(ror_columns) + list(df.columns)
    already_mapped["method"] = "dict"
    already_mapped["count_non_matches"] = 0
    already_mapped["non_matches"] = len(already_mapped) * [[]]
    already_mapped = already_mapped.reindex(columns=necessary_columns)
    # Sort the columns alphabetically
    already_mapped = already_mapped.sort_index(axis=1)
    append_safe(already_mapped, output_file)
    return remaining


def write_to_mapping_dict(df, mapping_dict_file, compare_id):
    ror_columns = get_empty_results().keys()
    necessary_columns = list(ror_columns) + compare_id
    matches_map = (
        df[df.name_resolved.notnull()]
        .reindex(columns=necessary_columns)
        .drop_duplicates()
    )
    if mapping_dict_file.exists():
        mapping_dict = load_mapping_dict(mapping_dict_file)
        new_matches = filter_unindexed(matches_map, mapping_dict, compare_id)
    else:
        new_matches = matches_map
    new_matches = new_matches.sort_index(axis=1)
    if not new_matches.empty:
        check_mapping_dict(mapping_dict, compare_id)
        append_safe(new_matches, mapping_dict_file)


def remove_special_chars(name, replace=" "):
    return re.sub(SPECIAL_CHARS_REGEX, replace, name)


def get_empty_results():
    return {
        "name_resolved": None,
        "ror": None,
        "organization_type": None,
        "lattitude": None,
        "longitude": None,
        "city_ror": None,
        "country_ror": None,
        "method": None,
        "count_non_matches": 0,
        "non_matches": [],
    }


def process_response(response, results, method=None):
    if response.get("organization"):
        response = response.get("organization")
    results["name_resolved"] = response["name"]
    results["ror"] = response["id"]
    results["organization_type"] = "_".join(response["types"])
    results["lattitude"] = response["addresses"][0]["lat"]
    results["longitude"] = response["addresses"][0]["lng"]
    results["city_ror"] = response["addresses"][0]["city"]
    results["country_ror"] = response["country"]["country_code"]
    results["method"] = method
    return results


def process_ror_json(name, session, results, extra_data):
    """
    There are three techniques for choosing a match from ror
    1. The highest score above 0.8 with matching
       city/country
    2. The highest score above 0.8 without country/city match, if industry
    3. Chosen (look for false positives)
    """
    try:
        resp = query_ror(name, session)
    except Exception:
        resp = None
    if not resp:
        results["method"] = "query_failed"
        return results
    potential_matches = []
    extra_city = extra_data.get("city")
    extra_country = extra_data.get("country")
    chosen = None
    for item in resp["items"]:
        if item["chosen"]:
            chosen = item
        if item["score"] >= 0.8:
            potential_matches.append(item)
    # Make sure we skip numpy.nan
    if (extra_country and extra_country == extra_country) and (
        extra_city and extra_city == extra_city
    ):
        for item in potential_matches:
            if (item["organization"]["country"]["country_code"] == extra_country) and (
                # Sometimes they are encoded differently, check city and geonames city
                (
                    item["organization"]["addresses"][0]["city"].lower()
                    == extra_city.lower()
                )
                or (
                    item["organization"]["addresses"][0]["geonames_city"][
                        "geonames_admin1"
                    ]["ascii_name"].lower()
                    == extra_city.lower()
                )
            ):
                return process_response(item, results, method="city/country")
    # Make sure we skip numpy.nan
    elif extra_country and extra_country == extra_country:
        for item in potential_matches:
            if item["organization"]["country"]["country_code"] == extra_country:
                return process_response(item, results, method="country")
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
    try:
        return query(url, session)
    except Exception as e:
        if e.response.status_code == 500:
            logging.error(f"Server Error, trying to strip quotations {name}")
            replaced = name.replace('"', "").replace("'", "")
            if replaced != name:
                url = make_url(name)
                return query(url, session, retries=1)


def apply_ror(data, session):
    """
    Try different techniques to resolve with ror
    Such as adding the country and city directly into the query string
    and removing special characters so ror treats it as one string
    """
    results = get_empty_results()
    name = data["name"]
    # TODO: this could/should happen once at the df level
    # TODO: this could be part of general cleaning/standardising
    name = remove_surrounding_double_quotes(str(name))
    if name != name:
        logging.debug("skipping null name")
        return results
    no_special = remove_special_chars(name, replace="")
    if no_special.replace(" ", "").isnumeric() or len(no_special) == 0:
        logging.debug(f"skipping numeric/special char name {name}")
        return results
    results = process_ror_json(name, session, results, extra_data=data)
    if not results["name_resolved"]:
        if no_special != name:
            logging.debug("Checking substring")
            results["method"] = "substring"
            results = process_ror_json(
                '"' + name + '"',
                session,
                results,
                extra_data=data,
            )
    return results


def process_file(args):
    trial_file = args.input_file
    mapping_dict_file = args.mapping_dict_file
    use_cache = args.use_cache
    keep_dups = args.keep_duplicates
    add_to_mapping_dict = args.add_to_mapping_dict
    chunk_size = args.chunk_size

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = trial_file.parent

    output_file = add_suffix(output_dir, trial_file, "ror")

    session = create_session("ror_cache", use_cache)

    column_names = pandas.read_csv(trial_file).columns

    compare_id = ["name"]
    if "country" in column_names:
        compare_id += ["country"]
    if "city" in column_names:
        compare_id += ["city"]
    unique_id = ["trial_id"] + compare_id

    trials = load_trials(trial_file, unique_id, (not keep_dups))
    already_processed = load_trials(output_file, unique_id)

    # Filter out entries already present in the output file
    remaining = filter_unindexed(trials, already_processed, unique_id)

    # If a harcoded map is provided, resolve those first
    # TODO: things that have ror already resolved
    if mapping_dict_file and mapping_dict_file.exists():
        remaining = filter_already_mapped(
            mapping_dict_file, remaining, output_file, compare_id, unique_id
        )

    all_results = {}
    query_num = 0
    total_chunks = math.ceil(len(remaining) / chunk_size)
    for key, data in remaining.groupby(compare_id, dropna=False):
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
            append_safe(combined, output_file)
            all_results = {}
        query_num += 1
    results_df = pandas.DataFrame.from_dict(all_results).T
    combined = (
        merge_drop_dups(results_df, remaining, on="index")
        .drop("_merge", axis=1)
        .sort_index(axis=1)
    )
    if not combined.empty:
        assert len(combined[(combined.ror.notnull()) & (combined.name.isnull())]) == 0
        if mapping_dict_file and add_to_mapping_dict:
            write_to_mapping_dict(combined, mapping_dict_file, compare_id)
        append_safe(combined, output_file)
    check_finished(trials, output_file)


def check_finished(trials, output_file):
    assert len(trials) == len(pandas.read_csv(output_file))


def get_ror_metadata(df, session, update_all=False, ror_column="ror"):
    def make_url(ror_id):
        return f"https://api.ror.org/organizations/{ror_id}"

    to_update = df[df[ror_column].notnull()]
    if not update_all:
        # TODO: potentially also check lattitude?
        if "name_resolved" in to_update.columns:
            to_update = to_update[to_update.name_resolved.isnull()]
    ror_ids = to_update[ror_column].dropna().unique()
    all_results = []
    for ror_id in ror_ids:
        url = make_url(ror_id)
        results = {}
        # TODO: catch errors
        resp = query(url, session)
        time.sleep(0.1)
        process_response(resp, results)
        all_results.append(results)

    if len(all_results) > 0:
        resolved = pandas.DataFrame(all_results)
        resolved = resolved.rename(columns={"ror": ror_column})
        # Add metadata to any row with a matching ror id
        # NOTE: this preserves the index
        merged = merge_and_update(to_update, resolved, on=[ror_column])
        # Make sure the original dataframe has all the necessary columns
        df = df.reindex(columns=df.columns.union(merged.columns).unique())
        # Only update the rows we wanted to update
        df.loc[merged.index] = merged
        return df
    logging.info("No ror metadata to update")
    return df


def map_ror_file(args):
    trial_file = args.input_file
    mapping_dict_file = args.mapping_dict_file
    use_cache = args.use_cache
    update_all = args.update_all

    column_names = pandas.read_csv(trial_file).columns

    compare_id = ["name"]
    if "country" in column_names:
        compare_id += ["country"]
    if "city" in column_names:
        compare_id += ["city"]
    unique_id = ["trial_id"] + compare_id

    trials = load_trials(trial_file, unique_id, True)

    session = create_session("ror_cache", use_cache)
    trials = get_ror_metadata(trials, session, update_all=update_all)
    write_to_mapping_dict(trials, mapping_dict_file, compare_id)


def update_metadata(args):
    trial_file = args.input_file
    use_cache = args.use_cache
    update_all = args.update_all
    ror_column = args.ror_column
    output_name = add_suffix(trial_file.parent, trial_file, "metadata")

    column_names = pandas.read_csv(trial_file).columns
    assert ror_column in column_names

    compare_id = ["name"]
    if "country" in column_names:
        compare_id += ["country"]
    if "city" in column_names:
        compare_id += ["city"]
    unique_id = ["trial_id"] + compare_id

    trials = load_trials(trial_file, unique_id, True)
    # In case manually added ror has leading/trailing spaces
    trials[ror_column] = trials[ror_column].str.strip()

    session = create_session("ror_cache", use_cache)
    trials = get_ror_metadata(
        trials, session, update_all=update_all, ror_column=ror_column
    )
    trials.to_csv(output_name, index=False)


def make_map(args):
    input_files = args.input_files
    file_filter = args.file_filter
    plot_world = args.plot_world
    country_column = args.country_column
    title = args.title
    keep_indiv = args.keep_indiv
    keep_companies = args.keep_companies
    if file_filter == "manual":
        df = load_glob(input_files, file_filter, True, keep_indiv, keep_companies)
    else:
        df = load_glob(input_files, file_filter, True, False, False)

    sources = sorted(df.source.unique())
    if country_column not in df.columns:
        raise argparse.ArgumentTypeError(f"Input data does not have {country_column}")

    counts = df.groupby(country_column).trial_id.size()
    if plot_world:
        world_map(counts, country_column=country_column)
    else:
        region_map(counts, country_column=country_column)
    plt.suptitle(f"{title}\nData from: {' '.join(sources)} ({file_filter})")
    plt.savefig(f"{'_'.join(sources)}_map.png", bbox_inches="tight")


def org_region(args):
    input_files = args.input_files
    file_filter = args.file_filter
    keep_indiv = args.keep_indiv
    keep_companies = args.keep_companies
    country_column = args.country_column
    df = load_glob(input_files, file_filter, True, not keep_indiv, not keep_companies)
    exclusion_str = " (excluding"
    if not keep_companies:
        exclusion_str += " companies"
    if not keep_indiv:
        if not keep_companies:
            exclusion_str += " and individuals)"
        else:
            exclusion_str += " individuals)"
    else:
        exclusion_str += ")"
    if keep_companies and keep_indiv:
        exclusion_str = ""
    sources = sorted(df.source.unique())
    region_pie(df, country_column)
    plt.suptitle(
        f"Sponsor Type by WHO Region with Registry Data\nData from: {' '.join(sources)}{exclusion_str}"
    )
    plt.savefig(
        f"{'_'.join(sources)}_sponsor_by_region{exclusion_str}.png", bbox_inches="tight"
    )


def site_sponsor(args):
    sponsor_files = args.sponsor_files
    site_files = args.site_files
    sponsor_filter = args.sponsor_filter
    site_filter = args.site_filter
    sponsor_country_column = args.sponsor_country_column
    site_country_column = args.site_country_column
    exclude_same = args.exclude_same

    site_df = load_glob(site_files, site_filter, True, False, False)
    sponsor_df = load_glob(sponsor_files, sponsor_filter, True, True, True)
    # Only use sites with same source as sponsor
    site_df = site_df[site_df.source.isin(sponsor_df.source.unique())]
    site_df["who_region"] = map_who(site_df[site_country_column])
    sponsor_df["sponsor_who_region"] = map_who(sponsor_df[sponsor_country_column])
    merged = site_df.merge(
        sponsor_df,
        left_on=["source", "trial_id"],
        right_on=["source", "trial_id"],
        how="left",
    )
    counts = (
        merged.groupby(["who_region", "sponsor_who_region"])
        .trial_id.count()
        .reset_index()
    )

    if exclude_same:
        counts = counts.loc[counts.who_region != counts.sponsor_who_region]
    # Map nodes to node ids
    who_map = {name: index for index, name in enumerate(counts.who_region.unique())}
    who_sponsor_map = {
        name: index + len(who_map)
        for index, name in enumerate(counts.sponsor_who_region.unique())
    }
    link = dict(
        source=list(counts.sponsor_who_region.map(who_sponsor_map)),
        target=list(counts.who_region.map(who_map)),
        value=list(counts.trial_id),
    )
    data = go.Sankey(
        link=link, node=dict(label=list(who_map.keys()) + list(who_sponsor_map.keys()))
    )
    fig = go.Figure(data)
    sources = sorted(set(merged.source))
    fig.update_layout(
        title=f"Mapping Sponsor to Trial Site by WHO Region (data from: {' '.join(sources)})",
    )
    fig.add_annotation(x=0, y=1.05, text="<b>Sponsor</b>", showarrow=False)
    fig.add_annotation(x=1, y=1.05, text="<b>Site</b>", showarrow=False)
    fig.write_html("sankey.html")


def flowchart(args):
    input_files = args.input_files
    df = load_glob(input_files, "manual")

    total = df.shape[0]

    individual = df.individual
    company = (~individual) & (df.organization_type == "Company")
    no_manual = (~individual) & (~company) & (df.no_manual_match | df.name.isnull())

    leftover = df[~(individual) & ~(no_manual) & ~(company)]

    ror_manual = (
        leftover.ror.isnull()
        & leftover.name_manual.isnull()
        & leftover.ror_manual.notnull()
    )
    ror_fixed = (
        leftover.ror.notnull()
        & leftover.name_manual.isnull()
        & leftover.ror_manual.notnull()
    )  # 32
    ror_right = (
        leftover.ror.notnull()
        & leftover.name_manual.isnull()
        & leftover.ror_manual.isnull()
    )

    ror_any = ror_manual | ror_fixed | ror_right

    manual = leftover.name_manual.notnull()

    assert (ror_any.sum() + manual.sum()) == len(leftover)

    with schemdraw.Drawing() as d:
        d.config(fontsize=10)
        d += flow.Start(w=6, h=2).label(f"Total trials\nn={total}")
        d += flow.Arrow().down(d.unit / 2)
        d += (step1 := flow.Box(w=0, h=0))
        d += flow.Arrow().down(d.unit / 2)
        d += (step2 := flow.Box(w=0, h=0))
        d += flow.Arrow().down(d.unit / 2)
        d += (step3 := flow.Box(w=0, h=0))

        d += flow.Arrow().theta(-135)
        d += (
            flow.Box(w=6, h=4)
            .label(f"ROR resolved\nn={ror_any.sum()}")
            .label(f"\n\n\n\n(n={ror_fixed.sum()} ROR manually corrected)", fontsize=8)
            .label(
                f"\n\n\n\n\n\n\n(n={ror_manual.sum()} ROR manually resolved)",
                fontsize=8,
            )
            .label(
                f"\n\n\n\n\n\n\n\n\n\n(n={ror_right.sum()} ROR correct)",
                fontsize=8,
            )
        )

        d.move_from(step3.S)
        d += flow.Arrow().theta(-45)
        d += flow.Box(w=6, h=4).label(f"Name manually resolved\nn={manual.sum()}")

        # Exclusions
        d.config(fontsize=8)
        d += flow.Arrow().right(d.unit / 4).at(step1.E)
        d += flow.Box(w=6, h=1).anchor("W").label(f"Individual\nn={individual.sum()}")
        d += flow.Arrow().right(d.unit / 4).at(step2.E)
        d += flow.Box(w=6, h=1).anchor("W").label(f"Company\nn={company.sum()}")
        d += flow.Arrow().right(d.unit / 4).at(step3.E)
        d += (
            flow.Box(w=6, h=1)
            .anchor("W")
            .label(f"No manual match\nn={no_manual.sum()}")
        )

    output_name = "_".join(sorted(df.source.unique()))
    plt.savefig(f"{output_name}_flowchart")
    leftover.organization_type.value_counts().to_csv(f"{output_name}_orgs.csv")


def multisite(args):
    input_files = args.input_files
    df = load_glob(input_files, "none")
    counts = df.groupby("trial_id").trial_id.count()
    table = (
        (counts > 1)
        .value_counts()
        .rename(index={False: "Single Site", True: "Multi-Site"})
    )
    output_name = "_".join(sorted(df.source.unique()))
    table.to_csv(f"{output_name}_single_multi.csv")


if __name__ == "__main__":
    base = get_base_parser()
    results = get_results_parser()
    parent = get_full_parser()
    ror_parser = argparse.ArgumentParser()
    subparsers = ror_parser.add_subparsers()

    preprocess_parser = subparsers.add_parser("preprocess", parents=[base])
    preprocess_parser.set_defaults(func=preprocess_trial_file)
    preprocess_parser.add_argument(
        "--source",
        required=True,
        help="Trial registry type i.e. pactr, anzctr",
    )

    meta_parser = subparsers.add_parser("update_metadata", parents=[base])
    meta_parser.set_defaults(func=update_metadata)
    meta_parser.add_argument(
        "--update-all",
        action="store_true",
        help="Re-query ror for additional metadata",
    )
    meta_parser.add_argument(
        "--ror-column",
        type=str,
        default="ror",
        help="Column name with ror url",
    )
    meta_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Cache queries and/or use cache",
    )

    map_parser = subparsers.add_parser("map_existing", parents=[base])
    map_parser.set_defaults(func=map_ror_file)
    map_parser.add_argument(
        "--mapping-dict-file",
        type=pathlib.Path,
        help="Path to dictionary with definted matches",
        required=True,
    )
    map_parser.add_argument(
        "--update-all",
        action="store_true",
        help="Re-query ror for additional metadata",
    )
    map_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Cache queries and/or use cache",
    )

    query_parser = subparsers.add_parser("query", parents=[parent])
    query_parser.set_defaults(func=process_file)
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

    map_parser = subparsers.add_parser("map", parents=[results])
    map_parser.add_argument(
        "--plot-world",
        action="store_true",
        help="Plot a world map, rather than by WHO region",
    )
    map_parser.add_argument("--title", type=str, help="Title for plot", required=True)
    map_parser.add_argument(
        "--country-column",
        type=str,
        help="Name of country column to use",
        default="country",
    )
    map_parser.set_defaults(func=make_map)

    org_parser = subparsers.add_parser("sponsor-org", parents=[results])
    org_parser.add_argument(
        "--country-column",
        type=str,
        help="Name of country column to use",
        default="country",
    )
    org_parser.set_defaults(func=org_region)

    flowchart_parser = subparsers.add_parser("flowchart", parents=[results])
    flowchart_parser.set_defaults(func=flowchart)

    multisite_parser = subparsers.add_parser("multisite", parents=[results])
    multisite_parser.set_defaults(func=multisite)

    verb = get_verbosity_parser()
    site_sponsor_parser = subparsers.add_parser("site-sponsor", parents=[verb])
    site_sponsor_parser.add_argument(
        "--site-files",
        required=True,
        action="append",
        type=match_paths,
        help="One or more glob patterns for matching input files",
    )
    site_sponsor_parser.add_argument(
        "--sponsor-files",
        required=True,
        action="append",
        type=match_paths,
        help="One or more glob patterns for matching input files",
    )
    site_sponsor_parser.add_argument(
        "--sponsor-filter",
        choices=["manual", "ror", "country"],
        default="country",
        help="Filter registry data",
    )
    site_sponsor_parser.add_argument(
        "--site-filter",
        choices=["manual", "ror", "country"],
        default="country",
        help="Filter registry data",
    )
    site_sponsor_parser.add_argument(
        "--sponsor-country-column",
        type=str,
        help="Name of sponsor country column to use",
        default="country",
    )
    site_sponsor_parser.add_argument(
        "--site-country-column",
        type=str,
        help="Name of site country column to use",
        default="country",
    )
    site_sponsor_parser.add_argument(
        "--exclude-same",
        action="store_true",
        help="Exclude site/sponsor region the same",
    )
    site_sponsor_parser.set_defaults(func=site_sponsor)

    args = ror_parser.parse_args()
    if hasattr(args, "func"):
        setup_logger(args.verbosity)
        args.func(args)
    else:
        ror_parser.print_help()
