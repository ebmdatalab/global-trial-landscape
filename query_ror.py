import argparse
import country_converter as coco
import pathlib
import requests
import requests_cache
from requests_cache import NEVER_EXPIRE, CachedSession
import re
import pandas
import time
import urllib
import numpy

SPECIAL_CHARS_REGEX = '[\+\-\=\|\>\<\!\(\)\\\{\}\[\]\^"\~\*\?\:\/\.\,\;]'


def create_session(use_cache=False):
    if use_cache:
        requests_cache.install_cache("ror_cache", backend="sqlite")
        # Do not expire the cache
        session = CachedSession(expire_after=NEVER_EXPIRE)
    else:
        session = requests.Session()
    return session


def load_mapping_dict(mapping_dict_path):
    return pandas.read_csv(
        mapping_dict_file, index_col=["raw_string", "city", "country"]
    )


def filter_already_mapped(mapping_dict_file, remaining):
    mapping_dict = load_mapping_dict(mapping_dict_file)

    # reset index so we preserve the index from the CTGOV file
    # and use that to determine what entries are left
    # (there may be more than one site per NCTId)
    merged = (
        remaining.reset_index()
        .merge(
            mapping_dict,
            left_on=["site", "city", "country"],
            right_on=["raw_string", "city", "country"],
            how="left",
        )
        .set_index("index")
    )
    assert len(remaining) == len(merged)
    already_mapped = merged[merged.ror.notnull()]
    already_mapped["method"] = "dict"

    already_mapped.to_csv(
        output_file, mode="a", header=not output_file.exists()
    )
    return remaining.loc[remaining.index.difference(already_mapped.index)]


def write_to_mapping_dict(df, mapping_dict_file):
    matches_map = (
        df[df.ror.notnull()][["site", "name", "ror", "city", "country"]]
        .drop_duplicates()
        .rename(columns={"site": "raw_string", "name": "name_ror"})
    )
    matches_map = matches_map.set_index(["raw_string", "city", "country"])
    if mapping_dict_file.exists():
        mapping_dict = load_mapping_dict(mapping_dict_file)
        joined = pandas.concat([mapping_dict, matches_map])
        new_matches = joined.loc[joined.index.difference(mapping_dict.index)]
    else:
        new_matches = matches_map
    new_matches.to_csv(mapping_dict_file, mode="a")


def remove_special_chars(sponsor, replace=" "):
    return re.sub(SPECIAL_CHARS_REGEX, replace, sponsor)


def clean(site):
    """
    The following strings were commonly the entire site name, in which case
    we do not want to try to match
    If they are part of the string, they may hamper the ror affiliation matching
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
    site = (
        site.str.lower()
        .str.split(criteria)
        .str[0]
        .str.strip()
        .replace("", numpy.nan)
    )
    return site


def convert_country(country_column):
    """
    Standardise to a country code
    Use ISO2 as per the ror schema
    https://ror.readme.io/docs/all-ror-fields-and-sub-fields
    """
    cc = coco.CountryConverter()
    return cc.pandas_convert(country_column, to="ISO2", not_found=numpy.nan)


def load_sponsors(name, index_col=None, drop_duplicates=True):
    try:
        if index_col is not None:
            sponsors = pandas.read_csv(name, index_col=index_col)
        else:
            sponsors = pandas.read_csv(name)
    except FileNotFoundError:
        sponsors = pandas.DataFrame({})
    return sponsors.drop_duplicates(ignore_index=True)


def chunk(sponsors, chunk_size=100):
    """
    Split list into chunks of size 'chunk_size'
    """
    chunks = [
        sponsors[i * chunk_size : (i + 1) * chunk_size]
        for i in range((len(sponsors) + chunk_size - 1) // chunk_size)
    ]
    return chunks


def get_empty_results():
    return {
        "name": None,
        "ror": None,
        "method": None,
        "method_ror": None,
        "count_non_matches": 0,
        "non_matches": [],
    }


def process_response(response, results, method="first", method_ror="chosen"):
    # Just return the name for now
    results["name"] = response["organization"]["name"]
    results["ror"] = response["organization"]["id"]
    results["method"] = method
    results["method_ror"] = method_ror
    return results


def process_ror_json(sponsor, session, results, extra_data):
    """
    There are two techniques for choosing a match from ror
    1. Use the ror "chosen field"
    2. The highest score above 0.8 among the ror objects with matching
       city/country
    """
    resp = query(sponsor, session)
    # TODO: there could be two chosen
    for item in resp["items"]:
        if item["chosen"]:
            return process_response(item, results)
    potential_matches = []
    for item in resp["items"]:
        if (
            item["organization"]["country"]["country_code"]
            == extra_data.country
        ) and (
            item["organization"]["addresses"][0]["city"] == extra_data.city
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


def query(sponsor, session, retries=2):
    """
    Given a sponsor name, query the ror api
    """
    url = f"https://api.ror.org/organizations?affiliation={urllib.parse.quote_plus(sponsor)}"
    try:
        response = session.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        import code

        code.interact(local=locals())
        if e.response.status_code == 403:
            print(f"Failed to download, retries left {retries}")
            retries -= 1
            if retries >= 0:
                time.sleep(60)
                return query(
                    sponsor,
                    session,
                    retries=retries,
                )
            else:
                # NOTE: we should not get here, debug
                import code

                code.interact(local=locals())
        else:
            print(f"Failed to download, retries left {retries}")
            # NOTE: we should not get here, debug
            import code

            code.interact(local=locals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sponsor-file",
        required=True,
        type=pathlib.Path,
        help="File with NCTId, site, city and country data",
    )
    parser.add_argument(
        "--output-name", type=str, required=True, help="Name of output file"
    )
    parser.add_argument(
        "--mapping-dict-file",
        type=pathlib.Path,
        help="Path to dictionary with definted matches",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Cache queries and/or use cache",
    )
    parser.add_argument(
        "--index-col", type=int, help="Column number of index", default=None
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not drop completely duplicated rows",
    )
    parser.add_argument(
        "--add-to-mapping-dict",
        action="store_true",
        help="Add newly resolved ids to the mapping dict",
    )
    args = parser.parse_args()
    sponsor_file = args.sponsor_file
    output_name = args.output_name
    mapping_dict_file = args.mapping_dict_file
    use_cache = args.use_cache
    index_col = args.index_col
    keep_dups = args.keep_duplicates
    add_to_mapping_dict = args.add_to_mapping_dict

    output_dir = sponsor_file.parent
    output_file = output_dir / output_name

    session = create_session(use_cache)

    sponsors = load_sponsors(sponsor_file, index_col, (not keep_dups))
    sponsors = sponsors[0:19999]
    sponsors.loc[:, "country"] = convert_country(sponsors.country)
    already_processed = load_sponsors(output_file, index_col).index
    remaining = sponsors.loc[sponsors.index.difference(already_processed)]

    if mapping_dict_file and mapping_dict_file.exists():
        remaining = filter_already_mapped(mapping_dict_file, remaining)

    remaining["site_lower"] = clean(sponsors.site)

    chunks = chunk(remaining, 1000)
    all_results = {}
    for chunk_num, chunk in enumerate(chunks):
        print(chunk_num, len(chunks))
        for index, data in chunk.iterrows():
            print(index)
            results = get_empty_results()
            sponsor = data["site_lower"]
            # sponsor = sponsor.replace('"', '').replace("'", "")
            if sponsor != sponsor:
                print("Skipping null")
                all_results[index] = results
                continue
            no_special = remove_special_chars(sponsor, replace="")
            if no_special.isnumeric():
                print(f"Skipping, only numeric {sponsor}")
                all_results[index] = results
                continue
            name = process_ror_json(sponsor, session, results, extra_data=data)
            print(index, sponsor, name, len(all_results))
            # If one work, try acronym (capitalize)
            if not name["name"] and len(sponsor.split()) == 1:
                print("Trying capitlization")
                results["method"] = "capitalize"
                name = process_ror_json(
                    sponsor.upper(),
                    session,
                    results,
                    extra_data=data,
                )
            # Try adding the country and city into the query
            if not name["name"]:
                print("Adding in city and country")
                results["method"] = "city/country"
                name = process_ror_json(
                    sponsor + f",{data.city}, {data.country}",
                    session,
                    results,
                    extra_data=data,
                )
                print(index, sponsor, name, len(all_results))
            if not name["name"]:
                if no_special != sponsor:
                    print("Checking removing special chars")
                    # ROR splits on special chars, remove special chars and try again
                    # https://github.com/ror-community/ror-api/blob/master/rorapi/matching.py#L28C1-L28C20
                    # i.e. Federal State Budgetary Scientific Institution ""Federal Research Centre of Nutrition, Biotechnology""
                    results["method"] = "remove_special"
                    name = process_ror_json(
                        remove_special_chars(sponsor),
                        session,
                        results,
                        extra_data=data,
                    )
                    print(index, sponsor, name, len(all_results))
            all_results[index] = name
        matches = pandas.DataFrame.from_dict(all_results).T
        df = chunk.join(matches).drop("site_lower", axis=1)
        if mapping_dict_file and add_to_mapping_dict:
            write_to_mapping_dict(df, mapping_dict_file)
        df.name.fillna(df.site)
        df.to_csv(output_file, mode="a", header=not output_file.exists())
