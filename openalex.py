import argparse
import logging
import os
import pathlib
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import numpy
import pandas

from setup import get_base_parser, get_verbosity_parser, setup_logger
from utils import create_session


DEFAULT_PROTOCOL = "(clinicaltrial[Filter] NOT editorial)"

ids_exact = [
    r"(?i)NCT\s*\W*0\d{7}",
    r"20\d{2}\W*0\d{5}\W*\d{2}",
    r"(?i)PACTR\s*\W*20\d{13}",
    r"(?i)ACTRN\s*\W*126\d{11}",
    r"(?i)ANZCTR\s*\W*126\d{11}",
    r"(?i)NTR\s*\W*\d{4}",
    r"(?i)KCT\s*\W*00\d{5}",
    r"(?i)DRKS\s*\W*000\d{5}",
    r"(?i)ISRCTN\s*\W*\d{8}",
    r"(?i)ChiCTR\s*\W*20000\d{5}",
    r"(?i)IRCT\s*\W*20\d{10,11}N\d{1,3}",
    r"(?i)CTRI\s?\W*\/\s*\W*202\d{1}\s?\W*\/\s*\W*\d{2,3}\s*\W*\/\s*\W*0\d{5}",
    r"(?i)Japic\s*CTI\s*\W*\d{6}",
    r"(?i)jrct\W*\w{1}\W*\d{9}",
    r"(?i)UMIN\s*\W*\d{9}",
    r"(?i)JMA\W*IIA00\d{3}",
    r"(?i)RBR\s*\W*\d\w{5}",
    r"(?i)RPCEC\s*\W*0{5}\d{3}",
    r"(?i)LBCTR\s*\W*\d{10}",
    r"(?i)SLCTR\s*\W*\d{4}\s*\W*\d{3}",
    r"(?i)TCTR\s*\W*202\d{8}",
    r"{?i}PER\s*\W*\d{3}\s*\W*\d{2}",
]


def read_dataset(fpath):
    df = pandas.read_csv(
        fpath,
        delimiter="\t",
        names=[
            "pmid",
            "accession",
            "abstract",
            "pubdate",
        ],
        parse_dates=["pubdate"],
        na_values="-",
    )
    return df


def build_cohort(args):
    """
    Get pubmed ids
    """
    start_date = args.start_date
    end_date = args.end_date
    protocol = args.protocol
    output_file = args.output_file
    try:
        sys.path.insert(1, os.path.dirname(shutil.which("xtract")))
        import edirect
    except Exception:
        logging.error("Is edirect installed?")
    # The date MUST be included in the query with [dp] (rather than
    # -mindate -maxdate) in order for 10k+ queries to work
    # cmd = f"efetch -db pubmed -id 30553130 -format xml | xtract -pattern PubmedArticle -sep '|' -def '-' -element MedlineCitation/PMID -element AccessionNumber -element AbstractText -block PubDate -sep '-' -element Year,Month,Day > {output_file}"
    cmd = f"esearch -db pubmed -query '({start_date}:{end_date}[dp]) AND ({protocol})' | efetch -format xml | xtract -pattern PubmedArticle -sep '|' -def '-' -element MedlineCitation/PMID -element AccessionNumber -element AbstractText -block PubDate -sep '-' -element Year,Month,Day > {output_file}"
    logging.info(cmd)
    edirect.pipeline(cmd)

    df = read_dataset(output_file)
    df = split_bar(df, columns=["accession"])
    df = get_ids_from_abstract(df)

    df.to_csv(output_file, index=False)
    return


def split_bar(df, columns=[]):
    for col in columns:
        df[col] = df[col].str.split("|")
    return df


def get_ids_from_abstract(df):
    criteria = "(" + "|".join([rf"{x}" for x in ids_exact]) + ")"
    found = df.abstract.str.extractall(criteria).groupby(level=0)[0].apply(list)
    df.loc[found.index, "abstract"] = found
    df.loc[~df.index.isin(found.index), "abstract"] = numpy.nan
    return df


def chunk_pmids(pmids, chunk_size=20):
    chunks = [
        pmids[i * chunk_size : (i + 1) * chunk_size]
        for i in range((len(pmids) + chunk_size - 1) // chunk_size)
    ]
    return chunks


# TODO: could use utils retry query
def query(pmids, session, email_address=None, use_cache=False):
    if email_address:
        headers = {"mailto": email_address}
    else:
        headers = {}
    # Openalex allows up to 200 results/page, but we query 40 at a time
    # Return 50 on the page to be safe
    url = "https://api.openalex.org/works?per-page=50&filter=pmid:" + "|".join(pmids)
    try:
        response = session.get(url, headers=headers)
        pmids_returned = [
            (x["ids"]["pmid"]).split("/")[-1] for x in response.json()["results"]
        ]
        if set(pmids_returned) != set(pmids):
            logging.error(f"Failed to download {set(pmids) - set(pmids_returned)}")
        response.raise_for_status()
    except Exception:
        logging.error("Failed to download")
        sys.exit(1)
    return process_response(response.json()["results"], pmids)


def process_response(response, pmids, first_last=True):
    results = []
    # One row per author
    for paper in response:
        paper_dict = defaultdict(str)
        paper_dict["pmid"] = paper["ids"]["pmid"].split("/")[-1]
        paper_dict["doi"] = paper["ids"].get("doi", None)
        paper_dict["paper_openalex"] = paper["ids"]["openalex"]
        for author in paper["authorships"]:
            position = author["author_position"]
            if first_last and position == "middle":
                continue
            author_dict = paper_dict.copy()
            author_dict["author_orcid"] = author["author"]["orcid"]
            author_dict["author_name"] = author["author"]["display_name"]
            author_dict["author_corresponding"] = author["is_corresponding"]
            author_dict["author_position"] = position
            author_dict["author_affiliation_raw"] = author["raw_affiliation_string"]
            author_dict["author_openalex"] = author["author"].get("id", None)
            author_dict["author_name"] = author["author"].get("display_name", None)
            # NOTE: author can have multiple rows for multiple affiliations
            for institution in author["institutions"]:
                # One row per author affiliation
                institution_dict = author_dict.copy()
                institution_dict["institution_openalex"] = institution["id"]
                institution_dict["institution_ror"] = institution["ror"]
                institution_dict["institution_country"] = institution["country_code"]
                institution_dict["institution_type"] = institution["type"]
                institution_dict["institution_name"] = institution["display_name"]
                results.append(institution_dict)
            # NOTE: authors may not have any resolved institutions
            if len(author["institutions"]) == 0:
                results.append(author_dict)
    return results


def query_openalex(args):
    """
    Given a file of pubmed ids, add openalex info
    """
    input_file = args.input_file
    output_file = args.output_file
    email_address = args.email_address
    use_cache = args.use_cache

    session = create_session("openalex_cache", use_cache=use_cache)

    df = pandas.read_csv(input_file, dtype={"pmid": str})
    pmids = df.pmid.unique()
    # https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists#addition-or
    # openalex allows up to 50 works in a query, to be safe use 40
    chunks = chunk_pmids(pmids, 40)
    for index, chunk in enumerate(chunks):
        results = query(
            chunk, session, email_address=email_address, use_cache=use_cache
        )
        metadata = pandas.DataFrame(results)
        # Use an outer join so that we only write the current chunk
        merged = df[df.pmid.isin(chunk)].merge(
            metadata, left_on="pmid", right_on="pmid", how="outer"
        )
        assert merged.pmid.nunique() == len(chunk)
        merged.to_csv(
            output_file, mode="a", header=not output_file.exists(), index=False
        )


if __name__ == "__main__":
    verbosity_parser = get_verbosity_parser()
    base_parser = get_base_parser()
    openalex_parser = argparse.ArgumentParser()
    subparsers = openalex_parser.add_subparsers()

    cohort_parser = subparsers.add_parser("build-cohort", parents=[verbosity_parser])
    cohort_parser.set_defaults(func=build_cohort)
    cohort_parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Minimum date for query in the form YYYY-MM-DD, YYYY-MM or YYYY",
    )
    cohort_parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Maximum date for query in the form YYYY-MM-DD, YYYY-MM or YYYY",
    )
    cohort_parser.add_argument(
        "--protocol",
        type=str,
        default=DEFAULT_PROTOCOL,
        help=f"Pubmed search protocol (default = {DEFAULT_PROTOCOL}) should follow pubmed guidance: https://pubmed.ncbi.nlm.nih.gov/help/",
    )
    cohort_parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Output file name to write cohort pmids",
    )

    query_parser = subparsers.add_parser("query-openalex", parents=[base_parser])
    query_parser.set_defaults(func=query_openalex)
    query_parser.add_argument(
        "--email-address",
        type=str,
        help="Email address to provide to api",
    )
    query_parser.add_argument(
        "--use-cache", action="store_true", help="Use cached queries"
    )
    query_parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Output file name to write openalex cohort",
    )
    args = openalex_parser.parse_args()
    if hasattr(args, "func"):
        setup_logger(args.verbosity)
        args.func(args)
    else:
        openalex_parser.print_help()
