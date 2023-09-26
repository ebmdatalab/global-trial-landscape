import argparse
import logging
import os
import pathlib
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import pandas

from setup import get_base_parser, get_verbosity_parser, setup_logger
from utils import create_session


DEFAULT_PROTOCOL = "(clinicaltrial[Filter] NOT editorial)"


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
    cmd = f"esearch -db pubmed -query '({start_date}:{end_date}[dp]) AND ({protocol})' | efetch -format uid > {output_file}"
    logging.info(cmd)
    edirect.pipeline(cmd)
    return


# TODO: take in a dataset file rather than pmid list so we can add oa data
# TODO: to any df
# TODO: get accession and abstract from pubmed


def load_pmids(name):
    with open(name) as f:
        pmids = f.read().splitlines()
    ordered = sorted(set(pmids))
    return ordered


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
    index = 0
    for paper in response:
        import code

        code.interact(local=locals())
        paper_dict = defaultdict(str)
        paper_dict["pmid"] = paper["ids"]["pmid"].split("/")[-1]
        index = index + 1
        paper_dict["doi"] = paper["ids"].get("doi", None)
        paper_dict["openalex_paper"] = paper["ids"]["openalex"]
        for author in paper["authorships"]:
            # TODO: also get author name from openalex
            # TODO: orcid? is_corresponding?
            author_dict = paper_dict.copy()
            position = author["author_position"]
            if first_last and position == "middle":
                continue
            author_dict["position"] = position
            author_dict["affiliation_raw"] = author["raw_affiliation_string"]
            author_dict["openalex_author"] = author["author"].get("id", None)
            author_dict["display_name"] = author["author"].get("display_name", None)
            for institution in author["institutions"]:
                # One row per author affiliation
                institution_dict = author_dict.copy() | institution
                results.append(institution_dict)
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

    pmids = load_pmids(input_file)
    # https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists#addition-or
    # openalex allows up to 50 works in a query, to be safe use 40
    chunks = chunk_pmids(pmids, 40)
    for index, chunk in enumerate(chunks):
        results = query(
            chunk, session, email_address=email_address, use_cache=use_cache
        )
        pandas.DataFrame.from_dict(results).to_csv(
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
