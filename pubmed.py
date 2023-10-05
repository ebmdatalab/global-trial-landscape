import argparse
import logging
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import sys
from functools import partial
from io import StringIO

import matplotlib.pyplot as plt
import pandas

from setup import get_env_setting, get_full_parser, get_verbosity_parser, setup_logger
from utils import (
    REGISTRY_MAP,
    create_session,
    filter_unindexed,
    load_trials,
    query,
)


def analyse_metadata(args):
    input_file = args.input_file
    output_file = args.output_file
    df = pandas.read_csv(
        input_file,
        parse_dates=["Date_enrollment", "epub_date", "journal_date"],
        index_col=[0],
        dtype={"pmid": str},
    )
    df = df[~(df.title.str.contains("protocol", flags=re.IGNORECASE) is True)]
    import code

    code.interact(local=locals())


def trials_in_pubmed(args):
    input_file = args.input_file
    output_file = args.output_file
    n = args.chunk_size

    session = create_session("pubmed_cache", use_cache=False)
    unique_id = ["trial_id"]
    trials = load_trials(input_file, unique_id, True, False)
    already_processed = load_trials(output_file, unique_id, True, False)
    remaining = filter_unindexed(trials, already_processed, unique_id)
    chunks = [remaining.iloc[i : i + n] for i in range(0, remaining.shape[0], n)]
    for index, chunk in enumerate(chunks):
        logging.info(f"{index}/{len(chunks)}")
        # Max at 3 with apikey
        with mp.Pool(3) as pool:
            pmids = pool.map(
                partial(process_pubmed_search, session=session), chunk.trial_id
            )
        pmids = pandas.Series(pmids)
        pmids.name = "pmids"
        combined = pandas.concat([chunk.reset_index(drop=True), pmids], axis=1)
        # Drop any rows that had an error
        # Since we are using map, we lose the index and cannot drop missing index
        combined.dropna(subset="pmids")
        combined.explode("pmids").to_csv(
            output_file, mode="a", header=not output_file.exists(), index=False
        )


def process_pubmed_search(trial_id, session):
    resp = query_pubmed_search(trial_id, session, protocol_str=None)
    search_result = resp["esearchresult"]
    c = int(search_result.get("count", 0))
    if resp.get("error"):
        logging.error(f"There was an error {resp} for {trial_id}")
        return None
    elif c < 1:
        logging.info(f"Found no results searching for {trial_id}")
        return []
    else:
        return search_result["idlist"]


def split_numeric(trial_id):
    # These trial IDs are too simple to split (likely false positives)
    if (
        "PER" in trial_id
        or "SLCTR" in trial_id
        or "NTR" in trial_id
        or "NL" in trial_id
    ):
        return None
    # They do not seem to be split up at all
    elif "chictr" in trial_id.lower():
        return None
    elif "JPRN" in trial_id:
        # TODO: could additionally split JacpiCTI-### with an AND
        return trial_id.split("-", 1)[1]
    m = re.search(r"(\D+)(\d.*)", trial_id)
    if m:
        match_id = m.group(2)
        if "EUCTR" in trial_id:
            # TODO: also allow for CTXXX, Eudra-CT-XXX
            # Remove the country code and replace with a wildcard
            return match_id.rsplit("-", 1)[0] + "*"
        else:
            return match_id
    else:
        return None


def query_pubmed_search(trial_id, session, completion_date=None, protocol_str=None):
    def make_url(trial_id, completion_date, protocol_str):
        alternate = split_numeric(trial_id)
        url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        url += "esearch.fcgi?db=pubmed&retmode=json&term="
        url += f"({trial_id}[si] OR {trial_id}[Title/Abstract]) "
        if alternate:
            url += f"OR ({alternate}[si] OR {alternate}[Title/Abstract]) "
        if completion_date:
            url += f"AND ({completion_date.strftime('%Y/%m/%d')}[pdat] : "
            url += "3000[pdat]) "
        if protocol_str:
            url += "AND" + protocol_str
        return url

    url = make_url(trial_id, completion_date, protocol_str)
    params = {"api_key": get_env_setting("NCBI_API_KEY")}
    try:
        result = query(url, session, params)
    except Exception as e:
        logging.error(e)
        return {"esearchresult": {"count": -1}, "error": e}
    return result


def add_pubmed_metadata(args):
    input_file = args.input_file
    output_file = args.output_file
    n = args.chunk_size

    df = pandas.read_csv(input_file, dtype={"pmids": "str"})
    df["source"] = df.trial_id.str[0:3].str.upper()
    df.loc[df.source.str.startswith("NL"), "source"] = "NTR"
    unique_pmids = df.pmids.dropna().unique()
    try:
        sys.path.insert(1, os.path.dirname(shutil.which("xtract")))
        import edirect
    except Exception:
        logging.error("Is edirect installed?")

    chunks = [unique_pmids[i : i + n] for i in range(0, len(unique_pmids), n)]
    assert len(chunks[0]) < 11000
    chunk_metadata = ""
    for index, chunk in enumerate(chunks):
        logging.info(f"{index}/{len(chunks)}")
        cmd = f"epost -db pubmed -id {','.join(chunk)} | efetch -format xml | xtract -pattern PubmedArticle -def '' -sep '|' -tab '%%' -element MedlineCitation/PMID -element ArticleTitle -element PublicationType -block Journal -sep '-' -tab '%%' -element Year,Month -block ArticleDate -sep '-' -element Year,Month,Day"
        logging.info(cmd)
        out = edirect.pipeline(cmd)
        assert out != ""
        logging.info(out)
        # NOTE: depending on how long this takes, we could write intermediate to a file
        chunk_metadata += "\n" + out
    metadata = pandas.read_csv(
        StringIO(chunk_metadata),
        delimiter="%%",
        names=["pmids", "title", "pub_types", "journal_date", "epub_date"],
        dtype={"pmids": "str"},
    )
    df = df.merge(metadata, on="pmids", how="left")
    df.to_csv(output_file)


def reported_over_time(args):
    input_file = args.input_file
    df = pandas.read_csv(input_file)

    fig, ax = plt.subplots(figsize=(12, 6))
    df["enrollment_year"] = pandas.to_datetime(df.Date_enrollment).dt.strftime("%Y")
    df["source"] = df.source.map(REGISTRY_MAP)
    counts = df.groupby(["enrollment_year", "source"]).agg(
        {"trial_id": "count", "pmids": "count"}
    )
    counts["pcnt"] = 100 * (counts.pmids / counts.trial_id)
    to_plot = counts.reset_index().pivot(index="source", columns=["enrollment_year"])[
        "pcnt"
    ]
    to_plot = to_plot.sort_values("2014", ascending=False)
    to_plot.plot.bar(ax=ax)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Enrollment Year")
    plt.title(
        "Percent of trials with trial id in Pubmed Accession or Abstract by registry"
    )
    plt.xlabel("Registry")
    plt.ylabel("Percent (%)")
    plt.xticks(rotation=45)
    plt.savefig("percent_reported.png", bbox_inches="tight")


if __name__ == "__main__":
    verb = get_verbosity_parser()
    parent = get_full_parser()
    pubmed_parser = argparse.ArgumentParser()
    subparsers = pubmed_parser.add_subparsers()

    query_parser = subparsers.add_parser("query", parents=[parent])
    query_parser.set_defaults(func=trials_in_pubmed)

    metadata_parser = subparsers.add_parser("metadata", parents=[parent])
    metadata_parser.set_defaults(func=add_pubmed_metadata)

    analyse_parser = subparsers.add_parser("analyse", parents=[parent])
    analyse_parser.set_defaults(func=analyse_metadata)

    reported_parser = subparsers.add_parser("percent-reported", parents=[verb])
    reported_parser.add_argument(
        "--input-file",
        required=True,
        type=pathlib.Path,
        help="Cohort file with discovered pmids",
    )
    reported_parser.set_defaults(func=reported_over_time)

    args = pubmed_parser.parse_args()
    if hasattr(args, "func"):
        setup_logger(args.verbosity)
        args.func(args)
    else:
        pubmed_parser.print_help()
