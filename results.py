import argparse
import glob
import pathlib
from itertools import chain

import geopandas
import matplotlib.pyplot as plt
import pandas
import schemdraw
from schemdraw import flow

from setup import get_verbosity_parser, setup_logger
from utils import (
    add_suffix,
    convert_country_simple,
    map_who,
)


def get_resolved_filter(df):
    assert (df.individual.dtype == "bool") and (df.no_manual_match.dtype == "bool")
    return (~df.individual) & (~df.no_manual_match)


def normalize_name(df):
    resolved_filter = get_resolved_filter(df)
    df.loc[resolved_filter, "name_normalized"] = df.name_manual.fillna(df.name_resolved)
    return df


def clean_individual_manual(df):
    df["individual"] = df["individual"].fillna(0).astype(bool)
    df["no_manual_match"] = df["no_manual_match"].fillna(0).astype(bool)
    return df


def flowchart(args, df):
    df = clean_individual_manual(df)
    df = normalize_name(df)
    total = df.shape[0]

    individual = df.individual
    no_manual = df.no_manual_match | df.name.isnull()

    leftover = df[~(individual | no_manual)]

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

        d += flow.Arrow().theta(-135)
        d += (
            flow.Box(w=6, h=4)
            .label(f"ROR resolved\nn={ror_any.sum()}")
            .label(f"\n\n\n\n(n={ror_fixed.sum()} ROR manually corrected)", fontsize=8)
            .label(
                f"\n\n\n\n\n\n\n(n={ror_manual.sum()} ROR manually resolved)",
                fontsize=8,
            )
        )

        d.move_from(step2.S)
        d += flow.Arrow().theta(-45)
        d += flow.Box(w=6, h=4).label(f"Name manually resolved\nn={manual.sum()}")

        # Exclusions
        d.config(fontsize=8)
        d += flow.Arrow().right(d.unit / 4).at(step1.E)
        d += flow.Box(w=6, h=1).anchor("W").label(f"Individual\nn={individual.sum()}")
        d += flow.Arrow().right(d.unit / 4).at(step2.E)
        d += (
            flow.Box(w=6, h=1)
            .anchor("W")
            .label(f"No manual match\nn={no_manual.sum()}")
        )

    output_name = "_".join(df.source.unique())
    plt.savefig(f"{output_name}_flowchart")
    orgs = leftover.organization_type.fillna(leftover.manual_org_type)
    orgs.groupby(orgs).count().sort_values(ascending=False).to_csv(
        f"{output_name}_orgs.csv"
    )


def sponsor_map(args, df):
    df = clean_individual_manual(df)
    df = normalize_name(df)
    column_to_map = "trial_id"

    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.boundary.plot(ax=ax)

    df["iso_a3"] = convert_country_simple(df["country"], to="ISO3")
    counts = df.groupby("iso_a3")[column_to_map].size()
    merged = world.merge(counts, on="iso_a3")

    world.boundary.plot(ax=ax)
    merged.plot(
        column=column_to_map,
        cmap="YlOrRd",
        ax=ax,
        legend=True,
        legend_kwds={"label": "Number of Trials"},
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    output_name = "_".join(df.source.unique())
    registries = " and ".join(df.source.str.upper().unique())
    ax.set_title(f"{registries} Sponsors by Country")
    plt.savefig(f"{output_name}_map")


def site_map(args, df):
    output_name = "_".join(df.source.unique())
    df["who_region"] = map_who(df.country)

    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    world["who_region"] = map_who(world.iso_a3)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.boundary.plot(ax=ax)

    # Trial sites per WHO region (sites with multiple trials counted multiple times)
    counts = df.groupby("who_region").trial_id.count()
    merged = world.merge(counts, on="who_region")

    world.boundary.plot(ax=ax)
    merged.plot(
        column="trial_id",
        cmap="YlOrRd",
        ax=ax,
        legend=True,
        legend_kwds={"label": "Number of Trials"},
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    output_name = "_".join(df.source.unique())
    registries = " and ".join(df.source.str.upper().unique())
    ax.set_title(f"{registries} Trial Sites by WHO Region")
    plt.savefig(f"{output_name}_trial_map")


def write_table(df, output_dir, input_file, suffix):
    path = add_suffix(output_dir, input_file, suffix)
    df.to_csv(path)


def over_time(args, df):
    df["enrollment_year"] = pandas.to_datetime(df.enrollment_date).dt.strftime("%Y")
    df["registration_year"] = pandas.to_datetime(df.registration_date).dt.strftime("%Y")
    by_enroll_date = df.groupby("enrollment_year").trial_id.count().reset_index()
    by_reg_date = df.groupby("registration_year").trial_id.count().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        by_reg_date["registration_year"].astype(int) - 0.2,
        by_reg_date["trial_id"],
        0.4,
        label="Registration year",
    )
    ax.bar(
        by_enroll_date["enrollment_year"].astype(int) + 0.2,
        by_enroll_date["trial_id"],
        0.4,
        label="Enrollment year",
    )
    ax.legend(bbox_to_anchor=(1.3, 1.05))
    registries = " and ".join(df.source.str.upper().unique())
    ax.set_title(f"New trials enrolled or registered in {registries}")
    fig.tight_layout()
    output_name = "_".join(df.source.unique())
    plt.savefig(f"{output_name}_registrations_over_time")


def sites(args, df):
    counts = df.groupby("trial_id").trial_id.count()
    table = (
        (counts > 1)
        .value_counts()
        .rename(index={False: "Single Site", True: "Multi-Site"})
    )
    output_name = "_".join(df.source.unique())
    table.to_csv(f"{output_name}_single_multi.csv")


def get_path(*args):
    return pathlib.Path(*args).resolve()


def match_paths(pattern):
    return [get_path(x) for x in glob.glob(pattern)]


if __name__ == "__main__":
    verb = get_verbosity_parser()
    results_parser = argparse.ArgumentParser()
    subparsers = results_parser.add_subparsers()

    map_parser = subparsers.add_parser("map", parents=[verb])
    map_parser.add_argument(
        "--input-files",
        required=True,
        action="append",
        type=match_paths,
        help="One or more glob patterns for matching input files",
    )
    map_parser.set_defaults(func=sponsor_map)

    site_map_parser = subparsers.add_parser("site_map", parents=[verb])
    site_map_parser.add_argument(
        "--input-files",
        required=True,
        action="append",
        type=match_paths,
        help="One or more glob patterns for matching input files",
    )
    site_map_parser.set_defaults(func=site_map)

    flowchart_parser = subparsers.add_parser("flowchart", parents=[verb])
    flowchart_parser.add_argument(
        "--input-files",
        required=True,
        action="append",
        type=match_paths,
        help="One or more glob patterns for matching input files",
    )
    flowchart_parser.set_defaults(func=flowchart)

    time_parser = subparsers.add_parser("time", parents=[verb])
    time_parser.add_argument(
        "--input-files",
        required=True,
        action="append",
        type=match_paths,
        help="One or more glob patterns for matching input files",
    )
    time_parser.set_defaults(func=over_time)

    sites_parser = subparsers.add_parser("sites", parents=[verb])
    sites_parser.add_argument(
        "--input-files",
        required=True,
        action="append",
        type=match_paths,
        help="One or more glob patterns for matching input files",
    )
    sites_parser.set_defaults(func=sites)

    args = results_parser.parse_args()
    filenames_flat = list(chain.from_iterable(args.input_files))
    frames = []
    for input_file in filenames_flat:
        temp = pandas.read_csv(input_file)
        frames.append(temp)
    df = pandas.concat(frames, ignore_index=True)

    if hasattr(args, "func"):
        setup_logger(args.verbosity)
        args.func(args, df)
    else:
        results_parser.print_help()
