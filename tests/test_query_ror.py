import numpy
import pandas
import vcr

from query_ror import (
    apply_ror,
    get_empty_results,
    merge_and_update,
    process_ror_json,
)
from utils import create_session, find_city_country, remove_noninformative


def make_query(fake_sponsor_row):
    sponsor = fake_sponsor_row["site"]
    session = create_session()
    results = get_empty_results()
    results = process_ror_json(sponsor, session, results, fake_sponsor_row)
    return results


@vcr.use_cassette("tests/fixtures/vcr_casettes/acronym.yaml")
def test_process_ror_json_chosen():
    assert (
        make_query(
            pandas.Series(
                {
                    "NCTId": "NCT05946603",
                    "site": "MSKCC",
                    "city": "New York",
                    "country": "US",
                }
            )
        )["name_resolved"]
        == "Memorial Sloan Kettering Cancer Center"
    )


@vcr.use_cassette("tests/fixtures/vcr_casettes/city_chosen.yaml")
def test_process_ror_json_city_chosen():
    assert (
        make_query(
            pandas.Series(
                {
                    "NCTId": "NCT05946772",
                    "site": "Department of Cardiology, Charité - Universitätsmedizin Berlin, Campus Benjamin Franklin",
                    "city": "Berlin",
                    "country": "DE",
                }
            )
        )["name_resolved"]
        == "Charité - Universitätsmedizin Berlin"
    )


@vcr.use_cassette("tests/fixtures/vcr_casettes/no_match_options.yaml")
def test_process_ror_json_no_match_options():
    result = make_query(
        pandas.Series(
            {
                "NCTId": "NCT05945199",
                "site": "CHU Dijon Bourgogne",
                "city": "Dijon",
                "country": "FR",
            }
        )
    )
    assert result["count_non_matches"] == 10
    assert result["non_matches"][0] == {"Colorado Heights University": 0.9}


@vcr.use_cassette("tests/fixtures/vcr_casettes/elizabeths.yaml")
def test_process_ror_elizabeth():
    result = make_query(
        pandas.Series(
            {
                "NCTId": "NCT03473223",
                "site": "4580004 - Queen Elizabeth Hospital II",
                "city": "Kota Kinabalu",
                "country": "MY",
            }
        )
    )
    assert result["ror"] == "https://ror.org/05pgywt51"


@vcr.use_cassette(cassette_library_dir="tests/fixtures/vcr_casettes")
def test_process_substring():
    session = create_session()
    result = apply_ror(
        pandas.Series(
            {
                "NCTId": "NCT05941169",
                "name": "Amsterdam Universitair Medische Centra - Academisch Medisch Centrum",
            }
        ),
        session,
    )
    assert result["ror"] == "https://ror.org/05grdyy37"


@vcr.use_cassette(cassette_library_dir="tests/fixtures/vcr_casettes")
def test_process_doublequote():
    session = create_session()
    result = apply_ror(
        pandas.Series(
            {
                "NCTId": "NCT03595553",
                "name": '"' + "ATTIKON University Hospital" + '"',  # noqa: ISC003
            }
        ),
        session,
    )
    # Do not resolve as Colorado University Hospital
    # Or any other "University Hospital"
    assert result["name_resolved"] is None


def test_remove_noninformative():
    # Remove non-informative (case insensitive)
    series = pandas.Series(["GSK Investigational Site", "GSK investigational Site"])
    assert (remove_noninformative(series) == "GSK").all()


def test_find_city_country():
    df = pandas.DataFrame(
        {
            "address": pandas.Series(
                [
                    "Bambolim, North Goa, Goa-403202North Goa GOA",
                    "Bangalore, Karnataka, 560099Bangalore KARNATAKA",
                    "Nagar New Delhi South DELHI",
                    "ICU 4 and 5 Casualty building Lok Nayak Hospital Central DELHI",
                    "Dr Rajendra Prasad Centre for Ophthalmic sciences, All india institute of medical sciences, new delhi-110029",
                    "Millennium Block, Punjagutta, - 500082Hyderabad TELANGANA",
                ]
            )
        }
    )
    find_city_country(df, "address", "IN", "india_states.txt")
    assert df.city[0] == "Bambolim"  # Ignore Goa (state)
    assert df.city[1] == "Bengaluru"  # Rename Bangalore
    assert df.city[2] == "New Delhi"  # Ignore South Delhi
    assert numpy.isnan(df.city[3])  # Ignore Central Delhi (actually New Delhi)
    assert df.city[4] == "New Delhi"  # We titleize new delhi
    assert df.city[5] == "Hyderabad"  # Regex should add spaces after numbers
    assert df.country[0] == "IN"  # Manually set country


def test_merge_and_update():
    df = pandas.DataFrame(
        {
            "trial_id": pandas.Series(["t1", "t5", "t7"]),
            "ror": pandas.Series(["https://1", "https://2", "https://1"]),
            "country": pandas.Series(["A", "B", "A"]),
            "lattitude": pandas.Series(["100", "-100"]),
        }
    )
    df = df.set_index(pandas.Index([1, 5, 7]))
    resolved = pandas.DataFrame(
        {
            "ror": pandas.Series(["https://1", "https://2"]),
            "lattitude": pandas.Series(["100", "-101"]),
            "longitude": pandas.Series(["10", "-10"]),
            "name_resolved": pandas.Series(["Name 1", "Name 2"]),
        }
    )
    merged = merge_and_update(df, resolved, on=["ror"])
    # preserve index
    assert list(merged.index) == [1, 5, 7]
    # make sure all the columns are there
    assert set(merged.columns) == set(
        ["trial_id", "ror", "country", "longitude", "name_resolved", "lattitude"]
    )
    # update an outdated value
    assert merged.lattitude.loc[5] == "-101"
    # single ror id gets mapped to both trials
    assert merged[merged.ror == "https://1"].trial_id.count() == 2
