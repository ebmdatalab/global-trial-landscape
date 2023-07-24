import pandas
import vcr

from query_ror import (
    create_session,
    get_empty_results,
    process_ror_json,
    remove_noninformative,
)


def make_query(fake_sponsor_row):
    sponsor = fake_sponsor_row["site"]
    session = create_session()
    results = get_empty_results()
    results = process_ror_json(sponsor, session, results, fake_sponsor_row)
    return results


@vcr.use_cassette("tests/fixtures/vcr_casettes/chosen.yaml")
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
        )["name"]
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
        )["name"]
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
    assert result["count_non_matches"] == 3
    assert result["non_matches"][0] == {
        "Centre Hospitalier Universitaire Dijon Bourgogne": 0.58
    }


def test_remove_noninformative():
    # Remove non-informative (case insensitive)
    series = pandas.Series(["GSK Investigational Site", "GSK investigational Site"])
    assert (remove_noninformative(series) == "GSK").all()
