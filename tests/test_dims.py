import datetime as dt

from pitwall.gold.dims import OPEN_END_DATE, build_dim_driver_team


def test_scd2_creates_new_spell_on_team_change_and_return(spark):
    d = dt.date
    entries = spark.createDataFrame(
        [
            ("lawson", "red_bull", d(2025, 3, 16), 1),
            ("lawson", "red_bull", d(2025, 3, 23), 2),
            ("lawson", "rb", d(2025, 4, 6), 3),
            ("lawson", "rb", d(2025, 4, 13), 4),
            ("lawson", "red_bull", d(2025, 5, 4), 5),  # hypothetical return
        ],
        "driver_id string, constructor_id string, race_date date, round int",
    )

    rows = build_dim_driver_team(entries).orderBy("valid_from").collect()

    assert [(r.constructor_id, r.valid_from, r.valid_to, r.is_current) for r in rows] == [
        ("red_bull", d(2025, 3, 16), d(2025, 4, 5), False),
        ("rb", d(2025, 4, 6), d(2025, 5, 3), False),
        ("red_bull", d(2025, 5, 4), OPEN_END_DATE, True),
    ]
    assert len({r.driver_team_key for r in rows}) == 3
    assert len({r.driver_key for r in rows}) == 1
