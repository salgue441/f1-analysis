import pytest

from pitwall.quality import Expectation, QualityGateError, apply_expectations

EXPECTATIONS = [
    Expectation("positive", "x > 0"),
    Expectation("small", "x < 100", severity="warn"),
]


def test_error_rows_are_quarantined_and_warn_rows_kept(spark):
    df = spark.createDataFrame([(1,), (-5,), (500,), (None,)], "x int")

    valid, quarantined, report = apply_expectations(df, EXPECTATIONS)

    assert sorted(r.x for r in valid.collect()) == [1, 500]
    assert report.total_rows == 4
    assert report.quarantined_rows == 2  # -5 and NULL (NULL counts as failure)
    assert report.failures == {"positive": 2, "small": 2}  # 500 and NULL
    assert "_failed_expectations" in quarantined.columns


def test_gate_raises_when_ratio_exceeded(spark):
    df = spark.createDataFrame([(1,), (-1,)], "x int")
    with pytest.raises(QualityGateError):
        apply_expectations(df, EXPECTATIONS, max_quarantine_ratio=0.1)
