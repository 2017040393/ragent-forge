from ragent_forge.core.ingestion.table_serialization import serialize_table


def test_serialize_table_outputs_markdown_table() -> None:
    result = serialize_table(
        [
            ["Method", "Hit@1", "MRR"],
            ["lexical", "0.67", "0.78"],
            ["hybrid", "1.00", "1.00"],
        ]
    )

    assert result.serialization == "markdown_table"
    assert result.row_count == 3
    assert result.column_count == 3
    assert result.warning_kinds == ()
    assert result.text == "\n".join(
        [
            "| Method | Hit@1 | MRR |",
            "|---|---|---|",
            "| lexical | 0.67 | 0.78 |",
            "| hybrid | 1.00 | 1.00 |",
        ]
    )


def test_serialize_table_normalizes_cells_and_escapes_pipes() -> None:
    result = serialize_table(
        [
            [" Name ", "Notes"],
            [None, "alpha | beta"],
        ]
    )

    assert "|  | alpha \\| beta |" in result.text


def test_serialize_table_warns_for_inconsistent_rows() -> None:
    result = serialize_table(
        [
            ["Method", "Hit@1"],
            ["lexical"],
            ["hybrid", "1.00", "extra"],
        ]
    )

    assert result.serialization == "markdown_table"
    assert result.column_count == 3
    assert result.warning_kinds == ("table_malformed",)
    assert "| lexical |  |  |" in result.text


def test_serialize_table_returns_empty_result_for_empty_table() -> None:
    result = serialize_table([[None, " "], []])

    assert result.text == ""
    assert result.serialization == "empty"
    assert result.row_count == 0
    assert result.column_count == 0
    assert result.warning_kinds == ("table_empty",)
