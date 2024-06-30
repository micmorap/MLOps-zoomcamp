from unittest import mock

import main


def test_print(capsys):
    name = "test"
    event = {
        "bucket": "some-bucket",
        "name": name,
        "metageneration": "some-metageneration",
        "timeCreated": "0",
        "updated": "0",
    }

    context = mock.MagicMock()
    context.event_id = "some-id"
    context.event_type = "gcs-event"

    # Call tested function
    main.hello_gcs(event, context)
    out, err = capsys.readouterr()
    assert f"File: {name}\n" in out
