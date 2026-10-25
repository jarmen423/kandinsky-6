from __future__ import annotations

import pytest

from kandinsky.cli import _conditioning_inputs


def test_one_image_stays_a_single_path() -> None:
    conditioning = _conditioning_inputs(
        image="first.png",
        last_image=None,
        sample_audio=True,
        audio_file=None,
        video_file=None,
    )
    assert conditioning.image == "first.png"
    assert conditioning.audio is None
    assert conditioning.video is None


def test_last_image_is_one_first_and_last_pair() -> None:
    conditioning = _conditioning_inputs(
        image="first.png",
        last_image="last.png",
        sample_audio=True,
        audio_file=None,
        video_file=None,
    )
    assert conditioning.image == [["first.png", "last.png"]]


def test_audio_and_video_files_are_passed_through() -> None:
    audio = _conditioning_inputs(
        image=None,
        last_image=None,
        sample_audio=True,
        audio_file="speech.wav",
        video_file=None,
    )
    video = _conditioning_inputs(
        image=None,
        last_image=None,
        sample_audio=True,
        audio_file=None,
        video_file="silent.mp4",
    )
    assert audio.audio == "speech.wav"
    assert audio.video is None
    assert video.video == "silent.mp4"
    assert video.audio is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"last_image": "last.png"}, "--last-image requires --image"),
        (
            {"audio_file": "speech.wav", "video_file": "silent.mp4"},
            "not both",
        ),
        ({"image": "first.png", "audio_file": "speech.wav"}, "cannot be combined with --image"),
        ({"image": "first.png", "last_image": "last.png", "video_file": "silent.mp4"}, "cannot be combined with --image"),
        ({"audio_file": "speech.wav", "sample_audio": False}, "cannot be combined with --no-audio"),
        ({"video_file": "silent.mp4", "sample_audio": False}, "cannot be combined with --no-audio"),
    ],
)
def test_conditioning_flags_reject_invalid_combinations(kwargs: dict[str, object], match: str) -> None:
    defaults = {
        "image": None,
        "last_image": None,
        "sample_audio": True,
        "audio_file": None,
        "video_file": None,
    }
    with pytest.raises(ValueError, match=match):
        _conditioning_inputs(**(defaults | kwargs))
