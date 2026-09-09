"""set_volume (nero/skills/volume/server.py).

People say "turn it up a bit", not "set output volume to 29". The model turns
the words into a number; this skill's job is to apply that number sensibly to
whatever the volume happens to be right now.
"""

import asyncio
import subprocess

import pytest

from nero.skills.volume.server import (
    NUDGE,
    STEP,
    LinuxVolume,
    MacOSVolume,
    SetVolumeSkill,
    VolumeController,
    clamp,
    describe,
)


def run(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


class Fake(VolumeController):
    """A speaker with a remembered level."""

    def __init__(self, level=50, muted=False, readable=True, writable=True):
        self.level, self.muted, self._readable, self._writable = level, muted, readable, writable
        self.writes = []

    def read(self):
        return (self.level, self.muted) if self._readable else None

    def write(self, level, muted):
        self.writes.append((level, muted))
        if not self._writable:
            return False
        self.level, self.muted = level, muted
        return True


class TestMeta:
    def test_metadata(self):
        meta = SetVolumeSkill().meta
        assert meta.name == "set_volume"
        assert meta.requires_network is False
        assert meta.permission_tier == "state_changing"
        assert meta.category == "Apps"

    def test_every_argument_is_optional_so_one_can_be_given_alone(self):
        schema = SetVolumeSkill().meta.input_schema
        assert schema["required"] == []
        assert set(schema["properties"]) == {"level", "change", "mute"}


class TestAbsolute:
    @pytest.mark.parametrize("asked,expected", [(0, 0), (30, 30), (100, 100)])
    def test_a_level_is_set_exactly(self, asked, expected):
        speaker = Fake(level=50)
        run(SetVolumeSkill(controller=speaker), level=asked)
        assert speaker.level == expected

    @pytest.mark.parametrize("asked,expected", [(-20, 0), (250, 100)])
    def test_a_level_out_of_range_is_clamped_not_refused(self, asked, expected):
        speaker = Fake()
        run(SetVolumeSkill(controller=speaker), level=asked)
        assert speaker.level == expected

    def test_setting_a_level_above_zero_unmutes(self):
        """Turning it up while muted and staying silent is technically obedient
        and practically useless."""
        speaker = Fake(level=0, muted=True)
        run(SetVolumeSkill(controller=speaker), level=40)
        assert speaker.level == 40 and speaker.muted is False


class TestRelative:
    def test_a_nudge_is_applied_to_the_current_level(self):
        """"A bit louder" is meaningless without knowing where it is now."""
        speaker = Fake(level=19)
        run(SetVolumeSkill(controller=speaker), change=NUDGE)
        assert speaker.level == 19 + NUDGE

    def test_a_bigger_step_moves_further(self):
        speaker = Fake(level=40)
        run(SetVolumeSkill(controller=speaker), change=STEP)
        assert speaker.level == 40 + STEP

    def test_lowering_below_zero_stops_at_silent(self):
        speaker = Fake(level=5)
        run(SetVolumeSkill(controller=speaker), change=-STEP)
        assert speaker.level == 0

    def test_raising_past_the_top_stops_at_maximum(self):
        speaker = Fake(level=95)
        run(SetVolumeSkill(controller=speaker), change=STEP)
        assert speaker.level == 100

    def test_turning_it_up_while_muted_unmutes(self):
        speaker = Fake(level=30, muted=True)
        run(SetVolumeSkill(controller=speaker), change=NUDGE)
        assert speaker.muted is False

    def test_turning_it_down_while_muted_stays_muted(self):
        """They did not ask to hear anything."""
        speaker = Fake(level=30, muted=True)
        run(SetVolumeSkill(controller=speaker), change=-NUDGE)
        assert speaker.muted is True


class TestMute:
    def test_mute_silences_without_losing_the_level(self):
        speaker = Fake(level=60)
        assert run(SetVolumeSkill(controller=speaker), mute=True) == "Muted."
        assert speaker.muted is True and speaker.level == 60

    def test_unmute_restores_sound(self):
        speaker = Fake(level=60, muted=True)
        run(SetVolumeSkill(controller=speaker), mute=False)
        assert speaker.muted is False and speaker.level == 60

    def test_unmuting_something_at_zero_gives_it_somewhere_to_go(self):
        """Unmuting into silence looks exactly like the unmute failing."""
        speaker = Fake(level=0, muted=True)
        run(SetVolumeSkill(controller=speaker), mute=False)
        assert speaker.level > 0 and speaker.muted is False


class TestReporting:
    @pytest.mark.parametrize(
        "level,muted,expected",
        [(0, False, "silent"), (12, False, "12% (low)"), (50, False, "50%"),
         (95, False, "95% (loud)"), (50, True, "muted")],
    )
    def test_the_level_is_described_in_words_people_use(self, level, muted, expected):
        assert describe(level, muted) == expected

    def test_the_reply_says_where_it_ended_up(self):
        assert "70%" in run(SetVolumeSkill(controller=Fake(level=50)), change=STEP)

    def test_asking_for_nothing_explains_what_to_ask_for(self):
        assert "0-100" in run(SetVolumeSkill(controller=Fake()))

    def test_a_write_that_fails_is_not_reported_as_done(self):
        speaker = Fake(writable=False)
        assert "couldn't" in run(SetVolumeSkill(controller=speaker), level=50)

    def test_an_unsupported_platform_says_so(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Plan9")
        assert "Plan9" in run(SetVolumeSkill(), level=50)


class TestUnreadableVolume:
    """Windows cannot report the current level without a package Nero does not
    ship, so relative changes work through media keys and absolute ones say so
    rather than pretending."""

    def test_an_absolute_level_admits_it_cannot(self):
        result = run(SetVolumeSkill(controller=Fake(readable=False)), level=50)
        assert "left it alone" in result

    def test_a_controller_with_media_keys_can_still_nudge(self):
        class Keys(Fake):
            def __init__(self):
                super().__init__(readable=False)
                self.nudges = []

            def nudge(self, steps, mute):
                self.nudges.append((steps, mute))
                return True

        speaker = Keys()
        assert "up" in run(SetVolumeSkill(controller=speaker), change=STEP)
        assert speaker.nudges[0][0] > 0
        run(SetVolumeSkill(controller=speaker), change=-STEP)
        assert speaker.nudges[1][0] < 0


class TestMacOS:
    def runner(self, responses):
        calls = []

        def run_cmd(cmd):
            calls.append(cmd)
            code, out = responses.pop(0) if responses else (0, "")
            return subprocess.CompletedProcess(cmd, code, out, "")

        run_cmd.calls = calls
        return run_cmd

    def test_level_and_mute_are_read_together(self):
        runner = self.runner([(0, "42,false")])
        assert MacOSVolume(runner=runner).read() == (42, False)

    def test_a_muted_speaker_reads_as_muted(self):
        assert MacOSVolume(runner=self.runner([(0, "42,true")])).read() == (42, True)

    def test_unreadable_output_is_none_not_a_crash(self):
        for bad in [(0, "nonsense"), (1, ""), (0, "abc,false")]:
            assert MacOSVolume(runner=self.runner([bad])).read() is None

    def test_volume_and_mute_are_written_in_one_call(self):
        """Two calls would let the speaker jump to the new level for an instant
        before muting."""
        runner = self.runner([(0, "")])
        MacOSVolume(runner=runner).write(30, True)
        assert len(runner.calls) == 1
        assert "output volume 30 with output muted" in runner.calls[0][-1]


class TestLinux:
    def test_missing_pactl_reads_as_unknown(self):
        controller = LinuxVolume(runner=lambda cmd: None, which=lambda name: None)
        assert controller.read() is None
        assert controller.write(50, False) is False

    def test_a_percentage_is_pulled_out_of_pactl_output(self):
        def runner(cmd):
            out = "Volume: front-left: 45874 /  70% / -9.29 dB" if "get-sink-volume" in cmd else "Mute: no"
            return subprocess.CompletedProcess(cmd, 0, out, "")

        controller = LinuxVolume(runner=runner, which=lambda name: "/usr/bin/pactl")
        assert controller.read() == (70, False)


def test_clamp_never_leaves_the_range():
    for value in (-500, -1, 0, 50, 100, 101, 9999):
        assert 0 <= clamp(value) <= 100
