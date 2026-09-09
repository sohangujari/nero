"""play_music: choosing a player, remembering the choice, and driving it.

The interesting behaviour is not "does it send `pause`" — it is what happens
when the machine has two players, or none, or one that isn't running yet.
"""

import asyncio
import subprocess

import pytest

from nero.skills.play_music.server import (
    ACTIONS,
    LinuxController,
    MacOSController,
    Outcome,
    PlayMusicSkill,
    match,
)


def fake_runner(responses=None):
    """A runner yielding scripted results, recording the commands it saw."""
    responses = list(responses or [])
    calls = []

    def run(cmd):
        calls.append(cmd)
        code, stdout, stderr = responses.pop(0) if responses else (0, "done", "")
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr=stderr)

    run.calls = calls
    return run


def apps(tmp_path, *names):
    """A fake /Applications holding `names`."""
    for name in names:
        (tmp_path / f"{name}.app").mkdir(exist_ok=True)
    return (tmp_path,)


def run_skill(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


class Stub:
    """A controller with a known set of players and no subprocesses."""

    def __init__(self, available, running=()):
        self._available = list(available)
        self._running = list(running)
        self.calls = []

    def available(self):
        return list(self._available)

    def running(self):
        return list(self._running)

    def control(self, action, player):
        self.calls.append((action, player))
        return Outcome(True, f"did {action} in {player}")


class TestMeta:
    def test_metadata(self):
        skill = PlayMusicSkill()
        assert skill.meta.name == "play_music"
        assert skill.meta.requires_network is False
        assert skill.meta.permission_tier == "state_changing"
        assert skill.meta.input_schema["properties"]["action"]["enum"] == list(ACTIONS)

    def test_the_app_argument_is_optional(self):
        """The model must be able to call this without naming a player — that
        is the whole point of remembering one."""
        schema = PlayMusicSkill().meta.input_schema
        assert "app" in schema["properties"]
        assert schema["required"] == ["action"]


class TestChoosingAPlayer:
    def test_one_player_installed_is_never_a_question(self, tmp_path):
        controller = Stub(["Music"])
        assert run_skill(PlayMusicSkill(controller=controller), action="play") == (
            "did play in Music"
        )

    def test_two_players_and_none_playing_asks_which(self):
        result = run_skill(PlayMusicSkill(controller=Stub(["Music", "Spotify"])), action="play")
        assert "Music, Spotify" in result
        assert "`app`" in result

    def test_two_players_but_only_one_playing_does_not_ask(self):
        """Asking when the answer is obvious is its own kind of broken."""
        controller = Stub(["Music", "Spotify"], running=["Spotify"])
        assert "Spotify" in run_skill(PlayMusicSkill(controller=controller), action="pause")

    def test_a_named_player_wins_over_everything(self):
        controller = Stub(["Music", "Spotify"])
        result = run_skill(
            PlayMusicSkill(controller=controller, preferred_app="Music"),
            action="play",
            app="Spotify",
        )
        assert result == "did play in Spotify"

    def test_a_remembered_player_is_used_without_asking(self):
        controller = Stub(["Music", "Spotify"])
        skill = PlayMusicSkill(controller=controller, preferred_app="Spotify")
        assert run_skill(skill, action="play") == "did play in Spotify"

    def test_a_remembered_player_that_was_uninstalled_falls_through(self):
        """A stale preference must not strand the skill on a player that is
        no longer there."""
        controller = Stub(["Music"])
        skill = PlayMusicSkill(controller=controller, preferred_app="Spotify")
        assert run_skill(skill, action="play") == "did play in Music"

    def test_a_player_they_do_not_have_says_what_they_do_have(self):
        result = run_skill(
            PlayMusicSkill(controller=Stub(["Music"])), action="play", app="Tidal"
        )
        assert "Tidal" in result and "Music" in result

    def test_no_player_at_all_is_reported(self):
        assert "install" in run_skill(PlayMusicSkill(controller=Stub([])), action="play")


class TestRememberingTheChoice:
    def test_a_named_player_is_remembered(self):
        saved = []
        skill = PlayMusicSkill(controller=Stub(["Music", "Spotify"]), on_app_chosen=saved.append)
        run_skill(skill, action="play", app="spotify")
        assert saved == ["Spotify"]

    def test_a_player_picked_by_default_is_not_remembered(self):
        """Only an explicit choice is a preference. Recording a fallback would
        silently freeze a guess into config."""
        saved = []
        skill = PlayMusicSkill(controller=Stub(["Music"]), on_app_chosen=saved.append)
        run_skill(skill, action="play")
        assert saved == []

    def test_a_choice_that_failed_is_not_remembered(self):
        """Remembering a player that just failed would make every later request
        fail the same way, with nobody asked again."""

        class Failing(Stub):
            def control(self, action, player):
                return Outcome(False, "Spotify is not installed properly")

        saved = []
        skill = PlayMusicSkill(
            controller=Failing(["Music", "Spotify"]), on_app_chosen=saved.append
        )
        run_skill(skill, action="play", app="Spotify")
        assert saved == []

    def test_a_later_explicit_choice_replaces_the_earlier_one(self):
        saved = []
        skill = PlayMusicSkill(
            controller=Stub(["Music", "Spotify"]),
            preferred_app="Music",
            on_app_chosen=saved.append,
        )
        run_skill(skill, action="play", app="Spotify")
        assert saved == ["Spotify"]


class TestMatching:
    @pytest.mark.parametrize(
        "spoken,expected",
        [
            ("Spotify", "Spotify"),
            ("spotify", "Spotify"),
            ("  SPOTIFY ", "Spotify"),
            ("apple music", "Music"),
            ("Apple Music", "Music"),
            ("music", "Music"),
        ],
    )
    def test_a_player_is_found_however_it_was_said(self, spoken, expected):
        assert match(spoken, ["Music", "Spotify"]) == expected

    def test_an_unknown_name_matches_nothing(self):
        assert match("Winamp", ["Music", "Spotify"]) is None


class TestMacOS:
    def test_installed_players_are_found_by_bundle_not_by_probing(self, tmp_path):
        """A stat() answers this. An osascript probe costs ~35 ms to learn that
        an app which was never installed is not running."""
        runner = fake_runner()
        controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "Music", "Spotify"))
        assert controller.available() == ["Music", "Spotify"]
        assert runner.calls == []

    def test_an_uninstalled_player_is_never_offered(self, tmp_path):
        controller = MacOSController(runner=fake_runner(), app_dirs=apps(tmp_path, "Music"))
        assert controller.available() == ["Music"]

    def test_running_asks_about_every_player_in_one_call(self, tmp_path):
        runner = fake_runner([(0, "false, true", "")])
        controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "Music", "Spotify"))
        assert controller.running() == ["Spotify"]
        assert len(runner.calls) == 1

    def test_play_launches_the_app_rather_than_refusing(self, tmp_path):
        """'Play some music' should start the player. Telling the user to go
        and open it themselves is the one answer nobody wants."""
        runner = fake_runner([(0, "", "")])
        controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "Music"))
        outcome = controller.control("play", "Music")
        assert outcome.ok
        script = runner.calls[0][-1]
        assert script == 'tell application "Music" to play'
        assert "is running" not in script  # no pre-check: one subprocess, not two

    def test_pause_checks_and_acts_in_a_single_call(self, tmp_path):
        runner = fake_runner([(0, "done", "")])
        controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "Music"))
        assert controller.control("pause", "Music").ok
        assert len(runner.calls) == 1
        assert "is running" in runner.calls[0][-1]

    def test_pausing_something_that_is_not_running_says_so(self, tmp_path):
        runner = fake_runner([(0, "idle", "")])
        controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "Music"))
        outcome = controller.control("pause", "Music")
        assert not outcome.ok
        assert "isn't running" in outcome.message

    def test_next_and_previous_use_itunes_vocabulary(self, tmp_path):
        for action, verb in (("next", "next track"), ("previous", "previous track")):
            runner = fake_runner([(0, "done", "")])
            controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "Music"))
            controller.control(action, "Music")
            assert verb in runner.calls[0][-1]

    def test_vlc_uses_its_own_vocabulary(self, tmp_path):
        """VLC is the common exception: `next`, not `next track`."""
        runner = fake_runner([(0, "done", "")])
        controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "VLC"))
        controller.control("next", "VLC")
        assert "next track" not in runner.calls[0][-1]

    def test_an_applescript_failure_carries_its_own_message(self, tmp_path):
        runner = fake_runner([(1, "", "Application isn't running.")])
        controller = MacOSController(runner=runner, app_dirs=apps(tmp_path, "Music"))
        outcome = controller.control("play", "Music")
        assert not outcome.ok and "isn't running" in outcome.message


class TestLinux:
    def linux(self, responses, playerctl="/usr/bin/playerctl"):
        return LinuxController(
            runner=fake_runner(responses), which=lambda name: playerctl
        )

    def test_missing_playerctl_is_reported(self):
        controller = LinuxController(runner=fake_runner(), which=lambda name: None)
        assert controller.available() == []
        assert "playerctl" in run_skill(PlayMusicSkill(controller=controller), action="play")

    def test_players_are_listed_from_playerctl(self):
        controller = self.linux([(0, "spotify\nvlc\n", "")])
        assert controller.available() == ["spotify", "vlc"]

    def test_a_bus_instance_suffix_is_not_part_of_the_name(self):
        """playerctl reports "spotify.instance123"; a person says "spotify"."""
        controller = self.linux([(0, "spotify.instance7\nspotify.instance9\n", "")])
        assert controller.available() == ["spotify"]

    def test_no_players_found_is_reported(self):
        controller = self.linux([(1, "", "No players found")])
        assert "install" in run_skill(PlayMusicSkill(controller=controller), action="play")

    def test_the_action_goes_to_the_named_player(self):
        controller = self.linux([(0, "", "")])
        outcome = controller.control("next", "spotify")
        assert outcome.ok

    def test_playerctl_failure_is_reported(self):
        controller = self.linux([(1, "", "boom")])
        outcome = controller.control("play", "spotify")
        assert not outcome.ok and "boom" in outcome.message


class TestSkillDispatch:
    def test_rejects_unknown_action(self):
        result = run_skill(PlayMusicSkill(controller=Stub(["Music"])), action="yeet")
        assert "play, pause, next, previous" in result

    def test_rejects_missing_action(self):
        assert "play, pause" in run_skill(PlayMusicSkill())

    def test_action_is_case_insensitive(self):
        controller = Stub(["Music"])
        run_skill(PlayMusicSkill(controller=controller), action="PLAY")
        assert controller.calls == [("play", "Music")]

    def test_unsupported_platform(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Plan9")
        result = run_skill(PlayMusicSkill(), action="play")
        assert "Error" in result and "Plan9" in result

    @pytest.mark.parametrize("action", ACTIONS)
    def test_all_actions_reach_the_controller(self, action):
        controller = Stub(["Music"])
        run_skill(PlayMusicSkill(controller=controller), action=action)
        assert controller.calls == [(action, "Music")]
