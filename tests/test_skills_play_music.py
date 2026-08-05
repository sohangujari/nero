import asyncio
import subprocess

import pytest

from nero.skills.play_music.server import (
    ACTIONS,
    LinuxController,
    MacOSController,
    PlayMusicSkill,
)


def fake_runner(responses):
    """Return a runner yielding scripted results, recording the commands it saw."""
    calls = []

    def run(cmd):
        calls.append(cmd)
        result = responses.pop(0) if responses else (0, "", "")
        code, stdout, stderr = result
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr=stderr)

    run.calls = calls
    return run


def run_skill(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


class TestMeta:
    def test_metadata(self):
        skill = PlayMusicSkill()
        assert skill.meta.name == "play_music"
        assert skill.meta.requires_network is False
        assert skill.meta.permission_tier == "state_changing"
        assert skill.meta.input_schema["properties"]["action"]["enum"] == list(ACTIONS)


class TestMacOS:
    def test_controls_spotify_when_running(self):
        runner = fake_runner([(0, "true", "")])
        result = MacOSController(runner=runner).control("play")
        assert "Spotify" in result
        # First call probes; second call acts.
        assert any("is running" in " ".join(cmd) for cmd in runner.calls)
        assert any("to play" in " ".join(cmd) for cmd in runner.calls)

    def test_falls_back_to_music_when_spotify_is_not_running(self):
        runner = fake_runner([(0, "false", ""), (0, "true", "")])
        result = MacOSController(runner=runner).control("pause")
        assert "Music" in result

    def test_no_player_running_is_reported(self):
        runner = fake_runner([(0, "false", ""), (0, "false", "")])
        result = MacOSController(runner=runner).control("play")
        assert "nothing is playing" in result.lower()

    def test_next_uses_next_track(self):
        runner = fake_runner([(0, "true", "")])
        MacOSController(runner=runner).control("next")
        assert any("next track" in " ".join(cmd) for cmd in runner.calls)

    def test_previous_uses_previous_track(self):
        runner = fake_runner([(0, "true", "")])
        MacOSController(runner=runner).control("previous")
        assert any("previous track" in " ".join(cmd) for cmd in runner.calls)


class TestLinux:
    def test_missing_playerctl_is_reported(self):
        controller = LinuxController(runner=fake_runner([]), which=lambda name: None)
        assert "playerctl" in controller.control("play")

    def test_no_players_found_is_reported(self):
        runner = fake_runner([(1, "", "No players found")])
        controller = LinuxController(runner=runner, which=lambda name: "/usr/bin/playerctl")
        assert "nothing is playing" in controller.control("play").lower()

    def test_sends_the_action(self):
        runner = fake_runner([(0, "Playing", ""), (0, "", "")])
        controller = LinuxController(runner=runner, which=lambda name: "/usr/bin/playerctl")
        result = controller.control("next")
        assert runner.calls[-1] == ["playerctl", "next"]
        assert "next" in result.lower()

    def test_playerctl_failure_is_reported(self):
        runner = fake_runner([(0, "Playing", ""), (1, "", "boom")])
        controller = LinuxController(runner=runner, which=lambda name: "/usr/bin/playerctl")
        assert "boom" in controller.control("play")


class TestSkillDispatch:
    def test_rejects_unknown_action(self):
        class Unused(MacOSController):
            def control(self, action):
                raise AssertionError("must not be reached")

        result = run_skill(PlayMusicSkill(controller=Unused(runner=fake_runner([]))), action="yeet")
        assert "play, pause, next, previous" in result

    def test_rejects_missing_action(self):
        assert "play, pause" in run_skill(PlayMusicSkill())

    def test_uses_the_injected_controller(self):
        class Stub:
            def control(self, action):
                return f"did {action}"

        assert run_skill(PlayMusicSkill(controller=Stub()), action="play") == "did play"

    def test_action_is_case_insensitive(self):
        class Stub:
            def control(self, action):
                return f"did {action}"

        assert run_skill(PlayMusicSkill(controller=Stub()), action="PLAY") == "did play"

    def test_unsupported_platform(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Plan9")
        result = run_skill(PlayMusicSkill(), action="play")
        assert "Error" in result and "Plan9" in result

    @pytest.mark.parametrize("action", ACTIONS)
    def test_all_actions_reach_the_controller(self, action):
        seen = []

        class Stub:
            def control(self, action):
                seen.append(action)
                return "ok"

        run_skill(PlayMusicSkill(controller=Stub()), action=action)
        assert seen == [action]
