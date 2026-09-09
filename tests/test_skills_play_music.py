"""play_music: choosing a player, remembering the choice, and driving it.

The interesting behaviour is not "does it send `pause`" — it is what happens
when the machine has two players, or none, or one that isn't running yet.
"""

import asyncio
import subprocess

import pytest

from nero.skills.play_music.server import (
    ACTIONS,
    applescript_string,
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

    def play_local(self, query, player):
        self.calls.append(("play_local", query, player))
        return Outcome(bool(self.local_hit), f"Playing {query} in {player}.")

    def open_track(self, url, player):
        self.calls.append(("open_track", url, player))
        return Outcome(True, f"Opened {url} in {player}.")

    local_hit = False


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


class TestPlayingANamedSong:
    """The ask: "play God's Plan by Drake". Before this the skill could only
    press play/pause on whatever was already queued."""

    def catalog(self, url="https://music.apple.com/x?i=1", label="God's Plan — Drake"):
        async def fake(query, client=None):
            return None if url is None else (url, label)

        return fake

    def skill(self, controller, catalog=None, **kw):
        skill = PlayMusicSkill(controller=controller, **kw)
        if catalog is not None:
            skill._catalog = lambda track, player: catalog(track)
        return skill

    def test_a_song_in_the_library_wins_over_the_catalogue(self):
        """The local copy is what plays. The lookup runs alongside it for speed
        and its answer is simply dropped."""
        controller = Stub(["Music"])
        controller.local_hit = True
        result = run_skill(
            self.skill(controller, catalog=self.catalog()), action="play", track="gods plan"
        )
        assert result == "Playing gods plan in Music."

    def test_a_song_in_the_library_still_plays_with_no_connection(self):
        """The lookup running in parallel must not be able to break the offline
        case — on a plane, a track you own still starts."""
        controller = Stub(["Music"])
        controller.local_hit = True

        async def offline(query, client=None):
            raise OSError("Network is unreachable")

        skill = PlayMusicSkill(controller=controller)
        skill._catalog = lambda track, player: offline(track)
        assert run_skill(skill, action="play", track="gods plan") == "Playing gods plan in Music."

    def test_a_dead_library_search_still_lets_the_catalogue_answer(self):
        """The mirror case: neither half may take the other down."""
        controller = Stub(["Music"])

        def boom(query, player):
            raise OSError("Music is wedged")

        controller.play_local = boom
        result = run_skill(
            self.skill(controller, catalog=self.catalog()), action="play", track="gods plan"
        )
        assert "https://music.apple.com/x?i=1" in result

    def test_the_library_search_and_the_lookup_run_at_the_same_time(self):
        """Done in sequence the lookup is pure added latency on every miss —
        and a miss is the normal case for a song the user does not own."""
        import time

        controller = Stub(["Music"])
        started: list = []

        def slow_local(query, player):
            started.append(("local", time.perf_counter()))
            time.sleep(0.3)
            return Outcome(False, "")

        controller.play_local = slow_local

        async def slow_catalog(query, client=None):
            started.append(("catalog", time.perf_counter()))
            await asyncio.sleep(0.3)
            return ("https://music.apple.com/x?i=1", "God's Plan — Drake")

        skill = self.skill(controller, catalog=slow_catalog)
        began = time.perf_counter()
        run_skill(skill, action="play", track="gods plan")
        elapsed = time.perf_counter() - began
        assert {name for name, _ in started} == {"local", "catalog"}
        assert elapsed < 0.55, f"ran in sequence: {elapsed:.2f}s for two 0.3s steps"

    def test_a_song_not_in_the_library_is_looked_up(self):
        controller = Stub(["Music"])
        skill = self.skill(controller, catalog=self.catalog())
        result = run_skill(skill, action="play", track="gods plan")
        assert "https://music.apple.com/x?i=1" in result
        assert ("open_track", "https://music.apple.com/x?i=1", "Music") in controller.calls

    def test_the_looked_up_title_is_reported_back(self):
        """"Playing God's Plan — Drake" tells the user the right song was found;
        echoing their own words back does not."""
        controller = Stub(["Music"])
        result = run_skill(self.skill(controller, catalog=self.catalog()),
                           action="play", track="gods plan")
        assert "God's Plan — Drake" in result

    def test_a_song_that_cannot_be_found_says_so(self):
        controller = Stub(["Music"])
        skill = self.skill(controller, catalog=self.catalog(url=None))
        result = run_skill(skill, action="play", track="asdkjhasd")
        assert "couldn't find" in result

    def test_naming_a_song_with_the_wrong_action_is_refused(self):
        """Skipping to the next track instead is a confusing way to say no."""
        result = run_skill(self.skill(Stub(["Music"])), action="next", track="gods plan")
        assert "only start a specific song with 'play'" in result

    def test_no_track_still_just_controls_playback(self):
        controller = Stub(["Music"])
        run_skill(self.skill(controller), action="pause")
        assert controller.calls == [("pause", "Music")]

    def test_the_chosen_player_is_the_one_told_to_play(self):
        controller = Stub(["Music", "Spotify"])
        skill = self.skill(controller, catalog=self.catalog(), preferred_app="Spotify")
        run_skill(skill, action="play", track="gods plan")
        assert controller.calls[0] == ("play_local", "gods plan", "Spotify")


class TestMacOSTrackPlayback:
    def controller(self, tmp_path, responses, *names):
        for name in names or ("Music",):
            (tmp_path / f"{name}.app").mkdir(exist_ok=True)
        runner = fake_runner(responses)
        return MacOSController(runner=runner, app_dirs=(tmp_path,)), runner

    def test_the_library_search_uses_the_apps_own_matcher(self, tmp_path):
        """`search` is the same matcher as the app's search field, so "gods plan
        drake" finds it without the apostrophe or exact casing."""
        controller, runner = self.controller(tmp_path, [(0, "God's Plan — Drake", "")])
        outcome = controller.play_local("gods plan drake", "Music")
        assert outcome.ok and "God's Plan — Drake" in outcome.message
        script = runner.calls[0][-1]
        assert 'search library playlist 1 for "gods plan drake"' in script

    def test_an_empty_library_result_is_a_miss_not_an_error(self, tmp_path):
        """A miss has to fall through to the catalogue, so it must not look like
        a failure."""
        controller, _ = self.controller(tmp_path, [(0, "", "")])
        outcome = controller.play_local("nothing here", "Music")
        assert outcome.ok is False

    def test_spotify_has_no_library_to_search(self, tmp_path):
        """Spotify's AppleScript has no search command; pretending otherwise
        would spend a subprocess to learn nothing."""
        controller, runner = self.controller(tmp_path, [], "Spotify")
        assert controller.play_local("gods plan", "Spotify").ok is False
        assert runner.calls == []

    def test_starting_a_song_gets_a_longer_leash_than_pressing_pause(self, tmp_path):
        """It launches the app if closed, loads a catalogue page, and waits for
        playback. Pressing pause does none of that."""
        from nero.skills.play_music.server import TIMEOUT_SECONDS, TRACK_TIMEOUT_SECONDS

        seen = []

        def runner(cmd, timeout=None):
            seen.append(timeout)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        (tmp_path / "Music.app").mkdir(exist_ok=True)
        controller = MacOSController(runner=runner, app_dirs=(tmp_path,))
        controller.control("pause", "Music")
        controller.play_local("gods plan", "Music")
        assert seen == [TIMEOUT_SECONDS, TRACK_TIMEOUT_SECONDS]
        assert TRACK_TIMEOUT_SECONDS > TIMEOUT_SECONDS

    def test_a_runner_that_takes_no_timeout_still_works(self, tmp_path):
        """Test doubles pass just the command; production passes a timeout."""
        (tmp_path / "Music.app").mkdir(exist_ok=True)
        controller = MacOSController(
            runner=lambda cmd: subprocess.CompletedProcess(cmd, 0, "A — B", ""),
            app_dirs=(tmp_path,),
        )
        assert controller.play_local("x", "Music").ok

    def test_it_waits_for_playback_rather_than_sleeping_a_fixed_time(self, tmp_path):
        """A catalogue page on a cold app can take seconds; a delay long enough
        to cover that would be dead air on every fast case."""
        controller, runner = self.controller(tmp_path, [(0, "A — B", "")])
        controller.open_track("https://music.apple.com/x?i=1", "Music")
        script = runner.calls[0][-1]
        assert "repeat" in script and "exit repeat" in script
        assert "delay 1.5" not in script

    def test_apple_music_is_told_what_actually_started(self, tmp_path):
        """`open location` navigates; whether playback starts is the app's
        decision. The script asks rather than assuming."""
        controller, runner = self.controller(tmp_path, [(0, "God's Plan — Drake", "")])
        outcome = controller.open_track("https://music.apple.com/x?i=1", "Music")
        assert outcome.ok and "Playing God's Plan — Drake in Music." == outcome.message
        assert "player state is playing" in runner.calls[0][-1]

    def test_a_link_that_opens_but_does_not_play_says_exactly_that(self, tmp_path):
        controller, _ = self.controller(tmp_path, [(0, "", "")])
        outcome = controller.open_track("https://music.apple.com/x?i=1", "Music")
        assert outcome.ok and "didn't start playing by itself" in outcome.message

    def test_spotify_is_told_to_play_the_track_not_shown_a_search(self, tmp_path):
        """"Play" has to mean play. Opening a search page and calling it done is
        how "I've queued it up, enjoy" ends up on screen with silence."""
        controller, runner = self.controller(tmp_path, [(0, "God's Plan — Drake", "")], "Spotify")
        outcome = controller.open_track("spotify:track:abc123", "Spotify")
        assert outcome.ok and outcome.message == "Playing God's Plan — Drake in Spotify."
        assert 'play track "spotify:track:abc123"' in runner.calls[0][-1]

    def test_spotify_accepting_the_track_without_playing_is_not_success(self, tmp_path):
        controller, _ = self.controller(tmp_path, [(0, "", "")], "Spotify")
        assert controller.open_track("spotify:track:abc", "Spotify").ok is False


class TestAppleScriptQuoting:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("God's Plan", "God's Plan"),          # apostrophes are safe
            ('12" mix', '12\\" mix'),
            ("back\\slash", "back\\\\slash"),
        ],
    )
    def test_a_title_cannot_break_out_of_its_string(self, raw, expected):
        """A song title is interpolated into a script. Same discipline as SQL."""
        assert applescript_string(raw) == expected

    def test_a_quoted_title_survives_into_the_script(self, tmp_path):
        (tmp_path / "Music.app").mkdir()
        runner = fake_runner([(0, "", "")])
        MacOSController(runner=runner, app_dirs=(tmp_path,)).play_local('12" mix', "Music")
        script = runner.calls[0][-1]
        # One balanced string, not two — the quote is escaped, not closing it.
        assert script.count('search library playlist 1 for "') == 1


class TestCatalogLookup:
    """Apple's search endpoint needs no key — that is why it is the one used.
    A "play this song" that only works after registering for an API is one most
    people never switch on."""

    def response(self, payload, status=200):
        import httpx

        class FakeClient:
            async def get(self, url, params=None):
                request = httpx.Request("GET", url)
                return httpx.Response(status, json=payload, request=request)

        return FakeClient()

    def test_the_first_hit_becomes_a_url_and_a_label(self):
        from nero.skills.play_music.server import apple_catalog_url

        client = self.response(
            {"results": [{"trackName": "God's Plan", "artistName": "Drake",
                          "trackViewUrl": "https://music.apple.com/x?i=1"}]}
        )
        assert asyncio.run(apple_catalog_url("gods plan", client)) == (
            "https://music.apple.com/x?i=1",
            "God's Plan — Drake",
        )

    def test_no_results_is_none_not_a_crash(self):
        from nero.skills.play_music.server import apple_catalog_url

        assert asyncio.run(apple_catalog_url("zzz", self.response({"results": []}))) is None

    def test_a_hit_with_no_url_is_unusable(self):
        from nero.skills.play_music.server import apple_catalog_url

        client = self.response({"results": [{"trackName": "x", "artistName": "y"}]})
        assert asyncio.run(apple_catalog_url("x", client)) is None

    def test_a_search_outage_never_reaches_the_user_as_a_traceback(self):
        from nero.skills.play_music.server import apple_catalog_url

        import httpx

        class Broken:
            async def get(self, url, params=None):
                raise httpx.ConnectError("no route to host")

        assert asyncio.run(apple_catalog_url("x", Broken())) is None

    def test_without_credentials_spotify_says_so_instead_of_claiming_success(self):
        """The bug this replaces: the skill reported "I opened that search —
        press play", the model relayed "I've queued it up, enjoy", and nothing
        played. A skill that cannot do the thing must say so."""
        from nero.skills.play_music.server import SPOTIFY_SETUP_HINT

        controller = Stub(["Spotify"])
        result = run_skill(
            PlayMusicSkill(controller=controller, preferred_app="Spotify"),
            action="play",
            track="gods plan drake",
        )
        assert result == SPOTIFY_SETUP_HINT
        assert "queued" not in result.lower()
        assert ("open_track", "spotify:search:gods plan drake", "Spotify") not in controller.calls

    def test_with_credentials_it_resolves_a_real_track_uri(self):
        from nero.skills.play_music.server import spotify_track_uri

        import httpx

        class FakeClient:
            async def post(self, url, data=None, headers=None):
                return httpx.Response(200, json={"access_token": "tok"},
                                      request=httpx.Request("POST", url))

            async def get(self, url, params=None, headers=None):
                return httpx.Response(
                    200,
                    json={"tracks": {"items": [{
                        "uri": "spotify:track:6DCZcSspjsKoFjzjrWoCd",
                        "name": "God's Plan",
                        "artists": [{"name": "Drake"}]}]}},
                    request=httpx.Request("GET", url),
                )

            async def aclose(self):
                pass

        got = asyncio.run(spotify_track_uri("gods plan", ("id", "secret"), FakeClient()))
        assert got == ("spotify:track:6DCZcSspjsKoFjzjrWoCd", "God's Plan — Drake")

    def test_bad_credentials_never_reach_the_user_as_a_traceback(self):
        from nero.skills.play_music.server import spotify_track_uri

        import httpx

        class Rejecting:
            async def post(self, url, data=None, headers=None):
                return httpx.Response(400, json={"error": "invalid_client"},
                                      request=httpx.Request("POST", url))

            async def aclose(self):
                pass

        assert asyncio.run(spotify_track_uri("x", ("bad", "bad"), Rejecting())) is None
