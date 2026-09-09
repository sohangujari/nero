import asyncio
import subprocess

import pytest

from nero.skills.open_website.server import OpenWebsiteSkill, resolve


@pytest.fixture
def skill():
    return OpenWebsiteSkill()


def run(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


@pytest.fixture
def opened(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "nero.skills.open_website.server.webbrowser.open",
        lambda url: calls.append(url) or True,
    )
    return calls


class TestResolve:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("youtube", "https://www.youtube.com"),
            ("YouTube", "https://www.youtube.com"),
            ("  github  ", "https://github.com"),
            ("open youtube", "https://www.youtube.com"),
            ("go to reddit", "https://www.reddit.com"),
        ],
    )
    def test_known_site_names(self, query, expected):
        assert resolve(query) == expected

    def test_full_url_passes_through(self):
        assert resolve("https://example.com/page") == "https://example.com/page"

    def test_http_url_passes_through(self):
        assert resolve("http://example.com") == "http://example.com"

    def test_bare_domain_gets_https(self):
        assert resolve("example.com") == "https://example.com"

    def test_domain_with_path_gets_https(self):
        assert resolve("example.com/docs") == "https://example.com/docs"

    def test_case_sensitive_path_is_preserved(self):
        # youtu.be video IDs are case-sensitive; lowercasing breaks the link.
        assert resolve("youtu.be/dQw4w9WgXcQ") == "https://youtu.be/dQw4w9WgXcQ"

    def test_mixed_case_domain_and_path_preserved(self):
        assert resolve("GitHub.com/Foo/Bar") == "https://GitHub.com/Foo/Bar"

    def test_known_site_lookup_is_still_case_insensitive(self):
        # The SITES table lookup itself stays case-insensitive; only the
        # pass-through domain path preserves original case.
        assert resolve("YouTube") == "https://www.youtube.com"

    @pytest.mark.parametrize("query", ["", "   ", "that site with the thing", "my bank"])
    def test_ambiguous_returns_none(self, query):
        assert resolve(query) is None


class TestMeta:
    def test_metadata(self, skill):
        assert skill.meta.name == "open_website"
        assert skill.meta.requires_network is True
        assert skill.meta.permission_tier == "state_changing"
        assert skill.meta.input_schema["required"] == ["site"]


class TestExecute:
    def test_opens_known_site(self, skill, opened):
        result = run(skill, site="youtube")
        assert opened == ["https://www.youtube.com"]
        assert "youtube.com" in result

    def test_opens_url(self, skill, opened):
        run(skill, site="https://example.com")
        assert opened == ["https://example.com"]

    def test_missing_site_returns_error(self, skill, opened):
        assert "Error" in run(skill)
        assert opened == []

    def test_ambiguous_asks_for_clarification(self, skill, opened):
        result = run(skill, site="that site with the thing")
        assert opened == []
        assert "not sure" in result.lower()
        assert "ask the user" in result.lower()

    def test_browser_failure_is_reported(self, skill, monkeypatch):
        monkeypatch.setattr(
            "nero.skills.open_website.server.webbrowser.open", lambda url: False
        )
        assert "couldn't open" in run(skill, site="youtube").lower()

    def test_browser_exception_is_caught(self, skill, monkeypatch):
        def boom(url):
            raise OSError("no display")

        monkeypatch.setattr("nero.skills.open_website.server.webbrowser.open", boom)
        result = run(skill, site="youtube")
        assert "Error" in result and "no display" in result


class TestBrowserChoice:
    """`webbrowser.open` always uses the OS default, so "open YouTube in Chrome"
    opened Safari on a Mac. The browser has to be chosen explicitly."""

    def macos(self, monkeypatch, returncode=0):
        seen = []

        def fake(cmd, **kwargs):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode, "", "")

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("nero.skills.open_website.server.subprocess.run", fake)
        monkeypatch.setattr(
            "webbrowser.open", lambda url: pytest.fail("must not fall back to the default")
        )
        return seen

    @pytest.mark.parametrize(
        "spoken,expected",
        [
            ("chrome", "Google Chrome"),
            ("Chrome", "Google Chrome"),
            ("google chrome", "Google Chrome"),
            ("firefox", "Firefox"),
            ("edge", "Microsoft Edge"),
            ("brave", "Brave Browser"),
            ("Brave Browser", "Brave Browser"),
        ],
    )
    def test_a_named_browser_is_the_one_used(self, monkeypatch, spoken, expected):
        seen = self.macos(monkeypatch)
        result = run(OpenWebsiteSkill(), site="youtube", browser=spoken)
        assert seen[0] == ["open", "-a", expected, "https://www.youtube.com"]
        assert expected in result

    def test_no_browser_named_keeps_the_old_default_behaviour(self, monkeypatch):
        """The existing path must not change for anyone who never names one."""
        opened = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
        monkeypatch.setattr(
            "nero.skills.open_website.server.subprocess.run",
            lambda *a, **k: pytest.fail("must not launch a specific browser"),
        )
        assert run(OpenWebsiteSkill(), site="youtube") == "Opened https://www.youtube.com."
        assert opened == ["https://www.youtube.com"]

    def test_a_remembered_browser_is_used_without_being_asked_for(self, monkeypatch):
        seen = self.macos(monkeypatch)
        run(OpenWebsiteSkill(preferred_browser="Chrome"), site="youtube")
        assert seen[0][2] == "Google Chrome"

    def test_a_named_browser_is_remembered(self, monkeypatch):
        self.macos(monkeypatch)
        saved = []
        run(OpenWebsiteSkill(on_browser_chosen=saved.append), site="youtube", browser="firefox")
        assert saved == ["firefox"]

    def test_a_browser_used_only_by_preference_is_not_re_remembered(self, monkeypatch):
        self.macos(monkeypatch)
        saved = []
        run(
            OpenWebsiteSkill(preferred_browser="Chrome", on_browser_chosen=saved.append),
            site="youtube",
        )
        assert saved == []

    def test_a_browser_that_failed_to_open_is_not_remembered(self, monkeypatch):
        self.macos(monkeypatch, returncode=1)
        saved = []
        result = run(OpenWebsiteSkill(on_browser_chosen=saved.append), site="youtube",
                     browser="brave")
        assert saved == []
        assert "Brave Browser" in result

    def test_asking_for_a_browser_they_lack_says_so_rather_than_using_another(
        self, monkeypatch
    ):
        """Silently opening Safari when they asked for Chrome is the bug, not
        the fallback."""
        self.macos(monkeypatch, returncode=1)
        result = run(OpenWebsiteSkill(), site="youtube", browser="chrome")
        assert "couldn't open Google Chrome" in result

    def test_a_stale_preference_falls_back_instead_of_stranding_the_request(self, monkeypatch):
        """If the remembered browser was uninstalled, the site should still open."""
        opened = []
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr(
            "nero.skills.open_website.server.subprocess.run",
            lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", ""),
        )
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
        result = run(OpenWebsiteSkill(preferred_browser="Vivaldi"), site="youtube")
        assert opened == ["https://www.youtube.com"]
        assert result == "Opened https://www.youtube.com."

    def test_an_unknown_browser_name_lists_the_ones_it_knows(self, monkeypatch):
        monkeypatch.setattr("webbrowser.open", lambda url: True)
        result = run(OpenWebsiteSkill(), site="youtube", browser="netscape")
        assert "netscape" in result and "Firefox" in result

    def test_a_full_url_still_works_with_a_browser(self, monkeypatch):
        seen = self.macos(monkeypatch)
        run(OpenWebsiteSkill(), site="https://example.com/x", browser="firefox")
        assert seen[0] == ["open", "-a", "Firefox", "https://example.com/x"]

    def test_linux_launches_the_executable_with_the_url(self, monkeypatch):
        started = []
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr(
            "nero.skills.open_website.server.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        monkeypatch.setattr(
            "nero.skills.open_website.server.subprocess.Popen",
            lambda cmd, **k: started.append(cmd),
        )
        run(OpenWebsiteSkill(), site="youtube", browser="chrome")
        assert started[0] == ["/usr/bin/google-chrome", "https://www.youtube.com"]
