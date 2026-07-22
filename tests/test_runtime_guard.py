from nero._runtime import REQUIRED, check_python_version


def test_warns_on_wrong_minor():
    msg = check_python_version((3, 13, 0))
    assert msg and "3.12" in msg


def test_warns_on_old_minor():
    assert check_python_version((3, 11, 9)) is not None


def test_no_warning_on_supported():
    assert check_python_version((3, 12, 5)) is None


def test_required_is_312():
    assert REQUIRED == (3, 12)
