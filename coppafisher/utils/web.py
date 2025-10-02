import socket
import ssl
import urllib

TIMEOUT: float = 3.0


def try_read_url_at(url: str) -> bytes | None:
    """
    Tries to read the URL's content.

    Args:
        url (str): the URL.

    Returns:
        (bytes or none): result. The resulting contents. None if it fails for any reason.
    """
    try:
        f = urllib.request.urlopen(url, timeout=TIMEOUT)
        return f.read()
    except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError):
        return None


def internet_is_active() -> bool:
    """
    Check for an internet connection.

    Returns:
        bool: whether the system is connected to the internet.
    """
    try:
        urllib.request.urlopen("http://www.google.com", timeout=TIMEOUT)
        return True
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        ValueError,
        socket.gaierror,
        TimeoutError,
        OSError,
        ssl.SSLError,
        ConnectionResetError,
        FileNotFoundError,
    ):
        return False
