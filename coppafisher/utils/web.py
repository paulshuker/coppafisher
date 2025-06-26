import socket
import ssl
import urllib


def try_read_url_at(url: str) -> bytes | None:
    """
    Tries to read the URL's content.

    Args:
        url (str): the URL.

    Returns:
        (bytes or none): result. The resulting contents. None if it fails for any reason.
    """
    try:
        f = urllib.request.urlopen(url)
        return f.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None


def internet_is_active() -> bool:
    """
    Check for an internet connection.

    Returns:
        bool: whether the system is connected to the internet.
    """
    try:
        urllib.request.urlopen("http://www.google.com")
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
