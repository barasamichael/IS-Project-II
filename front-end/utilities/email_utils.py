import time
import requests
from threading import Thread
from flask import current_app
from flask import render_template

# Simple in-memory token cache; for multi-process deployments use Redis.
_access_token_cache: dict = {"token": None, "expires_at": 0}


def _cfg(key: str, default=None):
    """Read a value from the Flask app config."""
    return current_app.config.get(key, default)


def _get_access_token() -> str:
    """
    Return a valid Zoho OAuth access token, refreshing via the stored
    refresh token when the cached one is within 30 seconds of expiry.
    """
    now = int(time.time())
    token = _access_token_cache.get("token")
    expires_at = int(_access_token_cache.get("expires_at") or 0)

    if token and now < (expires_at - 30):
        return token

    accounts_base = _cfg(
        "ZOHO_ACCOUNTS_BASE", "https://accounts.zoho.com"
    ).rstrip("/")
    client_id = _cfg("ZOHO_CLIENT_ID")
    client_secret = _cfg("ZOHO_CLIENT_SECRET")
    refresh_token = _cfg("ZOHO_REFRESH_TOKEN")

    missing = [
        k
        for k, v in {
            "ZOHO_CLIENT_ID": client_id,
            "ZOHO_CLIENT_SECRET": client_secret,
            "ZOHO_REFRESH_TOKEN": refresh_token,
        }.items()
        if not v
    ]
    if missing:
        raise RuntimeError(f"Missing Zoho config keys: {', '.join(missing)}")

    resp = requests.post(
        f"{accounts_base}/oauth/v2/token",
        data={
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        },
        timeout=_cfg("ZOHO_HTTP_TIMEOUT", 20),
    )
    resp.raise_for_status()
    data = resp.json()

    token = data["access_token"]
    expires_in = int(data.get("expires_in", 3600))
    _access_token_cache["token"] = token
    _access_token_cache["expires_at"] = now + expires_in
    return token


def _join(addr):
    """Return a comma-separated address string as expected by the Zoho API."""
    if not addr:
        return None
    if isinstance(addr, str):
        return addr
    return ",".join(addr)


def _zoho_send_message(payload: dict) -> dict:
    """POST a message payload to the Zoho Mail API."""
    mail_base = _cfg("ZOHO_MAIL_BASE", "https://mail.zoho.com").rstrip("/")
    account_id = _cfg("ZOHO_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("Missing Zoho config key: ZOHO_ACCOUNT_ID")

    access_token = _get_access_token()
    url = f"{mail_base}/api/accounts/{account_id}/messages"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Zoho-oauthtoken {access_token}",
    }

    r = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=_cfg("ZOHO_HTTP_TIMEOUT", 20),
    )
    r.raise_for_status()
    return r.json()


def _send_async(app, payload: dict) -> None:
    with app.app_context():
        _zoho_send_message(payload)


def send_email(to, subject, template, cc=None, bcc=None, **kwargs):
    """
    Asynchronously send an HTML email via the Zoho Mail API.

    :param to: Recipient address(es) — string or list of strings.
    :param subject: Email subject line.
    :param template: Template base name (without extension). The file
        ``<template>.html`` is rendered from the Flask template folder.
    :param cc: Carbon-copy address(es) — optional.
    :param bcc: Blind-carbon-copy address(es) — optional.
    :param kwargs: Extra variables forwarded to the template renderer.
    :return: The daemon Thread handling the send.
    """
    app = current_app._get_current_object()
    rendered_html = render_template(f"{template}.html", **kwargs)

    from_address = _cfg("ZOHO_FROM_ADDRESS")
    if not from_address:
        raise RuntimeError("Missing Zoho config key: ZOHO_FROM_ADDRESS")

    payload = {
        "fromAddress": from_address,
        "toAddress": _join(to),
        "ccAddress": _join(cc),
        "bccAddress": _join(bcc),
        "subject": subject,
        "content": rendered_html,
    }

    thread = Thread(target=_send_async, args=(app, payload), daemon=True)
    thread.start()
    return thread
