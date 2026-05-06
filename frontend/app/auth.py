from __future__ import annotations

import hmac
import time
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse

from .config import FrontendConfig, get_config


SESSION_KEY = "mlsystem_user"
SESSION_TS_KEY = "mlsystem_login_ts"


def verify_credentials(username: str, password: str, config: FrontendConfig | None = None) -> bool:
    config = config or get_config()
    return hmac.compare_digest(username, config.username) and hmac.compare_digest(password, config.password)


def login_session(request: Request, username: str) -> None:
    request.session[SESSION_KEY] = username
    request.session[SESSION_TS_KEY] = int(time.time())


def logout_session(request: Request) -> None:
    request.session.clear()


def current_user(request: Request, config: FrontendConfig | None = None) -> str | None:
    config = config or get_config()
    user = request.session.get(SESSION_KEY)
    login_ts = request.session.get(SESSION_TS_KEY)
    if not user or not isinstance(login_ts, int):
        return None
    if int(time.time()) - login_ts > config.session_ttl_seconds:
        request.session.clear()
        return None
    return str(user)


def require_user(request: Request) -> str:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def redirect_if_unauthorized(request: Request) -> RedirectResponse | None:
    if current_user(request):
        return None
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)


def is_authenticated(request: Request) -> bool:
    return current_user(request) is not None

