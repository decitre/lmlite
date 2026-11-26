# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""``request()`` for the camera and microphone widgets: ask the user for a photo or a recording and wait for it.

The widget shows the prompt and needs a tap before it captures. What it sends back carries the id of the request, so
only media captured for this request resolves it. A cancel, a timeout or a denied permission clear the prompt and
release the camera or microphone.
"""

import asyncio
import itertools

import traitlets


class CaptureCancelled(Exception):
    """The user dismissed the request with Cancel. A tool can try another route or answer without the media."""


class _Requests(traitlets.HasTraits):
    # The request the widget shows, {id, prompt, ...options}, or {} when there is none. A trait and not a message, so a
    # widget that is displayed after the request is made (or while the page is still connecting to the kernel) shows it.
    pending_request = traitlets.Dict().tag(sync=True)
    # The browser sets these, after the media: the id of the request a capture answers, the id the user cancelled,
    # and {id, name, message} when the browser refused the camera or microphone.
    answered = traitlets.Int(0).tag(sync=True)
    cancelled = traitlets.Int(0).tag(sync=True)
    error = traitlets.Dict().tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ids = itertools.count(1)
        self._pending = None  # (id, future) of the request in progress
        self.observe(self._on_answer, names="answered")
        self.observe(self._on_cancel, names="cancelled")
        self.observe(self._on_error, names="error")

    def _settle(self, request_id, outcome):
        """Resolve the pending request with ``outcome()`` (a result or an exception) if ``request_id`` is its id."""
        if self._pending and self._pending[0] == request_id:
            future = self._pending[1]
            self._pending = None
            if not future.done():
                try:
                    future.set_result(outcome())
                except Exception as exc:
                    future.set_exception(exc)

    def _on_answer(self, change):
        self._settle(change["new"], self._media)

    def _on_cancel(self, change):
        def cancelled():
            raise CaptureCancelled("The user cancelled the request.")

        self._settle(change["new"], cancelled)

    def _on_error(self, change):
        error = change["new"]
        if not error:
            return

        def refused():
            message = error.get("message") or error.get("name") or "The browser refused the device."
            if error.get("name") in ("NotAllowedError", "SecurityError"):
                raise PermissionError(message)
            raise RuntimeError(message)

        self._settle(error.get("id"), refused)

    async def _request(self, prompt, timeout, **options):
        if self._pending:
            raise RuntimeError("A request is already pending: await it, or cancel its task, first.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must say what the user is asked to do.")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive, or None to wait for ever.")
        request_id = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending = (request_id, future)
        self.error = {}
        self.pending_request = {"id": request_id, "prompt": prompt, **options}
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            if self._pending and self._pending[0] == request_id:
                self._pending = None
            # Settled, timed out or cancelled: the widget clears the prompt and releases a device it started.
            self.pending_request = {}
