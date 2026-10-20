"""Default HTTP transport DNS cancellation must not delay a search response."""

import asyncio
import socket
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from test_index import shared


class SearchDeadlineTests(unittest.TestCase):
    def test_slow_dns_does_not_join_the_default_executor_after_timeout(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocked_dns(*args, **kwargs):
            entered.set()
            try:
                release.wait(2)
                return []
            finally:
                finished.set()

        async def request():
            # Unlike MockTransport, the real default transport resolves DNS in
            # the event loop's executor. The resolver is stubbed before I/O.
            async with httpx.AsyncClient(trust_env=False) as client:
                await client.get("http://menu-search.invalid")

        started = time.monotonic()
        try:
            with patch.object(socket, "getaddrinfo", side_effect=blocked_dns):
                with self.assertRaises(asyncio.TimeoutError):
                    shared.run_search_with_deadline(request(), 0.1)
            self.assertTrue(entered.is_set(), "Default transport never reached DNS")
            self.assertLess(time.monotonic() - started, 1)
            self.assertFalse(finished.is_set(), "Search waited for DNS shutdown")
        finally:
            release.set()
            self.assertTrue(finished.wait(1), "Test resolver thread did not clean up")