# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""Drive a notebook in the real JupyterLab UI of a built xeus-python site, with Playwright."""

import os

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

TIMEOUT = int(os.getenv("LMLITE_MODEL_TIMEOUT_MS", "300000"))
console = []  # the page's console messages, filled by the test, shown when the cells do not finish

# True once every code cell has run: none is still `[*]:` and none is waiting without a number.
ALL_CELLS_DONE = """() => {
  const prompts = [...document.querySelectorAll('.jp-CodeCell .jp-InputPrompt')].map(e => e.textContent.trim());
  return prompts.length > 0 && prompts.every(p => /^\\[\\d+\\]:$/.test(p));
}"""

OUTPUTS = """() => ({
  images: [...document.querySelectorAll('.jp-OutputArea img')].map(i => ({src: i.src.slice(0, 24), width: i.naturalWidth, height: i.naturalHeight})),
  audios: [...document.querySelectorAll('.jp-OutputArea audio')].map(a => ({src: a.src.slice(0, 24), duration: a.duration, error: a.error?.message ?? null})),
  errors: [...document.querySelectorAll('.jp-OutputArea [data-mime-type="application/vnd.jupyter.error"]')].map(e => e.textContent.slice(0, 400)),
  text: [...document.querySelectorAll('.jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stdout"], .jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"]')].map(e => e.textContent),
})"""


def open_notebook(page, site, name):
    """Open `name` in Lab and wait until its kernel is idle."""
    page.goto(f"{site}/lab/index.html?path={name}")
    page.wait_for_selector(".jp-Notebook")
    page.wait_for_function(
        "document.querySelector('.jp-Notebook-ExecutionIndicator')?.getAttribute('data-status') === 'idle'",
        timeout=TIMEOUT,
    )


def run_all(page):
    """Run > Run All Cells."""
    page.get_by_role("menuitem", name="Run", exact=True).click()
    page.get_by_role("menuitem", name="Run All Cells").first.click()


def wait_for_all_cells(page, timeout=None):
    try:
        page.wait_for_function(ALL_CELLS_DONE, timeout=timeout or TIMEOUT)
    except PlaywrightTimeoutError:
        state = page.evaluate(
            """() => ({
              kernel: document.querySelector('.jp-Notebook-ExecutionIndicator')?.getAttribute('data-status'),
              prompts: [...document.querySelectorAll('.jp-CodeCell .jp-InputPrompt')].map(e => e.textContent.trim()),
            })"""
        )
        pytest.fail(
            f"Cells did not finish. {state}\nShown so far: {outputs(page)}\nConsole:\n" + "\n".join(console[-25:])
        )


def outputs(page):
    """What the cells show: images, audio players, error outputs and printed text."""
    return page.evaluate(OUTPUTS)


def run_in_repl(site, code, steps, flags=(), permissions=(), timeout=300000):
    """Run `code` in the xeus-python REPL page, call `steps(page)` for the taps it needs, and return the RESULT JSON."""
    import json
    import urllib.parse

    from playwright.sync_api import sync_playwright

    url = f"{site}/repl/index.html?kernel=xpython&execute=1&code={urllib.parse.quote(code)}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=list(flags))
        try:
            context = browser.new_context()
            context.grant_permissions(list(permissions))
            page = context.new_page()
            page.set_default_timeout(timeout)
            messages = []
            page.on("console", lambda message: messages.append(f"{message.type}: {message.text[:300]}"))
            page.on("pageerror", lambda error: messages.append(f"pageerror: {error}"))
            page.goto(url)
            try:
                steps(page)
                page.wait_for_function("document.body.innerText.includes('RESULT{')", timeout=120000)
            except Exception as error:
                pytest.fail(
                    f"{error}\nPage:\n{page.inner_text('body')[-2500:]}\nConsole:\n" + "\n".join(messages[-15:])
                )
            text = page.inner_text("body")
            extra = page.evaluate("() => [...document.querySelectorAll('video')].map(v => v.srcObject === null)")
        finally:
            browser.close()
    line = next(line for line in text.splitlines() if line.startswith("RESULT{"))
    return json.loads(line.removeprefix("RESULT")), extra


def tap(page, prompt, button):
    """Click `button` (1 for the first, 2 for the second) next to the prompt text of a widget's pending request."""
    page.get_by_text(prompt, exact=True).locator(f"xpath=following-sibling::button[{button}]").click()
