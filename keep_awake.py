# keep_awake.py
import sys
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

URL = "https://pro-breast.streamlit.app"


# A substring is used to stay robust to whitespace/casing quirks in the DOM.
RENDER_MARKER = "Predict 30-day complications"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle", timeout=60000)

        # If the app is asleep, click the wake button.
        try:
            page.get_by_text("get this app back up", exact=False).click(timeout=5000)
            print("App was asleep — clicked wake button, waiting for restart.")
        except PlaywrightTimeout:
            print("No wake button found — app was already awake.")

        # The app iframe doesn't exist while sleeping; wait for it to appear
        # (covers both the already-awake and post-wake-restart cases).
        page.locator("iframe").first.wait_for(timeout=120000)

        # App runs inside an iframe on Streamlit Community Cloud.
        app_frame = page.frame_locator("iframe").first
        app_container = app_frame.locator('[data-testid="stAppViewContainer"]')
        app_container.wait_for(timeout=120000)

        # Assert the app actually rendered. Fails the job if not found.
        try:
            app_container.get_by_text(RENDER_MARKER, exact=False).wait_for(
                timeout=30000
            )
            print(f"Render confirmed: found '{RENDER_MARKER}'.")
        except PlaywrightTimeout:
            print(
                f"ERROR: '{RENDER_MARKER}' never appeared. "
                "App may be broken, mid-deploy, or showing an error screen.",
                file=sys.stderr,
            )
            page.screenshot(path="failure.png")
            browser.close()
            sys.exit(1)

        # Hold the session open so the WebSocket registers as active.
        page.wait_for_timeout(15000)
        browser.close()
        print("Done — session held open, app kept awake.")


if __name__ == "__main__":
    main()
