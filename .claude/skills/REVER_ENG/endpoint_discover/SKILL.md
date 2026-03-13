<!--
HARD DEPENDENCIES (required MCP tools / services):
  - mcp__firecrawl__firecrawl_browser_create
  - mcp__firecrawl__firecrawl_browser_execute
  - mcp__firecrawl__firecrawl_browser_delete

INJECT DEPENDENCIES (conceptual — used when available):
  - Playwright (Tier 2/3 cookie extraction and in-browser endpoint calls)
  - curl / fetch (Tier 1 direct endpoint testing)
-->

# Skill: endpoint-discover

TRIGGER: User invokes `/endpoint-discover` or asks to reverse engineer a website's internal API endpoints.
DESCRIPTION: Uses FireCrawl MCP browser tools to navigate a target website, observe network requests/responses, and reverse engineer internal endpoints for programmatic use. Discovers only real endpoints evidenced by actual UI interactions — never fabricates endpoints.

## Instructions

### Phase 1: Browser Setup & Navigation

1. Accept the target URL from the user.
2. Create a FireCrawl browser session via `mcp__firecrawl__firecrawl_browser_create`.
3. Navigate to the target URL via `mcp__firecrawl__firecrawl_browser_execute`.

### Phase 2: Endpoint Discovery

1. **Interact with the UI to discover endpoints.** Only document endpoints that correspond to observable UI actions (buttons, dropdowns, tabs, pagination, search bars, sort controls, filters). Do NOT guess or fabricate endpoints.

2. **For each UI interaction**, use `mcp__firecrawl__firecrawl_browser_execute` to:
   a. Open the browser DevTools Network tab / intercept network traffic before clicking.
   b. Perform the UI action (click sort button, apply filter, submit search, etc.).
   c. Capture the resulting request: method, URL, headers (especially cookies/tokens), query params, and request body.
   d. Capture the response: status, content-type, and a representative sample of the response body.

3. **Filtering before sorting.** When the site offers scope/time-range filters (e.g., "All Time" vs "Recent", "Whole period" vs "Last month"), always apply the broadest filter FIRST, then discover sort endpoints within that filtered state. This ensures endpoints return complete datasets, not partial recent-only results.

4. **Record only essential parameters.** Strip noise — keep only the parameters that change the response (sort field, sort direction, page number, filter values, query string). Drop analytics/tracking params.

### Phase 3: Cookie & Auth Analysis

For each discovered endpoint, determine the cookie/auth binding:

1. **Test cookie portability:**
   a. Extract cookies from the FireCrawl MCP browser session.
   b. Attempt a direct call (e.g., via `curl` or `fetch`) to the endpoint using only the extracted cookie, from a different context.
   c. If it succeeds → **Tier 1: Direct call with MCP cookie.** Record the cookie name(s) and value(s) needed.

2. **If Tier 1 fails (cookie may be IP-bound):**
   a. Use Playwright (if available in the project) to open the same site, authenticate if needed, and extract cookies from that browser context.
   b. Attempt the direct call with the Playwright-extracted cookie.
   c. If it succeeds → **Tier 2: Direct call with Playwright cookie.** Note that cookie must be refreshed via Playwright.

3. **If Tier 2 fails (cookie is browser+IP bound):**
   a. → **Tier 3: Call via Playwright.** The endpoint must be invoked inside a Playwright browser context using `page.evaluate()` or `page.route()` interception.

4. **NEVER extract cookies from the host machine's browser or cookie store.**

### Phase 4: Output

1. For each discovered endpoint, produce a concise spec:

```
## Endpoint: <descriptive name>
- Tier: <1|2|3>
- Method: <GET|POST|...>
- URL: <full URL with path>
- Required Headers: <only auth-relevant headers>
- Required Params:
  - <param>: <description> (e.g., sort=citations, order=desc, filter=all_time)
- Example Request: <minimal curl/fetch snippet>
- Example Response: <trimmed JSON sample, key fields only>
```

2. If the user has existing frontend code calling these endpoints incorrectly, identify and fix the issues:
   - Wrong parameter names or values
   - Missing filters (e.g., not setting time range to "all time")
   - Incorrect sort field/direction
   - Missing required headers or cookies

3. Clean up: delete the FireCrawl browser session via `mcp__firecrawl__firecrawl_browser_delete`.

## Rules

- **No fabricated endpoints.** Every endpoint must be observed from a real network request triggered by a UI interaction.
- **No public/documented APIs.** Use only the internal endpoints the website's own frontend calls.
- **Minimal params.** Include only parameters required to reproduce the desired response. Drop optional/cosmetic/tracking params.
- **Broadest filter first.** Always prefer "all time" / "whole period" scope before applying sort, to avoid missing data.
- **Cookie hierarchy.** Always attempt Tier 1 first, fall back to Tier 2, then Tier 3. Document which tier each endpoint requires.
