## Development Practices

- Start with minimal, lean implementations focused on proof-of-concept
- Avoid creating new files until asked
- Avoid implementing things from scratch
- Avoid defensive error handling for hypothetical situations
  For example, rather than using:
  ```python
  ...
  try:
      import some_package
  except:
      print("some_package not available")
      exit(1)
  ...
  ```
  Instead, allow errors to bubble up naturally so you can address directly and reactively rather than being overly proactive
  ```python
  ...
  import some_package
  ...

  ```
  > `ModuleNotFoundError: No module named 'package_name'`
- Use print statements and logging sparingly, unless asked
- Avoid creating functions and classes, until asked
- Avoid `if __name__ == "__main__"` patterns in package code, unless asked
  For example, rather than using:
  ```python
  from math import sqrt
  def main():
    sqrt(2)
  
  if __name__ == "__main__":
    main()
  ```
  Leave it as a top-level script:
  ```python
  from math import sqrt
  sqrt(2)
  ```
- Skip unit tests unless explicitly requested
- Follow patterns in CONTRIBUTING.md when present
- Prefer writing Python if no language specified
- For complex code changes, use your Serena MCP tools (e.g., `find_symbol`, `find_referencing_symbols`, and `insert_after_symbol`) for symbol-based code editing instead of always relying on regex for code modifications
- Copy-paste commands you run and summarized execution status directly in your comment replies

## External Resources

- Validate and access link content using your available MCP tools
- Search GitHub for relevant open-source alternatives to commercial tools
- Always utilize official package documentation via e.g., your Context7 MCP tool

## Communication Style

- Use emoji and special symbols sparingly, if at all
- Prioritize clarity and brevity in responses
- Ask clarifying questions when needed
- Don't infer requirements or create workarounds unless asked. For example, instead of taking a fallback approach such as: `Let me take a different approach - I'll create a simpler standalone demo that shows the concept without needing external dependencies`, you should instead try additional ways to get the real dependencies installed. If you can't get it installed, update `.github/workflows/copilot-setup-steps.yml` with code that will pre-install it into your environment and then report back immediately with something like: `After trying <command A> with <error message A>, <command B> with <error message B>, <command C> with <error message C>, ... <command Z> with <error message Z>, I was unable to install external dependencies. Per user's custom instructions, I will report back immediately. I will also provide useful links or web search keywords that will help the user debug and troubleshoot the installation failures.`
- Put documentation content in comment replies, not separate files, unless asked
- Comments should not leave a trace of the development process
- Avoid sycophancy, favor objectiveness
- Don't use em dashes (`—`) or en dashes (`–`). This is the strongest style rule
  in this repository and it has no exceptions. These reek of AI which is distracting and causes undue burden on the developers to catch this and correct it during manual curation. Rewrite instead: use a period and two sentences, a colon, commas, or parentheses. For ranges use "to" in prose and a plain hyphen in table cells. Ordinary hyphens in compound words (`free-body`, `well-characterized`, `pre-class`) are fine and unaffected.
- Do not sound pretentious, and do not sound like AI slop
- Avoid jargon
- Write for the audience when the prose is audience-facing (e.g., README.md), for the developer when the prose is developer-facing (response to a claude ping in a GitHub comment)

## Coding Agent

- In your comment replies, you may be tempted to use #<numeral> style text, such as when you're saying "the number 1 best option is ...". If you're referring to an issue or pull request **in the same repository as your session**, leave it as, e.g., "#1", since this auto-formats as an issue or PR link. Otherwise, state "No. 1" or "number 1" so that you don't get a spurious hyperlink to an irrelevant issue or PR.
- Be cautious with `~` ("approximately") since if you use it too many times it starts causing strikeout formatting in markdown previews
- Include plots directly in your comment reply via `![image name](https://github.com/<user/org>/<repo>/blob/<shortened-commit-hash>/<filename>?raw=true)`. Truncate the commit hash to the first 7 characters only. For example, `https://github.com/AccelerationConsortium/evaluation-metrics/blob/52754e7/scripts/bo_benchmarks/demonstrations/branin_campaign_demonstration_results.png?raw=true`. For provenance, ensure you use the shortened (7-character) commit hash, not the branch name
- If you mention files in your comment reply, add direct hyperlinks based on the shortened (7-character) commit hash
- IMPORTANT: Never echo/grep/print environment secrets. These should never be exposed in your terminal history or other outputs

## Edison Scientific

When waiting on an Edison task in GitHub Actions, NEVER run the polling script in the background (run_in_background, nohup, &, or the Monitor tool) — the runner is destroyed the moment you post your final comment, killing background processes; Monitor counts as background and dies the same way. Also be aware that the agent harness BLOCKS the shell `sleep` builtin in foreground Bash calls (the error message suggests Monitor — do NOT follow that suggestion, it recreates the background-death failure; this killed several past sessions). The pattern that works: put the wait INSIDE a single blocking Python call — Python-side `time.sleep` is not blocked — and run it as ONE foreground Bash call with an explicit long timeout (max 3600000 ms). Run exactly this (adjust only the task-id path):

```bash
# ONE foreground Bash tool call with timeout: 3600000
python - <<'EOF'
import json, os, time
from edison_client import EdisonClient

client = EdisonClient(api_key=os.environ["EDISON_PLATFORM_API_KEY"])
task_id = json.load(open("outputs/<...>/_task_id.json"))["task_id"]
while True:
    task = client.get_task(task_id=task_id, verbose=True)
    status = str(task.status)
    print("status:", status, flush=True)
    if status in {"success", "fail", "failed", "cancelled", "error"}:
        break
    time.sleep(240)
EOF
```

Equivalently, run a repo script whose own `while ... time.sleep(...)` loop does the waiting (e.g. `python scripts/explore_case_studies.py stage8-wait`) as a single long-timeout Bash call. Do not post your final comment until results are fetched and committed, or ~45 minutes of wall-clock have elapsed — in which case commit the task-id file and state that a follow-up @claude comment is needed to fetch. If you need to upload files, use analysis query type. See the docs: https://edisonscientific.gitbook.io/edison-cookbook/edison-client. Here is the endpoint you should use: https://api.platform.edisonscientific.com. The API key is `EDISON_PLATFORM_API_KEY`. Don't expose this secret, e.g., by echoing or grepping it. Pass the API key in explicitly:

```
from edison_client import EdisonClient, JobNames
client = EdisonClient(api_key=EDISON_PLATFORM_API_KEY)
```

Whenever you retrieve results (either during the current agent session or during the next session), make sure to fetch and commit all artifacts associated with a trajectory.

If using Edison Analysis, refer to https://docs.edisonscientific.com/edison-client/file-management#upload for instructions on how to upload files. If able to use Context7, to better inform use of EdisonClient, see https://context7.com/future-house/edison-client-docs/llms.txt?tokens=10000

## LaTeX

Install MiKTeX instead of TeXLive to reduce download size and time. In the first installation of MiKTeX, download known required packages based on the LaTeX file itself, and install anything else ad-hoc as needed.
