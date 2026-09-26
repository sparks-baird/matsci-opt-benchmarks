"""Open an SSH connection to BYU's supercomputer (ORC) for a @claude-orc session.

ORC takes no SSH keys: every login asks for the account password and a 6-digit
verification code from the operator's authenticator app. This posts one comment
asking the operator for a code, then checks the thread every 2 seconds for a
comment from the operator, new or edited since that request, that holds just a
code. It logs in as soon as one appears. If ORC rejects the code, the request
comment says so and the operator edits their comment with a fresh code.

The login leaves an OpenSSH master connection (Host orc in ~/.ssh/config) that
later steps reuse with plain `ssh orc ...` until the job ends. The password is
handed to ssh by orc_askpass.sh from a file that is deleted once the connection
is up, so it is not in the connection's environment or on disk afterwards.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

repo = os.environ["GITHUB_REPOSITORY"]
issue = os.environ["ISSUE_NUMBER"]
operator = os.environ["OPERATOR"]
here = Path(__file__).resolve().parent
wait_minutes = 15

ssh_dir = Path.home() / ".ssh"
ssh_dir.mkdir(mode=0o700, exist_ok=True)
(ssh_dir / "config").write_text(
    f"""Host orc
    HostName ssh.rc.byu.edu
    User {os.environ["BYU_HPC_USERNAME"]}
    ControlMaster auto
    ControlPath ~/.ssh/orc.sock
    ControlPersist yes
    ServerAliveInterval 60
    NumberOfPasswordPrompts 1
    StrictHostKeyChecking yes
    UserKnownHostsFile {here / "orc_known_hosts"}
"""
)
secret_dir = Path(tempfile.mkdtemp())  # mode 0700
password_file = secret_dir / "password"
password_file.write_text(os.environ.pop("BYU_HPC_PASSWORD"))
log_file = secret_dir / "ssh.log"


def gh(*args):
    out = subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def say(text):
    gh("-X", "PATCH", f"repos/{repo}/issues/comments/{request['id']}", "-f", f"body={text}")


def login(code):
    env = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "SSH_ASKPASS": str(here / "orc_askpass.sh"),
        "SSH_ASKPASS_REQUIRE": "force",
        "ORC_CODE": code,
        "ORC_PASSWORD_FILE": str(password_file),
    }
    # -f sends ssh to the background once it has logged in; it keeps the log
    # file open, so its output goes there rather than to a pipe
    with log_file.open("w") as log:
        subprocess.run(
            ["ssh", "-fN", "-o", "ControlMaster=yes", "orc"],
            env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=120,
        )
    ok = subprocess.run(["ssh", "-O", "check", "orc"], capture_output=True).returncode == 0
    if not ok:
        print("".join(log_file.read_text().splitlines(keepends=True)[-20:]))
    return ok


request = gh(
    "-X", "POST", f"repos/{repo}/issues/{issue}/comments", "-f",
    f"body=ORC login is waiting for a verification code. @{operator}, reply with just the "
    "current 6-digit code from your authenticator app, or edit an earlier code comment of "
    f"yours to hold the new code. Waiting {wait_minutes} minutes.",
)
print(f"asked for a code in {request['html_url']}")
tried = set()
deadline = time.time() + wait_minutes * 60
try:
    while time.time() < deadline:
        comments = gh(
            f"repos/{repo}/issues/{issue}/comments?since={request['created_at']}&per_page=100"
        )
        codes = [
            (c["updated_at"], c["id"], m.group(1))
            for c in comments
            if c["user"]["login"] == operator
            and (m := re.fullmatch(r"\s*(?:code:?\s*)?(\d{6})\s*", c["body"], re.I))
        ]
        new = sorted(k for k in codes if k not in tried)
        if new:
            tried.add(new[-1])
            print(f"trying the code from comment {new[-1][1]} ({new[-1][0]})")
            if login(new[-1][2]):
                now = datetime.now(timezone.utc).strftime("%H:%M UTC")
                say(f"Connected to ORC at {now}. The connection stays open until this job ends.")
                print("connected")
                break
            say(
                f"ORC did not accept that code (they last about 30 seconds). @{operator}, "
                "edit your comment with a fresh code. Waiting until "
                f"{datetime.fromtimestamp(deadline, timezone.utc).strftime('%H:%M UTC')}."
            )
        time.sleep(2)
    else:
        say(f"No accepted code within {wait_minutes} minutes. Ping @claude-orc again to retry.")
        raise SystemExit("no accepted verification code")
finally:
    shutil.rmtree(secret_dir)
