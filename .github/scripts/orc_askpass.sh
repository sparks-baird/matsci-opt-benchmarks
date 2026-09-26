#!/bin/sh
# SSH_ASKPASS helper for orc_connect.py. ssh runs it once per ORC login prompt,
# with the prompt text as $1, and sends whatever it prints. Anything other than
# the password or verification code prompt (a host key question, say) fails.
case "$1" in
*[Vv]erification*) printf '%s\n' "$ORC_CODE" ;;
*[Pp]assword*) cat "$ORC_PASSWORD_FILE" ;;
*) exit 1 ;;
esac
