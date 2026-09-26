# Parse the model selector tag for claude.yml (both jobs). Reads TRIGGER_TEXT,
# PRIMARY_MODEL, FALLBACK_MODEL and DEFAULT_EFFORT; writes phrase, model,
# effort, fallback_model and primary_args to GITHUB_OUTPUT.
# Tags: @claude or @claude-orc, then optional +<model> and :<effort>, e.g.
# @claude+opus-5:max or @claude-orc:high.

# First tag in the text wins
TAG=$(grep -oiE '@claude(-orc)?(\+[a-z0-9.-]*[a-z0-9])?(:(low|medium|high|xhigh|max))?' <<<"$TRIGGER_TEXT" | head -n1)
TAG=${TAG:-@claude}

tag_lc=$(tr '[:upper:]' '[:lower:]' <<<"$TAG")
rest=${tag_lc#@claude}
rest=${rest#-orc}
effort=""
if [[ "$rest" == *:* ]]; then
  effort=${rest##*:}
  rest=${rest%:*}
fi
model_part=${rest#+}

# "opus-4.8" becomes claude-opus-4-8; a bare "opus" stays "opus", which
# the CLI reads as an alias for the latest model of that family.
resolve_model() {
  local p=${1,,}
  if [[ -z "$p" ]]; then
    printf ''
  elif [[ "$p" != *[0-9]* ]]; then
    printf '%s' "${p#claude-}"
  else
    local m=${p//./-}
    [[ "$m" == claude-* ]] || m="claude-$m"
    printf '%s' "$m"
  fi
}

model=$(resolve_model "$model_part")
model=${model:-$PRIMARY_MODEL}
fallback=$(resolve_model "$FALLBACK_MODEL")

case "$effort" in
  low|medium|high|xhigh|max) ;;
  "") effort=$DEFAULT_EFFORT ;;
  *) echo "Ignoring unknown effort level ':$effort'; using $DEFAULT_EFFORT"
     effort=$DEFAULT_EFFORT ;;
esac

# "opus" and "claude-opus-5" name the same model, so compare with the
# "claude-" prefix stripped and treat a bare family name as matching
# any version of it.
a=${model#claude-}
b=${fallback#claude-}
if [[ -z "$fallback" ]]; then
  echo "Fallback is off (FALLBACK_MODEL is empty)."
elif [[ "$a" == "$b" || "$a" == "$b"-* || "$b" == "$a"-* ]]; then
  echo "Fallback $fallback is the same model as the primary ($model), so no retry will be attempted."
  fallback=""
fi

# --fallback-model is the CLI's own in-session switch. It covers an
# overloaded or unavailable model, not an exhausted allowance, and it
# keeps the session going instead of restarting it, so it is worth
# having in addition to the job-level retry.
primary_args="--model $model --effort $effort"
if [[ -n "$fallback" ]]; then
  primary_args="$primary_args --fallback-model $fallback"
fi

echo "Tag: $TAG -> $primary_args"
if [[ -n "$fallback" ]]; then
  echo "If the allowance for $model is spent, the job retries with $fallback."
fi

echo "phrase=$TAG" >> "$GITHUB_OUTPUT"
echo "model=$model" >> "$GITHUB_OUTPUT"
echo "effort=$effort" >> "$GITHUB_OUTPUT"
echo "fallback_model=$fallback" >> "$GITHUB_OUTPUT"
echo "primary_args=$primary_args" >> "$GITHUB_OUTPUT"
