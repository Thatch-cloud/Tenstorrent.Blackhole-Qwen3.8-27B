set +e
echo "--- on $(hostname)"
getent hosts spark-76cb; grep -i -A3 76cb /etc/hosts ~/.ssh/config 2>/dev/null
ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null spark-76cb 'bash -s' <<'INNER'
set +e
echo "--- on $(hostname)"; uptime; echo; free -g | head -2; echo
systemctl --user is-active thatch-node-agent; systemctl --user status thatch-node-agent --no-pager 2>&1 | head -25
echo; echo "--- agent log since 07:50Z (no heartbeats)"
journalctl --user -u thatch-node-agent --since "2026-09-26 07:50:00 UTC" --no-pager 2>&1 | grep -vi heartbeat | tail -80
echo; echo "--- kernel since 07:50Z"
journalctl -k --since "2026-09-26 07:50:00 UTC" --no-pager 2>&1 | grep -iE "oom|killed process|out of memory|nvrm|xid|hung" | tail -20
echo; nvidia-smi 2>&1 | head -25
echo; docker ps -a --format "{{.Names}} {{.Image}} {{.Status}}" 2>&1 | head -20
INNER
