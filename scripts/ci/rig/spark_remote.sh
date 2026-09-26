set +e
echo "--- on $(hostname)"
ip -4 -o addr show 2>/dev/null | awk '{print $2, $4}'
grep -iE "76cb|spark" /etc/hosts ~/.ssh/config 2>/dev/null
cat > /tmp/inner76.sh <<'INNER'
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
found=""
cands="192.168.2.34 192.168.2.69 192.168.2.70 192.168.2.72 192.168.2.145 192.168.2.173 192.168.2.192 192.168.2.197"
for i in $(seq 1 254); do cands="$cands 10.10.11.$i"; done
for ip in $cands; do
  timeout 1 bash -c "</dev/tcp/$ip/22" 2>/dev/null || continue
  name=$(ssh -o BatchMode=yes -o ConnectTimeout=4 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$ip" hostname 2>/dev/null)
  echo "ssh $ip -> ${name:-<no key access>}"
  case "$name" in *76cb*) found="$ip";; esac
  [ -n "$found" ] && break
done
if [ -n "$found" ]; then
  echo "=== spark-76cb is $found"
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$found" 'bash -s' < /tmp/inner76.sh
else
  echo "=== spark-76cb not reachable by key from $(hostname)"
fi
rm -f /tmp/inner76.sh
