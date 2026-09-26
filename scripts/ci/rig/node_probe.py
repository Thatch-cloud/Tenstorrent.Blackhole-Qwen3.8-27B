"""Why has a fleet node stopped heart-beating? A read-only probe from the rig host.

Runs on the rig (thatch-control-plane-prod), on the same LAN as the Sparks. Prints: the down scrape
targets and every series naming the site in the control plane's Prometheus, the k3s node list, the
site's address, a ping, and - only if the rig already holds a key the host accepts (BatchMode, no
password, no known_hosts write) - uptime, the node agent's state and log tail, nvidia-smi and disk.
Nothing is started, stopped or written on the target.
Usage: node_probe.py [site]   (default spark-76cb)
"""
import json
import re
import subprocess
import sys
import urllib.parse

NL = chr(10)
SITE = sys.argv[1] if len(sys.argv) > 1 else 'spark-76cb'
SHORT = SITE.replace('spark-', '')


def run(command, timeout=60):
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout + out.stderr
    except Exception as error:
        return None, repr(error)


def section(title, command, timeout=60):
    print('=== ' + title, flush=True)
    code, text = run(command, timeout)
    print(text[-6000:].rstrip(), flush=True)
    print('(exit %s)' % code, flush=True)
    return text


def prom(query):
    url = 'http://localhost:9090/api/v1/query?' + urllib.parse.urlencode({'query': query})
    code, text = run(['sudo', '-n', 'k3s', 'kubectl', '-n', 'thatch-monitoring', 'exec', 'deploy/thatch-prometheus',
                      '--', 'wget', '-qO-', url], 120)
    if code:
        return 'error: ' + text[-300:]
    try:
        body = json.loads(text)
    except ValueError:
        return 'unparseable: ' + text[:300]
    rows = []
    for row in body.get('data', {}).get('result', []):
        metric = dict(row.get('metric', {}))
        value = row.get('value', [None, None])[1]
        rows.append('%s %s' % (json.dumps(metric, sort_keys=True)[:400], value))
    return NL.join(rows) or '(no series)'


QUERIES = [
    ('down scrape targets', 'up == 0'),
    ('targets naming the site', 'up{instance=~".*%s.*"} or up{node=~".*%s.*"} or up{site=~".*%s.*"} or up{hostname=~".*%s.*"}' % ((SHORT,) * 4)),
    ('metric families labelled with the site', 'count by (__name__) ({site="%s"})' % SITE),
    ('metric families with the site in instance', 'count by (__name__) ({instance=~".*%s.*"})' % SHORT),
    ('seconds since each target last scraped up', 'time() - timestamp(up)'),
    ('uptime of every node exporter (s)', 'time() - node_boot_time_seconds'),
    ('up over the last hour, min', 'min_over_time(up[1h]) == 0'),
]
for title, query in QUERIES:
    print('=== prometheus: %s: %s' % (title, query), flush=True)
    print(prom(query), flush=True)

nodes = section('k3s nodes', ['sudo', '-n', 'k3s', 'kubectl', 'get', 'nodes', '-o', 'wide'])
section('pods on or about the site', ['bash', '-c', "sudo -n k3s kubectl get pods -A -o wide 2>&1 | grep -i '%s' | head -40" % SHORT])
hosts = section('name lookup', ['bash', '-c', 'getent hosts %s %s.local %s.lan 2>&1; grep -i %s /etc/hosts 2>&1' % (SITE, SITE, SITE, SHORT)])
section('neighbours', ['bash', '-c', 'ip neigh 2>&1 | head -40'])

address = None
for text in (hosts, nodes):
    for line in text.splitlines():
        if SHORT in line:
            match = re.search(r'\b(\d{1,3}(?:\.\d{1,3}){3})\b', line)
            if match:
                address = match.group(1)
                break
    if address:
        break
print('=== address: %s' % address, flush=True)
if address:
    section('ping', ['ping', '-c', '4', '-W', '2', address], 30)
    section('tcp ports 22, 30051 and the node agent defaults', ['bash', '-c',
            'for p in 22 30051 8080 8000 9100; do timeout 3 bash -c "</dev/tcp/%s/$p" 2>/dev/null && echo "$p open" || echo "$p closed"; done' % address], 40)
    remote = ('uptime; echo; systemctl --user is-active thatch-node-agent; systemctl --user status thatch-node-agent --no-pager 2>&1 | head -25; '
              'echo; journalctl --user -u thatch-node-agent --since -2h --no-pager 2>&1 | grep -vi heartbeat | tail -60; '
              'echo; nvidia-smi 2>&1 | head -25; echo; df -h / 2>&1 | tail -1; free -g 2>&1 | head -2; '
              'echo; docker ps -a --format "{{.Names}} {{.Image}} {{.Status}}" 2>&1 | head -20')
    section('on the host (key auth only, read-only)', ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6',
            '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', 'thatch@' + address, remote], 90)
