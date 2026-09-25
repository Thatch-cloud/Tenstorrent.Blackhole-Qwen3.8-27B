"""What real Qwen traffic looks like, from the control plane's Prometheus (read-only).

Runs on the rig host. Lists every thatch_serving_* family, then for the last N days reports:
histograms (prompt/generation tokens, latencies) as bucket increases by model; counters as
increases; gauges (running/waiting requests) as max and a coarse distribution.
Usage: serving_metering.py [days]
"""
import json
import subprocess
import sys
import urllib.parse

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
BASE = 'http://localhost:9090/api/v1/'


def prom(path, **params):
    url = BASE + path + ('?' + urllib.parse.urlencode(params) if params else '')
    out = subprocess.run(['sudo', '-n', 'k3s', 'kubectl', '-n', 'thatch-monitoring', 'exec', 'deploy/thatch-prometheus',
                          '--', 'wget', '-qO-', url], capture_output=True, text=True, timeout=120)
    if out.returncode:
        raise RuntimeError(out.stderr[-400:])
    body = json.loads(out.stdout)
    if body.get('status') != 'success':
        raise RuntimeError(str(body)[:400])
    return body['data']


names = [n for n in prom('label/__name__/values') if n.startswith('thatch_serving_') or n.startswith('vllm')]
print('families (%d):' % len(names))
for name in names:
    print('  ' + name)
window = '%dd' % DAYS
for name in names:
    try:
        if name.endswith('_bucket'):
            data = prom('query', query='sum by (le, model, node, site) (increase(%s[%s]))' % (name, window))
            rows = sorted(((r['metric'].get('model', '?'), r['metric'].get('site', r['metric'].get('node', '?')),
                            r['metric'].get('le')), float(r['value'][1])) for r in data['result'])
            if rows:
                print('\n%s, increase over %s (cumulative buckets):' % (name, window))
                for (model, site, le), value in rows:
                    if value:
                        print('  %-40s %-12s le=%-10s %.0f' % (model[:40], site, le, value))
        elif name.endswith('_total') or name.endswith('_count') or name.endswith('_sum'):
            data = prom('query', query='sum by (model, node, site) (increase(%s[%s]))' % (name, window))
            for r in data['result']:
                if float(r['value'][1]):
                    print('%s %s %.0f' % (name, json.dumps(r['metric'], sort_keys=True), float(r['value'][1])))
        else:
            data = prom('query', query='max by (model, node, site) (max_over_time(%s[%s]))' % (name, window))
            for r in data['result']:
                print('%s max %s %s' % (name, json.dumps(r['metric'], sort_keys=True), r['value'][1]))
    except Exception as error:
        print('%s: %s' % (name, str(error)[:200]))
