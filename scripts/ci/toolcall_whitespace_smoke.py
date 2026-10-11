"""The smoke rule of the tool-call whitespace fix (c2_toolcall_whitespace): what a serving container log must show.

    problems(env, container_text) -> [str]   (an empty list is a clean log)

The integrator wires it in as a smoke rule (scripts/ci/fusion-wp/TW.json: "smoke_rules"); c2_smoke_check.fusion_problems
calls it once per arm with the profile's env mapping and the container log. The fix is armed for every profile in the
vLLM API server, so the rule is about the log, not a flag:

  1. A log of a booted API server (the contract's "profile NAME: vLLM argv" line) holds the "armed" line. No line, no
     fix: the arm served the stripping parser.
  2. An armed log holds the "installed on" line (or "not needed", a vLLM whose converter already keeps the edges): vLLM
     imported its qwen3 parser and the converter was swapped (graft mounted is not graft executed). An install the
     module refused kills the boot, so it shows as a dead container.
  3. A "disabled" line is a problem unless the arm's env sets QWEN_C2_TOOLCALL_WS=0 (a deliberate kill-switch arm).
"""

import c2_toolcall_whitespace as fix

BOOTED = ': vLLM argv '
INSTALLED = 'toolcall whitespace: installed on '


def problems(env, container_text):
    env = env or {}
    lines = container_text.splitlines()
    if not any('[QWEN-C2] profile ' in line and BOOTED in line for line in lines):
        return []      # not the API server's boot log (a status or reset step): nothing to judge
    armed = any(fix.ARMED in line for line in lines)
    disabled = any(fix.DISABLED in line for line in lines)
    found = []
    if disabled and env.get(fix.ENV_SWITCH) != '0':
        found.append('the log says "%s" and the arm does not set %s=0: the arm served the stripping tool-call parser'
                     % (fix.DISABLED, fix.ENV_SWITCH))
    if not armed and not disabled:
        found.append('no "%s" line in the container log: the API server booted without the tool-call whitespace fix '
                     '(its argument values lose their edge whitespace)' % fix.ARMED)
    if armed and not any(INSTALLED in line or fix.NOT_NEEDED in line for line in lines):
        found.append('"%s" is logged but no "%s" line: vLLM never imported its qwen3 parser, or the converter swap did '
                     'not run' % (fix.ARMED, INSTALLED.strip()))
    return found
