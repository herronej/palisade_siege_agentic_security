# PALISADE G4 — Semgrep ruleset

**Authoritative copies of the rules:** [`backend/src/vista_backend/palisade/contracts/semgrep/`](../../backend/src/vista_backend/palisade/contracts/semgrep/)
**Install:** `pip install vista-backend[palisade-g4]` (Semgrep is an
opt-in extra so default installs don't pay its transitive-dependency
cost.)

The PALISADE-specific ruleset is layered on top of the community
`p/security-audit` configuration. The default `semgrep_config`
points at `p/security-audit`; PALISADE's rules are loaded
alongside by passing the package-bundled directory as an additional
`--config` argument:

```
semgrep --config=p/security-audit \
        --config=backend/src/vista_backend/palisade/contracts/semgrep/
```

The G4 gate's runtime invokes Semgrep this way when
`semgrep_enabled=True`; the rules below are the bundled defaults.
Operators can drop replacement YAML files at
`<contracts_dir>/semgrep/` to override or extend without touching
the bundled set.

## Threat-surface mapping

Each rule is tagged with `metadata.vista-threat-surface` (the
category in our attack taxonomy) and `metadata.vista-taxonomy` (the
section number for cross-reference). The table below summarizes.

| Rule ID                            | File                       | Severity | Threat surface         | Taxonomy |
| ---------------------------------- | -------------------------- | -------- | ---------------------- | -------- |
| `vista-subprocess-shell-true`      | `subprocess.yml`           | ERROR    | code-execution         | §4.1     |
| `vista-os-system`                  | `subprocess.yml`           | ERROR    | code-execution         | §4.1     |
| `vista-os-popen`                   | `subprocess.yml`           | ERROR    | code-execution         | §4.1     |
| `vista-eval-exec`                  | `code_execution.yml`       | ERROR    | code-execution         | §4.2     |
| `vista-pickle-untrusted`           | `code_execution.yml`       | ERROR    | code-execution         | §4.3     |
| `vista-credential-file-read`       | `credentials.yml`          | ERROR    | credential-exfiltration | §4.4     |
| `vista-credential-env-var-read`    | `credentials.yml`          | WARNING  | credential-exfiltration | §5.7     |
| `vista-env-var-preload-hijack`     | `env_var_manipulation.yml` | ERROR    | sandbox-escape         | §4.5     |
| `vista-env-var-pythonpath-pollution` | `env_var_manipulation.yml` | WARNING  | sandbox-escape         | §4.5     |
| `vista-env-var-path-pollution`     | `env_var_manipulation.yml` | WARNING  | sandbox-escape         | §4.5     |
| `vista-typo-squat-import`          | `typo_squat.yml`           | ERROR    | supply-chain           | §4.6     |
| `vista-direct-outbound-network`    | `network.yml`              | WARNING  | data-exfiltration      | §5.5     |
| `vista-globus-untrusted-endpoint`  | `network.yml`              | ERROR    | data-exfiltration      | §5.7     |

## Rule-by-rule rationale

### `vista-subprocess-shell-true` — taxonomy §4.1

**What it catches.** `subprocess.run(..., shell=True, ...)` and the
sibling functions (`Popen`, `call`, `check_call`, `check_output`).

**Why.** With `shell=True`, the first positional argument is
interpreted by `/bin/sh`. Any embedded shell metacharacter (`;`,
`|`, `&&`, backticks, `$(...)`) in user- or RAG-derived input
breaks out of the intended command and runs arbitrary code. This
is the classic OS-command-injection vulnerability (CWE-78). VISTA's
sandbox MCP tool already pre-validates argument lists; the
`shell=True` path bypasses that vetting and is therefore the right
place to detect adversarial code emission.

**Fix.** Pass an argument list and leave `shell=False`:
`subprocess.run(["ls", "-la", path], shell=False)`.

### `vista-os-system` — taxonomy §4.1

**What it catches.** `os.system(...)` calls.

**Why.** `os.system` always invokes `/bin/sh`. There is no way to
disable shell interpretation. Same threat as `shell=True` above.

**Fix.** Use `subprocess.run([cmd, *args])`.

### `vista-os-popen` — taxonomy §4.1

**What it catches.** `os.popen(...)` calls.

**Why.** `os.popen` pipes its stdout from a shell-interpreted
command. Same shell-injection surface as `os.system`.

**Fix.** `subprocess.run([...], capture_output=True)`.

### `vista-eval-exec` — taxonomy §4.2

**What it catches.** `eval`, `exec`, and `compile` calls.

**Why.** Dynamic-code execution is the canonical sandbox-escape
primitive (CWE-95). Even when the input *looks* safe, an LLM
emitting these inside the sandbox is a strong signal of attempted
escape. The live evaluation found that attackers reliably emit
`eval`/`exec` after refusing the explicit "ignore previous
instructions" jailbreak — the indirection works around the G1
prompt regex.

**Fix.** Use `ast.literal_eval` for parsing data literals; refuse
arbitrary-string-to-code conversions.

### `vista-pickle-untrusted` — taxonomy §4.3

**What it catches.** `pickle.loads`, `pickle.load`, and the
deprecated `cPickle` + `dill` equivalents.

**Why.** Pickle deserialization invokes `__reduce__` on every
deserialized object. A maliciously-crafted pickle payload executes
arbitrary Python code on `loads`. This is so well-known that the
`pickle` module's own documentation warns against untrusted input
(CWE-502). An LLM choosing pickle for round-tripping data is
almost always a smell — JSON / msgpack with explicit schemas covers
the same ground without the RCE surface.

**Fix.** JSON for data, msgpack with strict schema for binary
formats, never pickle.

### `vista-credential-file-read` — taxonomy §4.4

**What it catches.** `open("...")` against ~14 well-known
credential-file paths: SSH keys, AWS/Azure/GCP CLI credentials,
`.netrc`, `/etc/passwd`, `/etc/shadow`, `~/.kube/config`, etc.

**Why.** The sandbox MCP server does not legitimately need to read
any of these files. An LLM emitting code that opens one is
attempting credential exfiltration (CWE-200). The list covers the
high-value targets; operators with additional credential locations
should extend by dropping a replacement file at
`<contracts_dir>/semgrep/credentials.yml`.

**Fix.** Refuse the operation. There is no legitimate reason the
sandbox needs `/etc/shadow`.

### `vista-credential-env-var-read` — taxonomy §5.7

**What it catches.** Reads of sensitive credential environment
variables — `AWS_SECRET_ACCESS_KEY`, `AZURE_CLIENT_SECRET`,
`GITHUB_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`NERSC_IRI_TOKEN`, `S3M_TOKEN`, etc.

**Why.** A read alone is not always malicious — some legitimate
tools need the value to authenticate. But paired with a
network-write to a non-allow-listed sink (`vista-direct-outbound-network`)
the result is credential exfiltration. SEVERITY: WARNING (rather
than ERROR) because the read alone is ambiguous; the slow-tier
Q-LLM intent extraction is the layer that promotes to ERROR.

**Fix.** Pass credentials to MCP tools via the existing
`vista_user_config` metadata path, not by re-reading env vars
inside the sandbox.

### `vista-env-var-preload-hijack` — taxonomy §4.5

**What it catches.** Writes to `LD_PRELOAD`, `LD_LIBRARY_PATH`,
`DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH` via
`os.environ[...] = ...` or `os.putenv`.

**Why.** These environment variables determine which shared
libraries the linker loads ahead of system libraries (CWE-426). An
attacker that writes them ahead of a subprocess spawn gets
arbitrary code execution inside the spawned process under the
identity of the agent. This is a well-documented privilege-escalation
primitive on both Linux (`LD_PRELOAD`) and macOS (`DYLD_*`).

**Fix.** Refuse. The sandbox does not need these env vars.

### `vista-env-var-pythonpath-pollution` — taxonomy §4.5

**What it catches.** Writes to `PYTHONPATH`.

**Why.** `PYTHONPATH` redirects `import` to attacker-controlled
modules. Same CWE-426 surface as the preload-hijack rule but
narrower scope (Python-only).

**Fix.** If the agent needs to add a path, do it in-process with
`sys.path.insert(0, path)` (and review the path).

### `vista-env-var-path-pollution` — taxonomy §4.5

**What it catches.** Writes to `PATH`.

**Why.** `PATH` redirects subsequent `subprocess` spawns to
attacker-controlled binaries.

**Fix.** Pass absolute paths to `subprocess.run`.

### `vista-typo-squat-import` — taxonomy §4.6

**What it catches.** Imports of known typo-squatted PyPI packages
(`urllib4`, `requests3`, `python_dateutils`, `colourama`,
`jeIlyfish`, `pytorch` (the typosquat), `tensorfllow`, etc.).

**Why.** These packages have shipped credential-exfiltration or
RCE payloads, per PyPI's own security advisories and the public
Snyk reports. The legitimate package name is almost certainly the
one without the typo (`urllib3`, `requests`, `python-dateutil`,
`colorama`, `jellyfish`, `torch`, `tensorflow`).

The list is **not exhaustive**. New typo-squats are added to the
PyPI advisory feed weekly; operators should extend the rule by
dropping a replacement `typo_squat.yml` in `<contracts_dir>/semgrep/`.
A future contract-library mechanism will let the
operator-supplied list merge with the bundled defaults rather than
replace them.

**Fix.** Use the correct package name.

### `vista-direct-outbound-network` — taxonomy §5.5

**What it catches.** Direct HTTP/HTTPS calls from inside the
sandbox via `urllib`, `requests`, `httpx`, `aiohttp`, or raw
sockets.

**Why.** Most legitimate VISTA tool calls reach the network
through the MCP server's controlled tool surface (which G2 can
allow-list). Code that bypasses that surface is either (a) a
legitimate but un-policy-vetted external call that *should* be
routed through MCP tools, or (b) an exfiltration attempt. SEVERITY:
WARNING — the slow-tier Q-LLM is the layer that promotes to ERROR
based on intent.

**Fix.** Add the external API as an MCP tool. The MCP server is
the trust boundary; routing through it gets you G2's allow-list
and the ETDI descriptor pinning.

### `vista-globus-untrusted-endpoint` — taxonomy §5.7

**What it catches.** `globus_sdk.TransferClient`, `TransferData`,
`submit_transfer`, `endpoint_activate`, `NativeAppAuthClient` calls.

**Why.** Globus is the standard scientific-data transfer fabric;
it moves files between facility-trusted identities. An attacker
that initiates a transfer from the deployment's identity to an
attacker-controlled endpoint moves data out under the legitimate
user's name — a particularly nasty exfiltration vector because
the audit trail looks normal.

A future operator-supplied allow-list of
endpoint UUIDs at `<contracts_dir>/globus_endpoints.json` is
planned. Until that lands, this rule fires on *any* Globus client construction
and the slow-tier Q-LLM is the next check ("is this a benign data
move?").

**Fix.** Route Globus transfers through an MCP tool that consults
the allow-list, not direct `globus_sdk` calls from the sandbox.

## Calibration and false-positive policy

The rules above default-deny on the high-confidence patterns
(everything at SEVERITY: ERROR). The WARNING rules — credential env
var reads, direct outbound network — exist as triggers for the G4
slow-tier Q-LLM. The live evaluation measures the per-rule
false-positive rate on a 200-prompt benign workload; the per-rule
calibration table will live in `docs/palisade/g1_g4_eval.md`
once the evaluation runs.

## Extending the ruleset

Two extension points:

1. **Operator override** (no code change). Drop a replacement YAML
   in `<contracts_dir>/semgrep/`. The runtime reads from
   `<contracts_dir>/semgrep/*.yml` and the package-bundled
   directory; the operator file shadows the bundled one of the
   same name.

2. **Bundled rule** (PR to VISTA). Add a new `.yml` to
   `backend/src/vista_backend/palisade/contracts/semgrep/` and
   update this document with the threat-surface mapping. Include
   a benign-pattern test fixture in
   `backend/src/vista_backend/palisade/tests/fixtures/semgrep_benign/`
   when the rule has plausible false-positive surface — the
   live evaluation harness picks those up automatically.

The bundled set is intentionally small. Rules at the boundary of
"useful" and "annoying" should default to the operator-extension
path until live measurement shows they earn their place.
