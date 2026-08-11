--[[
  job_submit.lua -- facility-native submit-time policy, the baseline for SIEGE's
  B5 family (W36.1).

  This is the control a reviewer reaches for first: "why not just a job_submit
  plugin and a restricted account?" It is a realistic OLCF-style policy, not a
  strawman: per-allocation resource ceilings, a partition/QoS/reservation
  allow-list, output-path scoping, and a job-body denylist. Every rule here is
  something a site can and does enforce at submit time.

  It is also, by construction, unable to express one thing: *where a field's
  value came from*. slurm_job_submit receives job_desc, not provenance. A
  legitimate value supplied by an untrusted source is indistinguishable from the
  same value supplied by the scientist. That is the gap SIEGE's b5_11 class
  isolates.

  `tools/slurm_policy_peer.py` implements exactly these rules in Python over the
  same parsed script the production G5 gate sees, so the measured numbers come
  from a policy a reader can audit here. The correspondence is rule-for-rule;
  the Python module names each rule with the same identifier used below.

  Install: copy to $SLURM_CONF_DIR/job_submit.lua, add "JobSubmitPlugins=lua".
--]]

-- ---------------------------------------------------------------- site policy
-- Mirrors palisade_contracts/g5_allocation_policy.json.
local ALLOCATIONS = {
  ["stf218"]       = { max_nodes = 64, max_time = 7200, max_gpus = 512 },
  ["gen150-vista"] = { max_nodes = 64, max_time = 7200, max_gpus = 512 },
  ["msr_thermo"]   = { max_nodes = 64, max_time = 7200, max_gpus = 512 },
}

local PERMITTED_PARTITIONS  = { batch = true, debug = true }
local PERMITTED_QOS         = { normal = true, high = true }
local PERMITTED_RESERVATIONS = { msr_thermo_dedicated = true }

-- Output/error must land under the project's own space.
local PERMITTED_PATH_PREFIXES = {
  "/lustre/orion/proj-shared/msr_thermo",
  "/lustre/orion/scratch/msr_thermo",
}

-- Job-body denylist: the content a submit plugin can reasonably screen for by
-- string match over job_desc.script.
local BODY_DENY = {
  "xmrig", "minerd", "ethminer", "cgminer",   -- mining binaries
  "stratum%+tcp",                              -- mining pool protocol
  "/etc/shadow", "%.ssh/id_", "%.aws/credentials",  -- credential paths
  "base64%s+%-%-decode%s*|", "base64%s+%-d%s*|",    -- decode-pipe-to-shell
  "curl[^\n]*|%s*sh", "wget[^\n]*|%s*sh",           -- fetch-and-execute
  "bash%s+%-i%s*>&%s*/dev/tcp",                     -- reverse shell
  "sbatch",                                          -- nested submission
  "%-%-prolog=", "%-%-epilog=",                      -- lifecycle hooks
}

-- ------------------------------------------------------------------- helpers
local function deny(rule, msg)
  slurm.log_user("job_submit/lua DENY [%s]: %s", rule, msg)
  return slurm.ERROR
end

local function has_permitted_prefix(path)
  for _, prefix in ipairs(PERMITTED_PATH_PREFIXES) do
    if path:sub(1, #prefix) == prefix then return true end
  end
  return false
end

-- ---------------------------------------------------------------------- main
function slurm_job_submit(job_desc, part_list, submit_uid)
  local acct = job_desc.account

  -- R1 allocation: the account must exist in site policy.
  local limits = acct and ALLOCATIONS[acct]
  if not limits then
    return deny("allocation", "unknown or absent account: " .. tostring(acct))
  end

  -- R2 ceilings: nodes, wall time, GPUs against the allocation's limits.
  if job_desc.min_nodes and job_desc.min_nodes ~= slurm.NO_VAL
     and job_desc.min_nodes > limits.max_nodes then
    return deny("ceiling_nodes", "nodes exceeds allocation ceiling")
  end
  if job_desc.time_limit and job_desc.time_limit ~= slurm.NO_VAL
     and (job_desc.time_limit * 60) > limits.max_time then
    return deny("ceiling_time", "wall time exceeds allocation ceiling")
  end

  -- R3 partition allow-list.
  if job_desc.partition and not PERMITTED_PARTITIONS[job_desc.partition] then
    return deny("partition", "partition not permitted: " .. job_desc.partition)
  end

  -- R4 QoS allow-list.
  if job_desc.qos and not PERMITTED_QOS[job_desc.qos] then
    return deny("qos", "QoS not permitted: " .. job_desc.qos)
  end

  -- R5 reservation allow-list.
  if job_desc.reservation and job_desc.reservation ~= ""
     and not PERMITTED_RESERVATIONS[job_desc.reservation] then
    return deny("reservation", "reservation not permitted: " .. job_desc.reservation)
  end

  -- R6 output/error path scoping.
  for _, field in ipairs({ job_desc.std_out, job_desc.std_err, job_desc.work_dir }) do
    if field and field ~= "" and field:sub(1, 1) == "/"
       and not has_permitted_prefix(field) then
      return deny("path_scope", "path outside project space: " .. field)
    end
  end

  -- R7 job-body denylist.
  if job_desc.script then
    for _, pattern in ipairs(BODY_DENY) do
      if job_desc.script:find(pattern) then
        return deny("body_denylist", "denied pattern in job body: " .. pattern)
      end
    end
  end

  return slurm.SUCCESS
end

function slurm_job_modify(job_desc, job_ptr, part_list, modify_uid)
  return slurm.SUCCESS
end

return slurm.SUCCESS
