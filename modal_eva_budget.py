"""CPU-only budget watchdog for the already-submitted EVA call."""
from pathlib import Path
import json
import os
import time

import modal

app = modal.App("coralscapes-eva-budget-guard")
image = modal.Image.debian_slim(python_version="3.12")
volume = modal.Volume.from_name("coral-training")
leases = modal.Dict.from_name("coralscapes-eva-budget-guards", create_if_missing=True)


def stop_decision(now, started, hourly, budget, previous_epoch_seconds=None):
    """Reserve $0.50 and allow for the next epoch plus checkpoint upload."""
    cutoff = started + (budget - .50) / hourly * 3600
    remaining = cutoff - now
    if remaining <= 120:
        return "hard_budget_deadline", cutoff
    if previous_epoch_seconds is not None:
        if remaining < previous_epoch_seconds * 1.4 + 180:
            return "insufficient_budget_for_another_epoch", cutoff
    return None, cutoff


@app.function(image=image, cpu=(.125, .125), memory=(256, 256), timeout=20000,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/workspace": volume})
def guard(call_id: str, group: str, started: float, budget: float, hourly: float):
    owner = modal.current_function_call_id()
    if not leases.put(call_id, owner, skip_if_exists=True) and leases.get(call_id) != owner:
        raise RuntimeError("A budget guard already owns this training call")
    out = Path("/workspace/checkpoints") / group / "eva"
    out.mkdir(parents=True, exist_ok=True)

    def save(name, data):
        target = out / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(target)
        volume.commit()

    def read(name):
        path = out / name
        return json.loads(path.read_text()) if path.exists() else None

    _, cutoff = stop_decision(time.time(), started, hourly, budget)
    control = {"budget_usd_total": budget, "conservative_hourly_usd": hourly,
               "reserved_usd": .50, "started_unix": started, "stop_no_later_than_unix": cutoff - 120,
               "training_call_id": call_id, "guard_call_id": owner,
               "policy": "Keep the matched recipe; cancel at an epoch checkpoint when another epoch may exceed budget. Never restart training."}
    save("budget_control.json", control)
    print("BUDGET_GUARD_ARMED " + json.dumps(control), flush=True)
    last_epoch = None
    while True:
        volume.reload()
        if read("completed.json") is not None:
            save("budget_guard_result.json", {"status": "already_completed", **control})
            return "completed"
        progress = read("progress.json")
        reason, _ = stop_decision(time.time(), started, hourly, budget)
        if progress is not None:
            epoch = progress["epoch_completed"]
            if epoch != last_epoch:
                last_epoch = epoch
                reason, _ = stop_decision(time.time(), started, hourly, budget,
                                           progress["last_epoch_seconds"])
                print(f"BUDGET_CHECK epoch={epoch} estimated_usd={(time.time()-started)*hourly/3600:.3f} reason={reason}", flush=True)
        if reason:
            request = {"reason": reason, "checkpoint_at_request": progress,
                       "estimated_usd_at_request": (time.time() - started) * hourly / 3600,
                       "requested_unix": time.time(), **control}
            save("budget_stop_requested.json", request)
            # Cancel only the authorized training call; completed checkpoints
            # are already committed by the trainer. Its keeper then exits too.
            modal.FunctionCall.from_id(call_id).cancel()
            time.sleep(5)
            volume.reload()
            result = {"status": "stopped_for_user_budget", "progress": read("progress.json"),
                      "full_training_state_preserved": (out / "last.pt").exists(), **request}
            save("budget_stopped.json", result)
            print("BUDGET_STOPPED " + json.dumps(result), flush=True)
            return result
        # Terminal failures do not get restarted and need no ongoing watchdog.
        failure = read("failure.json")
        if failure is not None:
            save("budget_guard_result.json", {"status": "training_failed", "failure": failure, **control})
            return "failed"
        time.sleep(20)


@app.local_entrypoint()
def main(budget: float = 9.80):
    if os.environ.get("MODAL_PROFILE") != "m12mm-4m2005":
        raise ValueError("Use the same Modal account as the EVA run")
    folder = Path(__file__).resolve().parent / "analysis/paper_v2_sources/m12mm-4m2005"
    path = folder / "paper-v2-eva40-s42-20261102-submission.json"
    state = json.loads(path.read_text())
    record = folder / "paper-v2-eva40-s42-20261102-budget-guard.json"
    if record.exists():
        print(record.read_text())
        return
    # Includes L40S, four CPU cores, 24 GiB, the passive keeper and this guard,
    # rounded upward from current public prices. Clock starts at submission.
    hourly = 2.40
    started = state["deadline_utc_timestamp"] - 86400
    call = guard.spawn(state["train_call_id"], state["group"], started, budget, hourly)
    _, cutoff = stop_decision(time.time(), started, hourly, budget)
    result = {"profile": "m12mm-4m2005", "guard_call_id": call.object_id,
              "training_call_id": state["train_call_id"], "budget_usd_total": budget,
              "conservative_hourly_usd": hourly, "reserved_usd": .50,
              "stop_no_later_than_unix": cutoff - 120, "dashboard": app.get_dashboard_url()}
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
