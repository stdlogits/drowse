import type { ManifoldJob } from "../stores/manifoldJobs.svelte";

/** A creation job's band, beyond its title: the line under it (a state, never an instruction, and
 *  empty when the title already says it) and the tip its library button carries for what to do
 *  next. */
export interface JobNote {
  note: string;
  tip: string;
}

const RUNNING: ManifoldJob["status"][] = ["running", "waiting", "cancelling"];

export function jobNote(job: Pick<ManifoldJob, "status" | "phase" | "fitAfterwards">, keepOpen: boolean): JobNote {
  if (RUNNING.includes(job.status)) {
    const step = job.phase === "generating" && job.fitAfterwards ? "Step 1 of 2" : job.fitAfterwards ? "Step 2 of 2" : "";
    return { note: [step, keepOpen ? "Keep this tab open" : ""].filter(Boolean).join(" · "), tip: "" };
  }
  if (job.status !== "complete") return { note: "Saved concept groups are kept", tip: "Reopen the creator or library to try again" };
  switch (job.phase) {
    case "generating": return { note: "", tip: "Fit them from the library when ready" };
    case "scoring": return { note: "", tip: "The template lab shows each answer's sum and mean probability" };
    case "installing": return { note: "", tip: "Check its model fits before steering or probing" };
    // drowse.ai's creation e2e matches this sentence exactly.
    default: return { note: "Saved and ready to steer or probe.", tip: "" };
  }
}
