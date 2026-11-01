/** The workspace of a conversation with no turns yet. Autosave keeps a chat only once it has a
 *  turn (an empty one would crowd Your chats), and in server mode the server keeps the tree and the
 *  probes but not the page's own workspace (the steer rack, sampling, the highlight), so a recipe
 *  built before the first message would vanish on reload. The page keeps that workspace here,
 *  keyed by the model and the tree's root, until the first turn's autosave record takes over. */
import type { ConversationSnapshotV8 } from "./conversationSnapshot";

const WORKSPACE_DRAFT_KEY = "drowse.workspace-draft.v1";

interface WorkspaceDraft {
  version: 1;
  modelId: string;
  rootId: string;
  snapshot: ConversationSnapshotV8;
}

type WebStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function localStore(): WebStorage | undefined {
  try {
    return globalThis.localStorage ?? undefined;
  } catch {
    return undefined;
  }
}

/** Keep ``snapshot`` (a tree with no turns) as the draft of its model and root. */
export function saveWorkspaceDraft(snapshot: ConversationSnapshotV8, storage: WebStorage | undefined = localStore()): boolean {
  if (!storage) return false;
  try {
    storage.setItem(WORKSPACE_DRAFT_KEY, JSON.stringify({
      version: 1,
      modelId: snapshot.model_id,
      rootId: snapshot.tree.root_id,
      snapshot,
    } satisfies WorkspaceDraft));
    return true;
  } catch {
    return false;
  }
}

/** The draft of ``modelId``'s tree rooted at ``rootId``, or ``null`` (none, another model or tree,
 *  or unreadable). The caller validates the snapshot as it restores it. */
export function readWorkspaceDraft(
  modelId: string,
  rootId: string,
  storage: WebStorage | undefined = localStore(),
): ConversationSnapshotV8 | null {
  if (!storage) return null;
  try {
    const raw = storage.getItem(WORKSPACE_DRAFT_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<WorkspaceDraft>;
    if (
      value.version !== 1 || value.modelId !== modelId || value.rootId !== rootId
      || !value.snapshot || value.snapshot.model_id !== modelId || value.snapshot.tree?.root_id !== rootId
    ) return null;
    return value.snapshot;
  } catch {
    return null;
  }
}

/** Drop the draft (its conversation has an autosave record now). */
export function clearWorkspaceDraft(storage: WebStorage | undefined = localStore()): void {
  try {
    storage?.removeItem(WORKSPACE_DRAFT_KEY);
  } catch {}
}
