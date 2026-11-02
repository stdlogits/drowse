// Opening a saved chat for the loaded model: save the current work, restore the chat's workspace,
// and make it the active saved chat. The one path for the saved-chats dialog and the thread's
// "Earlier" list.
import { restoreConversationSnapshot } from "./conversationWorkspace";
import { conversationLibrary, flushConversationAutosave, savedConversationState } from "./stores/savedConversations.svelte";

export async function openSavedChat(id: string): Promise<{ name: string }> {
  await flushConversationAutosave();
  const record = await conversationLibrary.get(id);
  await restoreConversationSnapshot(record.snapshot);
  savedConversationState.activeId = record.id;
  savedConversationState.avatarSeed = record.avatarSeed;
  savedConversationState.accent = record.accent ?? "purple";
  return { name: record.name };
}
