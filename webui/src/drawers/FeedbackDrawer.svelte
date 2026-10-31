<script lang="ts">
  import ContactForm from "../lib/ui/ContactForm.svelte";
  import { CONTACT_ADDRESS } from "../lib/contact";
  import { IconButton } from "@stdlogits/ledger";
  import { closeDrawer } from "../lib/stores/drawers.svelte";
  import { runtimeClient } from "../lib/runtime/client";
  let { params: _params }: { params: unknown } = $props();
</script>

<section class="feedback-drawer" aria-label="Submit Feedback">
  <header class="dlg-head"><h2>Share feedback</h2><IconButton icon="close" label="Close feedback" onclick={closeDrawer} /></header>
  <div class="body">
    <ContactForm source="feedback" ondone={closeDrawer} />
    <p class="contact-link">Need help? <a class="prose-link" href={runtimeClient.mode === "http" ? "https://drowse.ai/contact" : "/contact"} target="_blank" rel="noreferrer">Contact us<span class="sr-only"> (opens a new tab)</span></a> or email <a class="prose-link" href={`mailto:${CONTACT_ADDRESS}`}>{CONTACT_ADDRESS}</a>.</p>
  </div>
</section>

<style>
  .feedback-drawer { display: flex; flex-direction: column; height: 100%; min-height: 0; }
  .dlg-head { display: flex; flex: none; align-items: center; justify-content: space-between; gap: var(--s4); padding: var(--s5) var(--s5) 0; }
  .dlg-head h2 { margin: 0; color: var(--ink); font: var(--type-mark); }
  .body { display: grid; gap: var(--s6); align-content: start; padding: var(--s5); overflow-y: auto; overscroll-behavior: contain; }
  .contact-link { color: var(--ink3); font: var(--type-meta); }
  @media (max-width: 720px) {
    .dlg-head { padding: var(--s4) var(--s4) 0; }
    .body { padding: var(--s4); }
  }
</style>
