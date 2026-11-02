/// <reference types="vite/client" />
import { mount } from "svelte";
import { HttpRuntimeClient } from "./lib/runtime/http-client";
import { installRuntimeClient } from "./lib/runtime/registry";
import { startTheme } from "./lib/theme";
import { initializeInputModality } from "./lib/inputModality";
import { loadWebMcp } from "./lib/webmcp/loader";
// Side-effect CSS imports, extracted into the bundled CSS. Ledger first (its faces, tokens and base),
// then the app's own globals. Imported here (not from App.svelte) so svelte-check, which runs
// without Vite's CSS plugin, doesn't trip on missing module declarations; vite/client above
// declares ``*.css``.
import "@stdlogits/ledger/ledger.css";
import "./lib/style/global.css";

const target = document.getElementById("app");
if (!target) throw new Error("drowse web: #app element missing in index.html");

startTheme();
initializeInputModality();
void loadWebMcp()?.then(webMcp => webMcp.startWebMcp(false));

installRuntimeClient(new HttpRuntimeClient());
const { default: App } = await import("./App.svelte");
const app = mount(App, { target });
export default app;
