// Thin typed wrappers over the local FastAPI backend. Every call hits the real
// verifiers on the user's machine — no cloud.
async function post(path: string, body: any) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}
async function get(path: string) {
  const r = await fetch(path);
  return r.json();
}
async function del(path: string) {
  const r = await fetch(path, { method: "DELETE" });
  return r.json();
}
async function postText(path: string, body: any): Promise<string> {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.text();
}
export { postText };

export type Field = { name: string; type: string; description?: string };
export type Capability = "plain" | "untrusted" | "sink" | "sanitizer";
export type NodeKind = "extract" | "decision" | "tool" | "sanitizer";

export interface AgentNode {
  id: string;
  kind: NodeKind;
  capability: Capability;
  system_prompt: string;
  provider?: string;
  model: string;
  temperature: number;
  max_tokens: number;
  fields: Field[];
  obligations: string[];
  sandbox_rule: string;
  consensus: number;
  confidence: number;
  retry: number;
  // Tool nodes: a declared external call. Aura proves its preconditions; it does
  // NOT execute it (that's your code / aura-runtime). Mock a return for testing.
  tool_name?: string;
  side_effect?: "read" | "write" | "external";
  mock_return?: string;
  x: number;
  y: number;
}

// verify uses the graph-spec shape (capability + obligations)
export const verifyGraph = (nodes: any[], edges: string[][], entry?: string) =>
  post("/api/verify", { nodes, edges, entry });

export const runFlow = (nodes: any[], edges: string[][], entry: string, input: string, provider: string, name?: string, memory?: any) =>
  post("/api/run", { nodes, edges, entry, input, provider, name, memory: memory || {} });
export const fetchUrl = (url: string) => post("/api/fetch_url", { url });

export const callAgent = (body: any) => post("/api/agent", body);
export const proveData = (data: any, obligations: string[]) => post("/api/prove", { data, obligations });
export const conformal = (body: any) => post("/api/conformal", body);
export const risk = (body: any) => post("/api/risk", body);
export const verifyDataset = (records: any[], obligations: string[]) =>
  post("/api/verify_dataset", { records, obligations });
export const providers = () => get("/api/providers");
export const testProvider = (name: string) => get("/api/providers/test/" + encodeURIComponent(name));
export const feed = () => get("/api/feed");
export const clearFeed = () => post("/api/feed/clear", {});
export const auditList = (limit = 200) => get("/api/audit?limit=" + limit);
export const auditVerify = () => get("/api/audit/verify");
export const auditLog = (action: string, summary: string, detail: any) => post("/api/audit", { action, summary, detail });
export const AUDIT_EXPORT_URL = "/api/audit/export";
export const listRuns = () => get("/api/runs");
export const getRun = (file: string) => get("/api/runs/" + encodeURIComponent(file));
export const runEval = (nodes: any[], edges: string[][], entry: string, provider: string, cases: any[]) =>
  post("/api/eval", { nodes, edges, entry, provider, cases });
export const ctlCheck = (nodes: any[], edges: string[][], entry: string, properties: any[]) =>
  post("/api/ctl", { nodes, edges, entry, properties });
export const certificate = (name: string, nodes: any[], edges: string[][], entry: string, invariants: string[]) =>
  post("/api/certificate", { name, nodes, edges, entry, invariants });
export const repair = (nodes: any[], edges: string[][], entry: string) =>
  post("/api/repair", { nodes, edges, entry });
export const policyScan = (nodes: any[]) => post("/api/policy/scan", { nodes });
export const tune = (node: string, prompt: string, examples: any[], new_input: string) =>
  post("/api/tune", { node, prompt, examples, new_input });
export const memoryPreview = (history: any[], max_messages: number, required_keys: string[]) =>
  post("/api/memory/preview", { history, max_messages, required_keys });
export const listVersions = (name: string) => get("/api/versions/" + encodeURIComponent(name));
export const getVersion = (name: string, file: string) =>
  get("/api/versions/" + encodeURIComponent(name) + "/" + encodeURIComponent(file));
export const listFlows = () => get("/api/flows");
export const loadFlow = (name: string) => get("/api/flows/" + encodeURIComponent(name));
export const saveFlow = (name: string, flow: any) => post("/api/flows/save", { name, flow });
export const deleteFlow = (name: string) => del("/api/flows/" + encodeURIComponent(name));
export const exportPython = (name: string, nodes: any[], edges: string[][], entry: string) =>
  postText("/api/export/python", { name, nodes, edges, entry });
