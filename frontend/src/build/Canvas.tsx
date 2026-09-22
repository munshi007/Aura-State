import React, { useMemo, useCallback, useEffect, useRef } from "react";
import ReactFlow, {
  Background, BackgroundVariant, Controls, MarkerType,
  NodeChange, Connection, ReactFlowProvider, useReactFlow,
} from "reactflow";
import { useStore } from "../store";
import { Icon, StatusChip } from "../ui";
import { stepReason } from "../explain";
import AgentNodeCard from "./AgentNodeCard";

const nodeTypes = { agent: AgentNodeCard };

function CanvasInner() {
  const { nodes, edges, selectedId, statusByNode, select, moveNode, connect, loadTemplate,
    traceActive, traceIndex, runTrace, set } = useStore();
  const rf = useReactFlow();
  const count = nodes.length;
  const prevCount = useRef(count);
  useEffect(() => {
    if (prevCount.current !== count) {
      prevCount.current = count;
      const t = setTimeout(() => rf.fitView({ padding: 0.2, maxZoom: 1.2, duration: 300 }), 60);
      return () => clearTimeout(t);
    }
  }, [count, rf]);

  // ── Trace overlay ──
  const trace = traceActive && runTrace && runTrace.some((t: any) => t.node !== "—") ? runTrace : null;
  const executed = trace ? trace.slice(0, traceIndex + 1) : [];
  const currentNode = trace ? trace[traceIndex]?.node : null;
  const traceStatus: Record<string, string> = {};
  const executedSet = new Set<string>();
  const takenEdges = new Set<string>();
  executed.forEach((s: any) => {
    executedSet.add(s.node);
    traceStatus[s.node] = s.verified === false ? "violated" : s.verified === true ? "proven" : "pending";
    if (s.next && s.next !== "END") takenEdges.add(s.node + "→" + s.next);
  });

  useEffect(() => {
    // gently focus the current node while scrubbing
    if (trace && currentNode) {
      const n = nodes.find((x) => x.id === currentNode);
      if (n) rf.setCenter(n.x + 93, n.y + 45, { zoom: 1, duration: 260 });
    }
  }, [traceIndex, traceActive]);

  const rfNodes = useMemo(
    () => nodes.map((n) => ({
      id: n.id, type: "agent", position: { x: n.x, y: n.y },
      data: {
        node: n,
        status: trace ? (traceStatus[n.id] || "pending") : (statusByNode[n.id] || "pending"),
        current: trace ? n.id === currentNode : false,
        dim: trace ? !executedSet.has(n.id) : false,
      },
      selected: !trace && selectedId === n.id,
    })),
    [nodes, statusByNode, selectedId, trace, traceIndex]
  );

  const rfEdges = useMemo(
    () => edges.map(([a, b], i) => {
      const target = nodes.find((n) => n.id === b);
      const gated = target?.capability === "sink";
      const taken = trace ? takenEdges.has(a + "→" + b) : false;
      return {
        id: `${a}-${b}-${i}`, source: a, target: b,
        animated: taken,
        style: trace
          ? (taken ? { stroke: "var(--ink)", strokeWidth: 2.4 } : { opacity: 0.18 })
          : (gated ? { strokeDasharray: "5 4" } : undefined),
        markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: taken ? "var(--ink)" : "var(--ink-4)" },
        label: !trace && gated ? "risk-gated" : undefined,
        labelStyle: { fontFamily: "JetBrains Mono", fontSize: 9, fill: "var(--ink-3)" },
        labelBgStyle: { fill: "var(--panel)" },
      };
    }),
    [edges, nodes, trace, traceIndex]
  );

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    changes.forEach((c) => { if (c.type === "position" && c.position) moveNode(c.id, c.position.x, c.position.y); });
  }, [moveNode]);
  const onConnect = useCallback((c: Connection) => { if (c.source && c.target) connect(c.source, c.target); }, [connect]);

  return (
    <div className="canvas">
      {count === 0 && (
        <div className="cv-empty">
          <div className="lbl" style={{ marginBottom: 10 }}>Empty canvas</div>
          <div style={{ color: "var(--ink-3)", marginBottom: 16, lineHeight: 1.6 }}>Add a node from the left,<br />or start from a template.</div>
          <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
            <button className="btn sm" onClick={() => loadTemplate("refund")}><Icon name="spark" size={13} /> Refund agent</button>
            <button className="btn sm" onClick={() => loadTemplate("lead")}><Icon name="spark" size={13} /> Lead qualifier</button>
          </div>
        </div>
      )}
      {count > 0 && !trace && (
        <div className="cv-hint">drag between node edges to connect · click a node to configure · <b>Verify design</b> to prove it</div>
      )}
      <ReactFlow
        nodes={rfNodes} edges={rfEdges} nodeTypes={nodeTypes}
        onNodesChange={onNodesChange} onConnect={onConnect}
        onNodeClick={(_, n) => !trace && select(n.id)} onPaneClick={() => !trace && select(null)}
        fitView fitViewOptions={{ padding: 0.2, maxZoom: 1.2 }} minZoom={0.35} maxZoom={1.8}
        proOptions={{ hideAttribution: true }} defaultEdgeOptions={{ type: "smoothstep" }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--line)" />
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
      {trace ? <TraceBar trace={trace} /> : (
        <div className="cv-legend">
          <span><span className="dot" style={{ background: "var(--proven)" }} /> proven</span>
          <span><span className="dot" style={{ background: "var(--pending)" }} /> pending</span>
          <span><span className="dot" style={{ background: "var(--violated)" }} /> violated</span>
        </div>
      )}
    </div>
  );
}

function TraceBar({ trace }: { trace: any[] }) {
  const { traceIndex, runHealth, edges, set } = useStore();
  const s = trace[traceIndex];
  const n = trace.length;
  const go = (i: number) => set({ traceIndex: Math.max(0, Math.min(n - 1, i)) });
  const st = s?.verified === false ? "violated" : s?.verified === true ? "proven" : "pending";
  const h = runHealth && s ? runHealth[s.node] : null;
  const candidates = s ? edges.filter(([a]) => a === s.node).map(([, b]) => b) : [];
  return (
    <div className="tracebar">
      <div className="tb-top">
        <span className="lbl" style={{ color: "var(--ink)" }}>Execution trace</span>
        <div className="tb-ctrl">
          <button className="icobtn" aria-label="First" onClick={() => go(0)}>⏮</button>
          <button className="icobtn" aria-label="Prev" onClick={() => go(traceIndex - 1)}>◀</button>
          <span className="mono" style={{ fontSize: 11, minWidth: 54, textAlign: "center" }}>step {traceIndex + 1} / {n}</span>
          <button className="icobtn" aria-label="Next" onClick={() => go(traceIndex + 1)}>▶</button>
          <button className="icobtn" aria-label="Last" onClick={() => go(n - 1)}>⏭</button>
        </div>
        <button className="btn sm" onClick={() => set({ traceActive: false })}><Icon name="build" size={13} /> Exit trace</button>
      </div>
      <div className="tb-dots">
        {trace.map((t: any, i: number) => {
          const ts = t.verified === false ? "violated" : t.verified === true ? "proven" : "pending";
          return <button key={i} className={"tb-dot" + (i === traceIndex ? " on" : "") + (i <= traceIndex ? " done" : "")}
            style={{ background: i <= traceIndex ? (ts === "violated" ? "var(--violated)" : ts === "proven" ? "var(--proven)" : "var(--pending)") : "var(--line)" }}
            title={t.node} onClick={() => go(i)} />;
        })}
      </div>
      <div className="tb-detail">
        <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 6 }}>
          <b className="mono">{s?.node}</b>
          {s?.verified != null && <StatusChip status={st as any} />}
          {s?.next && <span className="mono" style={{ color: "var(--ink-3)", fontSize: 11 }}>→ {s.next}</span>}
        </div>
        <div style={{ fontSize: 12, color: "var(--ink-2)", lineHeight: 1.5 }}>{stepReason(s)}</div>
        {candidates.length > 1 && s?.next && (
          <div className="tb-route">
            <span className="lbl" style={{ color: "var(--ink-2)" }}>routing</span>
            {candidates.map((c) => (
              <span key={c} className={"tb-branch" + (c === s.next ? " taken" : "")}>{c === s.next ? "→ " : ""}{c}</span>
            ))}
            <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>rule-routed · CTL-feasible</span>
          </div>
        )}
        {s?.extracted && Object.keys(s.extracted).length > 0 && <pre style={{ marginTop: 8 }}>{JSON.stringify(s.extracted)}</pre>}
        <div className="meta" style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--ink-3)", marginTop: 8, display: "flex", gap: 14, flexWrap: "wrap" }}>
          {s?.ms != null && <span>{s.ms} ms</span>}
          {s?.model && <span>{s.provider ? s.provider + " · " : ""}{s.model}</span>}
          {s?.consensus > 1 && <span>consensus ×{s.consensus}</span>}
          {h && <span>health · {h.total_executions}× · {h.avg_latency_ms}ms · fail {(h.fail_rate * 100).toFixed(0)}%</span>}
        </div>
      </div>
    </div>
  );
}

export default function Canvas() {
  return <ReactFlowProvider><CanvasInner /></ReactFlowProvider>;
}
