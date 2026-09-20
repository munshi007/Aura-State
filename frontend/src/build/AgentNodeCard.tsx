import React from "react";
import { Handle, Position } from "reactflow";
import { KIND, StatusChip } from "../ui";
import { useStore } from "../store";
import { statusReason } from "../explain";
import type { AgentNode, Capability } from "../api";
import type { Status } from "../store";

const CAP_SHORT: Record<Capability, string> = { plain: "", untrusted: "untrusted", sink: "sink", sanitizer: "sanitizer" };

export default function AgentNodeCard({ data, selected }: { data: { node: AgentNode; status: Status; current?: boolean; dim?: boolean }; selected: boolean }) {
  const { node, status, current, dim } = data;
  const cap = CAP_SHORT[node.capability];
  const diff = useStore((s) => s.diffOverlay[node.id]);
  const verify = useStore((s) => s.verify);
  const reason = statusReason(node.id, verify) || undefined;
  return (
    <div className={"node" + (selected ? " sel" : "") + (diff ? " diff-" + diff : "") + (current ? " trace-current" : "") + (dim ? " trace-dim" : "")}>
      {current && <span className="difftag" style={{ color: "var(--ink)" }}>running</span>}
      {diff && <span className="difftag">{diff}</span>}
      <Handle type="target" position={Position.Left} />
      <div className="nh">
        <span className="ty" style={{ background: KIND[node.kind].shade }} />
        <span className="nm">{node.id}</span>
        <span className="st" title={reason}><StatusChip status={status} /></span>
      </div>
      <div className="nb">
        <div className="kv"><b>{KIND[node.kind].label}</b>{cap && <span>· {cap}</span>}</div>
        {node.kind === "extract" && <div className="kv">{node.model} · {node.fields.length} field{node.fields.length !== 1 ? "s" : ""}{node.consensus > 1 ? ` · ×${node.consensus}` : ""}</div>}
        {node.obligations.length > 0 && <div className="obl" title={node.obligations.join("  ∧  ")}>{node.obligations.join("  ∧  ")}</div>}
        {node.kind === "decision" && node.sandbox_rule && <div className="obl" title={node.sandbox_rule}>{node.sandbox_rule}</div>}
        {node.kind === "tool" && node.tool_name && <div className="obl" title={"declared tool · " + (node.side_effect || "")}>{node.tool_name}()</div>}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
