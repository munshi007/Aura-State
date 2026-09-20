import React from "react";
import { useStore } from "../store";

const RAIL = 52;                 // rail column width (keep in sync with .main grid)
const LIMITS = { tree: [170, 520], insp: [260, 680] } as const;

/** A thin drag bar overlaying a panel boundary; rendered by App inside `.main`
 *  (NOT inside the scrolling panel, so it stays put while the panel scrolls). */
export function Resizer({ side }: { side: "tree" | "insp" }) {
  const { treeW, inspW, treeCollapsed, inspCollapsed, setPanelWidth } = useStore();
  if (side === "tree" ? treeCollapsed : inspCollapsed) return null;
  const style: React.CSSProperties = side === "tree" ? { left: RAIL + treeW - 3 } : { right: inspW - 3 };
  const onDown = (e: React.PointerEvent) => {
    e.preventDefault();
    const [min, max] = LIMITS[side];
    const move = (ev: PointerEvent) => {
      const w = side === "tree" ? ev.clientX - RAIL : window.innerWidth - ev.clientX;
      setPanelWidth(side, Math.max(min, Math.min(max, Math.round(w))));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.style.userSelect = "";
    };
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  return <div className="resizer" style={style} onPointerDown={onDown}
    title="Drag to resize" role="separator" aria-orientation="vertical" />;
}

/** Small ‹/› button to collapse a panel; shown in the panel header. */
export function CollapseBtn({ which }: { which: "tree" | "insp" }) {
  const toggle = useStore((s) => s.togglePanel);
  return (
    <button className="collapse-btn" title="Collapse panel" aria-label="Collapse panel"
      onClick={() => toggle(which)}>{which === "tree" ? "‹" : "›"}</button>
  );
}
