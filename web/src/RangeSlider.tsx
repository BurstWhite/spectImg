import { useCallback, useRef, useState } from "react";

interface RangeSliderProps {
  min: number;
  max: number;
  step: number;
  lo: number;
  hi: number;
  onChange: (lo: number, hi: number) => void;
  /** Minimum gap between the two handles. */
  minGap?: number;
  /** Format a value for the tick labels. */
  format?: (v: number) => string;
}

/** A single-track, two-handle range slider.
 *
 * The native <input type="range"> is single-value only, so this draws one
 * track with two pointer-draggable handles (the nearer one wins when both
 * could apply). Keyboard: focus a handle, arrow keys move it.
 */
export default function RangeSlider({
  min,
  max,
  step,
  lo,
  hi,
  onChange,
  minGap = 1,
  format = (v) => String(v),
}: RangeSliderProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [focus, setFocus] = useState<"lo" | "hi" | null>(null);
  const clamp = (v: number) => Math.min(max, Math.max(min, v));

  const valueFromPointer = useCallback(
    (clientX: number): number => {
      const el = trackRef.current;
      if (!el) return min;
      const rect = el.getBoundingClientRect();
      const t = (clientX - rect.left) / Math.max(rect.width, 1);
      const raw = min + clamp01(t) * (max - min);
      return Math.round(raw / step) * step;
    },
    [min, max, step],
  );

  const handlePointer = (e: React.PointerEvent, which: "lo" | "hi") => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => {
      const v = valueFromPointer(ev.clientX);
      if (which === "lo") onChange(Math.min(v, hi - minGap), hi);
      else onChange(lo, Math.max(v, lo + minGap));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const onTrackPointerDown = (e: React.PointerEvent) => {
    const v = valueFromPointer(e.clientX);
    // grab whichever handle is nearer (or the one being pushed past)
    const which = Math.abs(v - lo) <= Math.abs(v - hi) ? "lo" : "hi";
    if (which === "lo") onChange(Math.min(v, hi - minGap), hi);
    else onChange(lo, Math.max(v, lo + minGap));
  };

  const onKey = (e: React.KeyboardEvent, which: "lo" | "hi") => {
    const delta = e.shiftKey ? step * 10 : step;
    let v = which === "lo" ? lo : hi;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") v -= delta;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") v += delta;
    else if (e.key === "Home") v = which === "lo" ? min : lo + minGap;
    else if (e.key === "End") v = which === "hi" ? max : hi - minGap;
    else return;
    e.preventDefault();
    v = clamp(v);
    if (which === "lo") onChange(Math.min(v, hi - minGap), hi);
    else onChange(lo, Math.max(v, lo + minGap));
  };

  const pct = (v: number) => ((v - min) / (max - min)) * 100;

  return (
    <div
      className={`range-slider ${focus ? "focused" : ""}`}
      ref={trackRef}
      onPointerDown={onTrackPointerDown}
    >
      <div className="rs-track" />
      <div
        className="rs-fill"
        style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%` }}
      />
      <div
        className={`rs-handle ${focus === "lo" ? "active" : ""}`}
        role="slider"
        aria-label="最低频率"
        aria-valuemin={min}
        aria-valuemax={hi - minGap}
        aria-valuenow={lo}
        tabIndex={0}
        style={{ left: `${pct(lo)}%` }}
        onPointerDown={(e) => handlePointer(e, "lo")}
        onKeyDown={(e) => onKey(e, "lo")}
        onFocus={() => setFocus("lo")}
        onBlur={() => setFocus(null)}
      />
      <div
        className={`rs-handle ${focus === "hi" ? "active" : ""}`}
        role="slider"
        aria-label="最高频率"
        aria-valuemin={lo + minGap}
        aria-valuemax={max}
        aria-valuenow={hi}
        tabIndex={0}
        style={{ left: `${pct(hi)}%` }}
        onPointerDown={(e) => handlePointer(e, "hi")}
        onKeyDown={(e) => onKey(e, "hi")}
        onFocus={() => setFocus("hi")}
        onBlur={() => setFocus(null)}
      />
      <span className="rs-min">{format(min)}</span>
      <span className="rs-max">{format(max)}</span>
    </div>
  );
}

function clamp01(t: number): number {
  return Math.min(1, Math.max(0, t));
}
