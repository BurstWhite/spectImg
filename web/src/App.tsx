import { useCallback, useEffect, useRef, useState } from "react";
import { Turnstile, type TurnstileInstance } from "@marsidev/react-turnstile";
import { convertImage, type ConvertOptions, type ConvertResult, type Engine, type FreqScale } from "./convert";
import RangeSlider from "./RangeSlider";
import "./App.css";

const SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY as string;

function formatBytes(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

const FONT_FAMILIES = [
  { label: "黑体（无衬线）", value: "bold {px}px system-ui, -apple-system, 'PingFang SC', sans-serif" },
  { label: "宋体（衬线）", value: "bold {px}px 'Songti SC', Georgia, serif" },
  { label: "等宽", value: "bold {px}px 'SF Mono', Menlo, monospace" },
];

/**Rasterize text to a white-on-black PNG File, sized to the text.
 *
 * White on black: in the spectrogram, lit pixels become sound and black is
 * silence, so the letters glow out of a quiet background.
 */
async function renderTextImage(
  text: string,
  fontTemplate: string,
  fontSize: number,
): Promise<File | null> {
  const lines = text.split("\n").filter((l) => l.trim() !== "");
  if (lines.length === 0) return null;
  const font = fontTemplate.replace("{px}", String(fontSize));

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.font = font;
  const widths = lines.map((l) => ctx.measureText(l).width);
  const pad = Math.ceil(fontSize * 0.4);
  canvas.width = Math.min(Math.ceil(Math.max(...widths)) + pad * 2, 2000);
  canvas.height = lines.length * Math.ceil(fontSize * 1.25) + pad * 2;

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.font = font;
  ctx.textBaseline = "top";
  const lineHeight = Math.ceil(fontSize * 1.25);
  lines.forEach((line, i) => {
    ctx.fillText(line, pad, pad + i * lineHeight);
  });

  const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/png"));
  if (!blob) return null;
  return new File([blob], "text.png", { type: "image/png" });
}

export default function App() {
  const [mode, setMode] = useState<"image" | "text">("image");
  const [text, setText] = useState("");
  const [fontIdx, setFontIdx] = useState(0);
  const [fontSize, setFontSize] = useState(120);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [freqScale, setFreqScale] = useState<FreqScale>("linear");
  const [engine, setEngine] = useState<Engine>("grad");
  const [fmin, setFmin] = useState(20);
  const [fmax, setFmax] = useState(20000);
  const [duration, setDuration] = useState<string>("");
  const [minDb, setMinDb] = useState(-80);
  const [token, setToken] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ConvertResult[]>([]);
  const tsRef = useRef<TurnstileInstance>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const acceptFile = useCallback((f: File | null) => {
    setError(null);
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("请选择图片文件");
      return;
    }
    if (f.size > 12 * 1024 * 1024) {
      setError("图片超过 12 MB 上限");
      return;
    }
    setFile(f);
  }, []);

  const onConvert = async () => {
    let target = file;
    if (mode === "text") {
      target = await renderTextImage(text, FONT_FAMILIES[fontIdx].value, fontSize);
      if (!target) {
        setError("请输入一些文字");
        return;
      }
    }
    if (!target || !token) return;
    setBusy(true);
    setError(null);
    try {
      const dur = duration.trim() === "" ? null : Number(duration);
      if (dur !== null && (!Number.isFinite(dur) || dur < 1 || dur > 60)) {
        throw new Error("时长必须是 1–60 之间的秒数（留空表示按图片宽度自动）");
      }
      const opts: ConvertOptions = { engine, freqScale, fmin, fmax, duration: dur, minDb };
      const result = await convertImage(target, token, opts);
      setResults((r) => [result, ...r]);
      // tokens are single-use: force the widget to issue a fresh one
      setToken(null);
      tsRef.current?.reset();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setToken(null);
      tsRef.current?.reset();
    } finally {
      setBusy(false);
    }
  };

  const removeResult = (id: number) => {
    setResults((rs) => {
      const target = rs.find((r) => r.id === id);
      if (target) URL.revokeObjectURL(target.url);
      return rs.filter((r) => r.id !== id);
    });
  };

  return (
    <div className="page">
      <header>
        <h1>img2spec</h1>
        <p className="sub">上传一张图片，生成一段频谱就是这张图的音频（.wav）</p>
      </header>

      <div className="mode-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={mode === "image"}
          className={mode === "image" ? "active" : ""}
          onClick={() => setMode("image")}
        >
          上传图片
        </button>
        <button
          role="tab"
          aria-selected={mode === "text"}
          className={mode === "text" ? "active" : ""}
          onClick={() => setMode("text")}
        >
          输入文字
        </button>
      </div>

      {mode === "text" ? (
        <section className="text-panel">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="输入要藏进频谱的文字，支持多行"
            rows={3}
          />
          <div className="text-controls">
            <label>
              字体
              <select value={fontIdx} onChange={(e) => setFontIdx(Number(e.target.value))}>
                {FONT_FAMILIES.map((f, i) => (
                  <option key={f.label} value={i}>
                    {f.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="grow">
              字号 {fontSize}px
              <input
                type="range"
                min={40}
                max={220}
                step={10}
                value={fontSize}
                onChange={(e) => setFontSize(Number(e.target.value))}
              />
            </label>
          </div>
          {text.trim() !== "" && (
            <p className="text-preview">
              <span
                style={{
                  fontFamily: FONT_FAMILIES[fontIdx].value.replace(/^bold \d+px /, "").replace(/\{px\}/, ""),
                  fontSize: "min(2rem, 6vw)",
                }}
              >
                {text.split("\n").filter((l) => l.trim()).join(" / ")}
              </span>
            </p>
          )}
        </section>
      ) : (
      <section
        className={`dropzone ${dragOver ? "drag" : ""} ${previewUrl ? "has-image" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          acceptFile(e.dataTransfer.files?.[0] ?? null);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            acceptFile(e.target.files?.[0] ?? null);
            e.target.value = ""; // allow re-selecting the same file later
          }}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="待转换图片预览" />
        ) : (
          <div className="hint">
            <strong>点击或拖入图片</strong>
            <span>PNG / JPG / WebP，最大 12 MB</span>
          </div>
        )}
      </section>
      )}
      {mode === "image" && file && (
        <p className="fileinfo">
          {file.name} · {formatBytes(file.size)}
          <button className="linklike" onClick={() => setFile(null)}>
            移除
          </button>
        </p>
      )}

      <section className="options">
        <label>
          还原质量
          <select value={engine} onChange={(e) => setEngine(e.target.value as Engine)}>
            <option value="grad">最还原（约 20–40 秒）</option>
            <option value="sines">快速（几秒，适合线条图）</option>
            <option value="gl">经典 Griffin-Lim</option>
          </select>
        </label>
        <label>
          频率轴
          <select
            value={freqScale}
            onChange={(e) => setFreqScale(e.target.value as FreqScale)}
          >
            <option value="linear">linear（Audition 默认显示）</option>
            <option value="log">log（查看器切到对数刻度时用）</option>
            <option value="mel">mel</option>
          </select>
        </label>
        <label>
          时长（秒，留空自动）
          <input
            type="number"
            min={1}
            max={60}
            placeholder="自动"
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
          />
        </label>
        <label className="wide">
          频率范围 {fmin} Hz – {fmax} Hz
          <RangeSlider
            min={20}
            max={20000}
            step={10}
            lo={fmin}
            hi={fmax}
            minGap={100}
            onChange={(lo, hi) => {
              setFmin(lo);
              setFmax(hi);
            }}
            format={(v) => (v >= 1000 ? `${v / 1000}kHz` : `${v}Hz`)}
          />
        </label>
        <label className="wide">
          最暗像素电平 {minDb} dB
          <input
            type="range"
            min={-100}
            max={-40}
            step={5}
            value={minDb}
            onChange={(e) => setMinDb(Number(e.target.value))}
          />
        </label>
      </section>

      <section className="gate">
        <Turnstile
          ref={tsRef}
          siteKey={SITE_KEY}
          onSuccess={setToken}
          onExpire={() => setToken(null)}
          onError={() => setToken(null)}
          options={{ theme: "auto" }}
        />
        <button
          className="convert"
          disabled={(mode === "image" ? !file : text.trim() === "") || !token || busy}
          onClick={onConvert}
        >
          {busy ? "合成中…（可能需要十几到几十秒，请勿关闭页面）" : "转换为 .wav"}
        </button>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="results">
        {results.length === 0 ? (
          <p className="empty">转换结果会保存在本页面（浏览器内存）里，可试听和下载。</p>
        ) : (
          results.map((r) => (
            <div className="result" key={r.id}>
              <div className="meta">
                <strong>{r.name}</strong>
                <span>
                  {r.seconds.toFixed(1)}s · {r.sampleRate ? `${r.sampleRate / 1000}kHz · ` : ""}
                  {formatBytes(r.sizeBytes)} · {r.createdAt.toLocaleTimeString()}
                </span>
              </div>
              <audio controls src={r.url} preload="metadata" />
              <div className="actions">
                <a className="button" href={r.url} download={r.name}>
                  下载 .wav
                </a>
                <button className="linklike" onClick={() => removeResult(r.id)}>
                  删除
                </button>
              </div>
            </div>
          ))
        )}
      </section>

      <footer>
        相位由优化算法估计，频谱可辨识但非逐位精确。服务器不保存任何上传内容。
      </footer>
    </div>
  );
}
