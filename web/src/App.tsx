import { useCallback, useEffect, useRef, useState } from "react";
import { Turnstile, type TurnstileInstance } from "@marsidev/react-turnstile";
import { convertImage, type ConvertOptions, type ConvertResult, type Engine, type FreqScale } from "./convert";
import "./App.css";

const SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY as string;

function formatBytes(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function App() {
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
    if (!file || !token) return;
    setBusy(true);
    setError(null);
    try {
      const dur = duration.trim() === "" ? null : Number(duration);
      if (dur !== null && (!Number.isFinite(dur) || dur < 1 || dur > 60)) {
        throw new Error("时长必须是 1–60 之间的秒数（留空表示按图片宽度自动）");
      }
      const opts: ConvertOptions = { engine, freqScale, fmin, fmax, duration: dur, minDb };
      const result = await convertImage(file, token, opts);
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
      {file && (
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
          <span className="range-row">
            <input
              type="range"
              min={20}
              max={2000}
              step={10}
              value={fmin}
              onChange={(e) => setFmin(Math.min(Number(e.target.value), fmax - 100))}
            />
            <input
              type="range"
              min={100}
              max={20000}
              step={100}
              value={fmax}
              onChange={(e) => setFmax(Math.max(Number(e.target.value), fmin + 100))}
            />
          </span>
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
          disabled={!file || !token || busy}
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
