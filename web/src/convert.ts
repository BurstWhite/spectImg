export type FreqScale = "linear" | "log" | "mel";
export type Engine = "grad" | "sines" | "gl";

export interface ConvertOptions {
  engine: Engine;
  freqScale: FreqScale;
  fmin: number;
  fmax: number;
  duration: number | null;
  minDb: number;
}

export interface ConvertResult {
  id: number;
  name: string;
  url: string; // object URL, lives only in this browser tab
  seconds: number;
  sampleRate: number;
  sizeBytes: number;
  createdAt: Date;
}

async function errorFromResponse(res: Response): Promise<Error> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail ?? body);
    return new Error(`server ${res.status}: ${detail}`);
  } catch {
    return new Error(`server ${res.status}`);
  }
}

export async function convertImage(
  file: File,
  turnstileToken: string,
  opts: ConvertOptions,
): Promise<ConvertResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("turnstile_token", turnstileToken);
  form.append("engine", opts.engine);
  form.append("freq_scale", opts.freqScale);
  form.append("fmin", String(opts.fmin));
  form.append("fmax", String(opts.fmax));
  form.append("min_db", String(opts.minDb));
  if (opts.duration !== null) form.append("duration", String(opts.duration));

  const res = await fetch("/api/convert", { method: "POST", body: form });
  if (!res.ok) throw await errorFromResponse(res);

  const blob = await res.blob();
  if (blob.size === 0) throw new Error("server returned an empty file");
  const seconds = Number(res.headers.get("X-Synthesis-Seconds") ?? 0);
  const sampleRate = Number(res.headers.get("X-Sample-Rate") ?? 0);
  return {
    id: Date.now() + Math.random(),
    name: file.name.replace(/\.[^.]+$/, "") + ".wav",
    url: URL.createObjectURL(blob),
    seconds,
    sampleRate,
    sizeBytes: blob.size,
    createdAt: new Date(),
  };
}
