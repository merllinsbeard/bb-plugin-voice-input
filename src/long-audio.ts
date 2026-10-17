import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import { CloudError } from "./cloud.ts";

const run = promisify(execFile);
export type AudioRange = { startSeconds: number; endSeconds: number };
type Silence = AudioRange;
export type AudioPart = { path: string; range: AudioRange; converted: boolean };

export function remainingMs(deadline: number): number {
  const remaining = deadline - Date.now();
  if (remaining <= 0) throw new Error("timeout");
  return remaining;
}

export function planAudioRanges(duration: number, silences: Silence[]): AudioRange[] {
  const ranges: AudioRange[] = [];
  let start = 0;
  while (duration - start > 330) {
    const target = start + 300;
    const candidates = silences
      .filter((pause) => pause.endSeconds >= start + 270 && pause.startSeconds <= start + 330)
      .map((pause) => pause.startSeconds <= target && pause.endSeconds >= target
        ? target
        : Math.max(start + 270, Math.min(start + 330, (pause.startSeconds + pause.endSeconds) / 2)))
      .sort((a, b) => Math.abs(a - target) - Math.abs(b - target) || a - b);
    const end = candidates[0] ?? target;
    ranges.push({ startSeconds: start, endSeconds: end });
    start = end;
  }
  ranges.push({ startSeconds: start, endSeconds: duration });
  return ranges;
}

export async function* prepareAudioParts(
  source: string,
  ffmpeg: string | null,
  deadline: number,
): AsyncGenerator<AudioPart> {
  if (!ffmpeg) {
    yield { path: source, range: { startSeconds: 0, endSeconds: 0 }, converted: false };
    return;
  }
  const scan = await run(ffmpeg, [
    "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-i", source, "-map", "0:a:0",
    "-af", "aresample=16000,asetpts=N/SR/TB,silencedetect=noise=-35dB:d=0.3,ametadata=mode=print:file=-,astats=measure_perchannel=none:measure_overall=Number_of_samples",
    "-f", "null", "-",
  ], { timeout: remainingMs(deadline), maxBuffer: 16 * 1024 * 1024 });
  const duration = Number(scan.stderr.match(/Number of samples:\s*(\d+)/)?.[1]) / 16000;
  if (!Number.isFinite(duration) || duration <= 0) throw new Error("ffmpeg decoded no audio");
  const silences: Silence[] = [];
  let pauseStart: number | undefined;
  for (const match of scan.stdout.matchAll(/lavfi\.silence_(start|end)=([\d.]+)/g)) {
    if (match[1] === "start") pauseStart = Number(match[2]);
    else {
      silences.push({ startSeconds: pauseStart ?? 0, endSeconds: Number(match[2]) });
      pauseStart = undefined;
    }
  }
  if (pauseStart !== undefined) silences.push({ startSeconds: pauseStart, endSeconds: duration });
  if (duration <= 330) {
    yield { path: source, range: { startSeconds: 0, endSeconds: duration }, converted: false };
    return;
  }
  const directory = await mkdtemp(join(tmpdir(), "voice-chunks-"));
  try {
    for (const [index, range] of planAudioRanges(duration, silences).entries()) {
      const path = join(directory, `${index}.wav`);
      await run(ffmpeg, [
        "-nostdin", "-v", "error", "-i", source, "-map", "0:a:0",
        "-af", `aresample=16000,atrim=start_sample=${Math.round(range.startSeconds * 16000)}:end_sample=${Math.round(range.endSeconds * 16000)},asetpts=PTS-STARTPTS`,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", path,
      ], { timeout: remainingMs(deadline) });
      try {
        yield { path, range, converted: true };
      } finally {
        await rm(path, { force: true });
      }
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

export function retryDelayMs(error: unknown, attempt: number): number | null {
  if (!(error instanceof CloudError) || !error.transient || attempt >= 2) return null;
  return error.retryAfterMs ?? (attempt === 0 ? 1000 : 4000);
}

export function suppressPromptEcho(text: string, hint: string | null): string {
  const normalize = (value: string) => value.replace(/[,.\s]/g, "").toLowerCase();
  return hint && normalize(text) === normalize(hint) ? "" : text;
}

export async function transcribeCloudAudio(
  source: string,
  ffmpeg: string | null,
  deadline: number,
  hint: string | null,
  transcribe: (part: AudioPart, timeoutMs: number) => Promise<string>,
): Promise<string> {
  const texts: string[] = [];
  for await (const part of prepareAudioParts(source, ffmpeg, deadline)) {
    for (let attempt = 0; ; attempt++) {
      try {
        const text = await transcribe(part, part.converted
          ? Math.min(90_000, remainingMs(deadline)) : remainingMs(deadline));
        texts.push(suppressPromptEcho(text, hint));
        break;
      } catch (error) {
        const wait = part.converted ? retryDelayMs(error, attempt) : null;
        if (wait === null || wait >= remainingMs(deadline)) throw error;
        await delay(wait);
      }
    }
  }
  return texts.filter(Boolean).join(" ");
}
