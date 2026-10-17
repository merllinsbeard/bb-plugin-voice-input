import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, stat } from "node:fs/promises";
import { dirname } from "node:path";
import { promisify } from "node:util";
import { prepareAudioParts } from "../src/long-audio.ts";

const run = promisify(execFile);
const cleanupOnly = process.argv[2] === "--cleanup";
for (const source of process.argv.slice(cleanupOnly ? 3 : 2)) {
  if (cleanupOnly) {
    const parts = prepareAudioParts(source, "/usr/bin/ffmpeg", Date.now() + 600_000);
    const first = await parts.next();
    assert.ok(!first.done && first.value.converted);
    const directory = dirname(first.value.path);
    await assert.rejects(parts.throw(new Error("consumer stopped")), /consumer stopped/);
    await assert.rejects(access(directory));
    console.log(JSON.stringify({ source, consumerErrorCleanup: true }));
    continue;
  }
  const decoded = await run("/usr/bin/ffprobe", [
    "-v", "error", "-select_streams", "a:0", "-show_frames",
    "-show_entries", "frame=nb_samples:stream=sample_rate", "-of", "json", source,
  ], { maxBuffer: 16 * 1024 * 1024 });
  const original = JSON.parse(decoded.stdout);
  const decodedDuration = original.frames.reduce((sum: number, frame: { nb_samples: number }) =>
    sum + frame.nb_samples, 0) / Number(original.streams[0].sample_rate);
  const originalSamples = Math.round(decodedDuration * 16000);
  let chunkSamples = 0;
  const rows = [];
  let previousEnd = 0;
  let previousPath: string | undefined;
  let directory: string | undefined;
  for await (const part of prepareAudioParts(source, "/usr/bin/ffmpeg", Date.now() + 600_000)) {
    if (previousPath) await assert.rejects(access(previousPath));
    const { stdout } = await run("/usr/bin/ffprobe", [
      "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels,duration_ts:format=duration,size",
      "-of", "json", part.path,
    ]);
    const probe = JSON.parse(stdout);
    const duration = Number(probe.format.duration);
    assert.equal(part.range.startSeconds, previousEnd);
    assert.ok(Math.abs(duration - (part.range.endSeconds - part.range.startSeconds)) < 0.1);
    if (part.converted) {
      assert.equal(probe.streams[0].codec_name, "pcm_s16le");
      assert.equal(probe.streams[0].sample_rate, "16000");
      assert.equal(probe.streams[0].channels, 1);
      assert.ok(duration <= 330.1);
      assert.ok((await stat(part.path)).size <= 330.1 * 32000 + 1024);
      const samples = Number(probe.streams[0].duration_ts);
      if (Math.abs(part.range.endSeconds - decodedDuration) < 0.1) {
        assert.equal(samples, originalSamples - chunkSamples,
          "final chunk must contain every decoded sample through original EOF");
      }
      chunkSamples += samples;
      previousPath = part.path;
      directory = dirname(part.path);
    } else {
      assert.equal(part.path, source);
      assert.ok(duration <= 330);
    }
    rows.push({ ...part.range, converted: part.converted, duration, bytes: probe.format.size,
      samples: part.converted ? Number(probe.streams[0].duration_ts) : undefined });
    previousEnd = part.range.endSeconds;
  }
  assert.ok(Math.abs(previousEnd - decodedDuration) <= 1 / 16000,
    `range endpoint ${previousEnd} must match decoded duration ${decodedDuration}`);
  if (directory) assert.equal(chunkSamples, originalSamples, "chunks must cover the entire decoded input");
  if (directory) await assert.rejects(access(directory));
  console.log(JSON.stringify({ source, decodedDuration, originalSamples, chunkSamples, parts: rows, eofCoverage: true, cleanup: true }));
}
