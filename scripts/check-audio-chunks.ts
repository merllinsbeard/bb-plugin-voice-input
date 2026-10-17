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
  const rows = [];
  let previousEnd = 0;
  let previousPath: string | undefined;
  let directory: string | undefined;
  for await (const part of prepareAudioParts(source, "/usr/bin/ffmpeg", Date.now() + 600_000)) {
    if (previousPath) await assert.rejects(access(previousPath));
    const { stdout } = await run("/usr/bin/ffprobe", [
      "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels:format=duration,size",
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
      previousPath = part.path;
      directory = dirname(part.path);
    } else {
      assert.equal(part.path, source);
      assert.ok(duration <= 330);
    }
    rows.push({ ...part.range, converted: part.converted, duration, bytes: probe.format.size });
    previousEnd = part.range.endSeconds;
  }
  if (directory) await assert.rejects(access(directory));
  console.log(JSON.stringify({ source, parts: rows, cleanup: true }));
}
