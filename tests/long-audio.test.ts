import assert from "node:assert/strict";
import { test } from "node:test";
import { planAudioRanges, retryDelayMs, suppressPromptEcho } from "../src/long-audio.ts";
import { CloudError, cloudHttpError } from "../src/cloud.ts";
import { aiPassBudget, completedAiText } from "../src/ai-pass.ts";

test("short audio keeps one range, continuous long audio uses contiguous 300 second cuts", () => {
  assert.deepEqual(planAudioRanges(330, []), [{ startSeconds: 0, endSeconds: 330 }]);
  assert.deepEqual(planAudioRanges(950, []), [
    { startSeconds: 0, endSeconds: 300 }, { startSeconds: 300, endSeconds: 600 },
    { startSeconds: 600, endSeconds: 900 }, { startSeconds: 900, endSeconds: 950 },
  ]);
});

test("planner picks nearest pause midpoint inside each window without gaps", () => {
  assert.deepEqual(planAudioRanges(920, [
    { startSeconds: 275, endSeconds: 277 }, { startSeconds: 305, endSeconds: 307 },
    { startSeconds: 601, endSeconds: 605 }, { startSeconds: 660, endSeconds: 665 },
  ]), [
    { startSeconds: 0, endSeconds: 306 }, { startSeconds: 306, endSeconds: 603 },
    { startSeconds: 603, endSeconds: 920 },
  ]);
});

test("pause spanning the planning window gives a cut inside the pause and window", () => {
  assert.deepEqual(planAudioRanges(600, [{ startSeconds: 100, endSeconds: 900 }]), [
    { startSeconds: 0, endSeconds: 300 }, { startSeconds: 300, endSeconds: 600 },
  ]);
});

test("only transient provider errors retry, with Retry-After or 1s/4s fallback", () => {
  for (const status of [408, 429, 500, 503, 599]) {
    const error = cloudHttpError(status, null, null);
    assert.equal(retryDelayMs(error, 0), 1000);
    assert.equal(retryDelayMs(error, 1), 4000);
    assert.equal(retryDelayMs(error, 2), null);
  }
  for (const status of [400, 401, 402, 403, 404, 413, 422]) {
    assert.equal(retryDelayMs(cloudHttpError(status, null, null), 0), null);
  }
  assert.equal(retryDelayMs(cloudHttpError(429, { error: { code: "insufficient_quota" } }, null), 0), null);
  assert.equal(retryDelayMs(cloudHttpError(429, null, "3"), 0), 3000);
  assert.equal(retryDelayMs(new CloudError("network", true), 0), 1000);
  assert.equal(retryDelayMs(new Error("missing text"), 0), null);
});

test("prompt echo is suppressed per part without transcript deduplication", () => {
  assert.equal(suppressPromptEcho("BB, DeepSeek.", "BB DeepSeek"), "");
  assert.equal(suppressPromptEcho("BB работает. BB работает.", "BB"), "BB работает. BB работает.");
});

test("AI completion budget scales with actual text", () => {
  assert.deepEqual(aiPassBudget(100), { maxTokens: 4096, timeoutMs: 30000 });
  assert.deepEqual(aiPassBudget(25393), { maxTokens: 12697, timeoutMs: 92996 });
});

test("only explicit stop and nonempty cleaned content can replace original", () => {
  assert.equal(completedAiText({ choices: [{ finish_reason: "stop", message: { content: "<think>x</think>Текст." } }] }), "Текст.");
  for (const finish of ["length", "content_filter", "tool_calls", null, undefined]) {
    assert.throws(() => completedAiText({ choices: [{ finish_reason: finish, message: { content: "Обрезанный текст" } }] }));
  }
  for (const payload of [null, {}, { choices: [null] }, { choices: [{ finish_reason: "stop", message: { content: "<think>only reasoning</think>" } }] }, { choices: [{ finish_reason: "stop", message: { content: [] } }] }]) {
    assert.throws(() => completedAiText(payload));
  }
});
