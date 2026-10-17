// Необязательный проход модели по распознанному тексту: то, что правилам не
// под силу — падежи, сильно искажённые термины, расстановка абзацев.
// Промпт запрещает пересказ: пользователь хочет свою речь, а не её изложение.

export type AiPassOptions = {
  baseUrl: string;
  model: string;
  apiKey: string;
  /** Словарь терминов: модели показывают, к чему подтягивать искажения. */
  vocabulary: string[];
  timeoutMs: number;
};

/**
 * Промпт перенесён из Voica (`Sources/GroqClient.swift`, MIT) и намеренно узкий:
 * модель правит только термины из словаря и ничего больше. Широкая формулировка
 * вроде «исправь ошибки распознавания» приводит к тому, что модель переписывает
 * верные слова и теряет предлоги — проверено на живой записи.
 */
function buildPrompt(text: string, vocabulary: string[]): string | null {
  if (vocabulary.length === 0) {
    return null;
  }
  return [
    "Ты — корректор диктовки. Ниже словарь терминов пользователя и распознанный текст.",
    "В тексте могут встречаться искажённые варианты этих терминов (речь распознавалась на слух).",
    "Верни ТОЛЬКО исправленный текст: замени искажённые варианты на правильные написания из словаря,",
    "согласуя с падежом и контекстом. Если под искажение подходят несколько терминов словаря —",
    "выбирай наиболее близкий по ЗВУЧАНИЮ к тому, что записано. Если слово в тексте уже совпадает",
    "со словарным термином (пусть и в другом регистре) — оно правильное: не трогай его и не меняй",
    "регистр. Больше ничего не меняй — ни слова, ни пунктуацию. Если исправлять нечего — верни",
    "текст как есть.",
    "",
    `СЛОВАРЬ: ${vocabulary.join(", ")}`,
    "",
    `ТЕКСТ: ${text}`,
  ].join(" ");
}

/**
 * Убрать рассуждения reasoning-моделей. Они пишут ход мысли прямо в ответе
 * тегом `<think>…</think>`, и без вычистки он уезжает в текст пользователя.
 * Незакрытый тег означает обрыв по лимиту токенов — полезного текста дальше нет.
 */
export function stripReasoning(value: string): string {
  let out = value.replace(/<think[^>]*>[\s\S]*?<\/think>/giu, "");
  out = out.replace(/<think[^>]*>[\s\S]*/giu, "");
  return out.trim();
}

export function aiPassBudget(textLength: number): { maxTokens: number; timeoutMs: number } {
  const maxTokens = Math.max(4096, Math.ceil(textLength / 2));
  return { maxTokens, timeoutMs: Math.ceil(30_000 * maxTokens / 4096) };
}

export async function runAiPass(
  text: string,
  options: AiPassOptions,
  fetchImpl: typeof fetch = fetch,
): Promise<string> {
  const prompt = buildPrompt(text, options.vocabulary);
  if (!prompt) {
    // Без словаря проходу нечего исправлять, а лишний запрос только тормозит.
    return text;
  }

  const budget = aiPassBudget(text.length);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Math.min(options.timeoutMs, budget.timeoutMs));
  try {
    const response = await fetchImpl(`${trimTrailingSlash(options.baseUrl)}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${options.apiKey}`,
      },
      body: JSON.stringify({
        model: options.model,
        temperature: 0,
        max_completion_tokens: budget.maxTokens,
        messages: [{ role: "user", content: prompt }],
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`ИИ-проход вернул HTTP ${response.status}`);
    }
    const payload: unknown = await response.json();
    controller.signal.throwIfAborted();
    return completedAiText(payload);
  } finally {
    clearTimeout(timer);
  }
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/u, "");
}

export function completedAiText(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    throw new Error("ИИ-проход вернул некорректный ответ");
  }
  const choices = (payload as { choices?: unknown }).choices;
  const first: unknown = Array.isArray(choices) ? choices[0] : undefined;
  if (!first || typeof first !== "object" ||
      (first as { finish_reason?: unknown }).finish_reason !== "stop") {
    throw new Error("ИИ-проход не завершён полностью");
  }
  const message = (first as { message?: unknown }).message;
  const content = message && typeof message === "object"
    ? (message as { content?: unknown }).content : undefined;
  const cleaned = typeof content === "string" ? stripReasoning(content) : "";
  if (!cleaned) throw new Error("ИИ-проход вернул пустой ответ");
  return cleaned;
}
