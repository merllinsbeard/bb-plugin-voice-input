// Хост-часть плагина: BB вызывает её на машине сервера, когда пользователь
// записал голосовое. Здесь распознавание и приведение текста в порядок;
// настройки приходят от серверной части методом applyConfig и переживают
// перезапуск воркера в config.json.
import { mkdir, readFile, readdir, rm, stat, writeFile, appendFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, extname, join } from "node:path";
import { randomUUID } from "node:crypto";
import { hostname, platform } from "node:os";
import { experimental_defineHostEntry } from "@get-bb/plugin-sdk/host";
import { type EngineId, type VoiceConfig, voiceConfigSchema } from "./src/contract.js";
import { hostContract } from "./src/host-contract.js";
import { hostSignals } from "./src/contract.js";
import { downloadModel } from "./src/download.js";
import {
  deleteModel as removeModel,
  listModels as listDiskModels,
  modelSize,
} from "./src/model-store.js";
import { hostPaths, type HostPaths } from "./src/paths.js";
import { readSecrets, writeSecret } from "./src/secret-store.js";
import { resolveAiPassKey, resolveProviderKey } from "./src/key-resolution.js";
import { callDaemon, ensureDaemon, materializePython, stopDaemon } from "./src/daemon.js";
import { engineReady, findFfmpeg, findPython, setupEngine } from "./src/setup.js";
import { buildRecognitionPrompt, parseVocabulary } from "./src/vocabulary.js";
import { cleanupText } from "./src/cleanup.js";
import { repairTerms } from "./src/term-repair.js";
import { TRANSCRIBE_TIMEOUT_MS } from "./src/transcription-budget.js";
import { remainingMs, suppressPromptEcho, transcribeCloudAudio } from "./src/long-audio.js";
import { runAiPass } from "./src/ai-pass.js";
import { guardAiPass } from "./src/ai-pass-guard.js";
import { type CloudRequest, transcribeGemini, transcribeOpenAi } from "./src/cloud.js";

const DEFAULT_CONFIG: VoiceConfig = {
  engine: "whisper",
  machine: "",
  whisperModel: "mlx-community/whisper-large-v3-turbo",
  language: "auto",
  vocabulary: "",
  vocabularyRepair: true,
  vocabularyPrompt: true,
  cleanupFillers: true,
  cleanupQuotes: true,
  cloud: {
    openaiBaseUrl: "https://api.openai.com/v1",
    openaiModel: "gpt-4o-transcribe",
    openaiApiKey: "",
    openaiKeyEnv: "",
    googleBaseUrl: "https://generativelanguage.googleapis.com",
    googleModel: "gemini-3.5-transcribe",
    googleApiKey: "",
    googleKeyEnv: "",
    groqBaseUrl: "https://api.groq.com/openai/v1",
    groqModel: "whisper-large-v3-turbo",
    groqApiKey: "",
    groqKeyEnv: "",
  },
  aiPass: { enabled: false, baseUrl: "https://api.openai.com/v1", model: "", apiKey: "", keyEnv: "" },
  saveRecordings: true,
  keepDays: 14,
};

/** Подсказка Whisper читается моделью лишь частично — держим бюджет Voica. */
const PROMPT_BUDGET_CHARS = 800;
/**
 * Через сколько секунд тишины демон выгружает модель и выходит. Модель весит
 * гигабайты и на машине сервера соседствует с тредами — тёплый старт того не
 * стоит. Следующая диктовка поднимет демон заново, первая фраза будет дольше.
 * 0 вернёт прежнее поведение «держать вечно».
 */
const DAEMON_IDLE_SECONDS = 900;

/** Движки с локальным демоном: только им есть что выгружать. */
const LOCAL_ENGINES: readonly EngineId[] = ["whisper", "gigaam"];

const EXTENSION_BY_MIME: Record<string, string> = {
  "audio/webm": ".webm",
  "audio/ogg": ".ogg",
  "audio/mpeg": ".mp3",
  "audio/mp4": ".m4a",
  "audio/x-m4a": ".m4a",
  "audio/wav": ".wav",
  "audio/x-wav": ".wav",
  "audio/flac": ".flac",
};

/**
 * Прочитать настройки, сохранённые сервером. Схема намеренно применяется
 * послойно: новая версия плагина добавляет поля, а на диске лежит конфиг от
 * прежней. Строгая проверка целиком откатила бы всё на умолчания и молча
 * потеряла бы словарь пользователя.
 */
/**
 * Ключ провайдера: либо введённый в настройках, либо из переменной окружения
 * машины распознавания. Второй путь удобен, когда ключ уже живёт на машине и
 * дублировать его в плагине не хочется.
 */
function resolveKey(ownKey: string, envName: string): string {
  if (envName.length > 0) {
    return process.env[envName] ?? "";
  }
  return ownKey;
}

async function readConfig(paths: HostPaths): Promise<VoiceConfig> {
  let stored: unknown;
  try {
    stored = JSON.parse(await readFile(paths.configFile, "utf8"));
  } catch {
    return DEFAULT_CONFIG;
  }
  const exact = voiceConfigSchema.safeParse(stored);
  if (exact.success) {
    return exact.data;
  }
  const partial = voiceConfigSchema.partial().safeParse(stored);
  if (!partial.success) {
    return DEFAULT_CONFIG;
  }
  return {
    ...DEFAULT_CONFIG,
    ...partial.data,
    cloud: { ...DEFAULT_CONFIG.cloud, ...(partial.data.cloud ?? {}) },
    aiPass: { ...DEFAULT_CONFIG.aiPass, ...(partial.data.aiPass ?? {}) },
  };
}

/** Конфиг вместе с ключами, которые хранятся отдельно от остальных настроек. */
async function readConfigWithSecrets(paths: HostPaths): Promise<VoiceConfig> {
  const config = await readConfig(paths);
  const secrets = await readSecrets(paths);
  return {
    ...config,
    cloud: {
      ...config.cloud,
      openaiApiKey: secrets.openaiApiKey ?? config.cloud.openaiApiKey,
      googleApiKey: secrets.googleApiKey ?? config.cloud.googleApiKey,
      groqApiKey: secrets.groqApiKey ?? config.cloud.groqApiKey,
    },
    aiPass: { ...config.aiPass, apiKey: secrets.aiPassApiKey ?? config.aiPass.apiKey },
  };
}

/** Обратное отображение: облачным службам нужен тип содержимого файла. */
function mimeTypeFor(path: string): string {
  const extension = extname(path).toLowerCase();
  for (const [mime, known] of Object.entries(EXTENSION_BY_MIME)) {
    if (known === extension) {
      return mime;
    }
  }
  return "application/octet-stream";
}

function extensionFor(filename: string, mimeType: string): string {
  const fromName = /\.[a-z0-9]{2,5}$/iu.exec(filename)?.[0];
  return fromName ?? EXTENSION_BY_MIME[mimeType.split(";")[0]?.trim() ?? ""] ?? ".webm";
}

/**
 * Сохранить исходную запись. Это не архив ради архива: пока запись на диске,
 * сбой распознавания или редактуры не уничтожает сказанное.
 */
async function keepRecording(paths: HostPaths, audio: Buffer, extension: string): Promise<string | null> {
  const day = new Date().toISOString().slice(0, 10);
  const directory = join(paths.recordingsDir, day);
  try {
    await mkdir(directory, { recursive: true });
    const name = `${new Date().toISOString().slice(11, 19).replace(/:/gu, "")}-${randomUUID().slice(0, 6)}${extension}`;
    const target = join(directory, name);
    await writeFile(target, audio);
    return target;
  } catch {
    return null;
  }
}

async function pruneRecordings(paths: HostPaths, keepDays: number): Promise<void> {
  if (keepDays <= 0 || !existsSync(paths.recordingsDir)) {
    return;
  }
  const threshold = Date.now() - keepDays * 24 * 60 * 60 * 1000;
  for (const entry of await readdir(paths.recordingsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) {
      continue;
    }
    const day = Date.parse(entry.name);
    if (Number.isFinite(day) && day < threshold) {
      await rm(join(paths.recordingsDir, entry.name), { recursive: true, force: true });
    }
  }
}

/** Правила словаря, очистка и — если включён — проход модели. */
async function postprocess(
  rawText: string,
  config: VoiceConfig,
  log: (message: string) => void,
  deadline: number,
): Promise<string> {
  const entries = parseVocabulary(config.vocabulary);
  let text = rawText;

  // Искажения ищутся сравнением по согласному костяку: непохожее не трогается.
  if (config.vocabularyRepair && entries.length > 0) {
    text = repairTerms(text, entries);
  }
  text = cleanupText(text, {
    fillers: config.cleanupFillers,
    quotes: config.cleanupQuotes,
    language: config.language,
  });

  const aiPassKey = resolveAiPassKey(config).key;
  if (config.aiPass.enabled && config.aiPass.model && aiPassKey) {
    try {
      const corrected = await runAiPass(text, {
        baseUrl: config.aiPass.baseUrl,
        model: config.aiPass.model,
        apiKey: aiPassKey,
        vocabulary: entries,
        timeoutMs: remainingMs(deadline),
      });
      // Модель иногда переписывает обычные слова вопреки промпту.
      text = guardAiPass(text, corrected, entries);
    } catch (error) {
      // Текст после правил уже пригоден — сбой прохода не должен его отменять.
      log(`ИИ-проход пропущен: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return text;
}

async function appendHistory(
  paths: HostPaths,
  record: {
    id: string;
    createdAt: string;
    engine: string;
    durationMs: number;
    audioPath: string | null;
    rawText: string;
    text: string;
  },
): Promise<void> {
  try {
    await appendFile(paths.historyFile, `${JSON.stringify(record)}\n`, "utf8");
    if (record.audioPath) {
      await writeFile(`${record.audioPath}.txt`, record.text, "utf8");
    }
  } catch {
    // История — удобство, а не условие выдачи текста.
  }
}


type TranscriptionOutcome = {
  engine: string;
  durationMs: number;
  rawText: string;
  text: string;
};

/**
 * Распознать файл, лежащий на этой машине, и привести текст в порядок.
 * Общий путь для голосового ввода BB и для проверки на готовом файле.
 */
async function runTranscription(
  paths: HostPaths,
  config: VoiceConfig,
  args: { audioPath: string; savedPath: string | null; prompt: string | null; timeoutMs: number },
): Promise<TranscriptionOutcome> {
  const started = Date.now();
  const deadline = started + args.timeoutMs;
  let rawText: string;

  const entries = parseVocabulary(config.vocabulary);
  const hint = config.vocabularyPrompt
    ? (buildRecognitionPrompt(entries, PROMPT_BUDGET_CHARS) ?? args.prompt)
    : args.prompt;

  if (
    config.engine === "openai" ||
    config.engine === "google" ||
    config.engine === "groq"
  ) {
    let transcribe: (request: CloudRequest) => Promise<string>;
    if (config.engine === "openai" || config.engine === "groq") {
      const groq = config.engine === "groq";
      const key = groq
        ? resolveKey(resolveProviderKey(config, "groq").key, config.cloud.groqKeyEnv)
        : resolveKey(resolveProviderKey(config, "openai").key, config.cloud.openaiKeyEnv);
      const keyEnv = groq ? config.cloud.groqKeyEnv : config.cloud.openaiKeyEnv;
      if (!key) {
        throw new Error(
          keyEnv
            ? `Переменной ${keyEnv} нет на машине распознавания.`
            : `Для движка ${groq ? "Groq" : "OpenAI"} нужен ключ: Настройки → Голосовой ввод → Облако.`,
        );
      }
      transcribe = (request) => transcribeOpenAi(
        {
          baseUrl: groq ? config.cloud.groqBaseUrl : config.cloud.openaiBaseUrl,
          apiKey: key,
          model: groq ? config.cloud.groqModel : config.cloud.openaiModel,
        },
        request,
      );
    } else {
      const key = resolveKey(resolveProviderKey(config, "google").key, config.cloud.googleKeyEnv);
      if (!key) {
        throw new Error(
          config.cloud.googleKeyEnv
            ? `Переменной ${config.cloud.googleKeyEnv} нет на машине распознавания.`
            : "Для движка Google нужен ключ: Настройки → Голосовой ввод → Облако.",
        );
      }
      transcribe = (request) => transcribeGemini(
        {
          baseUrl: config.cloud.googleBaseUrl,
          apiKey: key,
          model: config.cloud.googleModel,
        },
        request,
      );
    }
    rawText = await transcribeCloudAudio(args.audioPath, await findFfmpeg(), deadline, hint,
      async (part, timeoutMs) => transcribe({
        audio: await readFile(part.path),
        filename: basename(part.path),
        mimeType: part.converted ? "audio/wav" : mimeTypeFor(part.path),
        language: config.language,
        prompt: hint,
        timeoutMs: Math.min(timeoutMs, remainingMs(deadline)),
      }));
  } else {
    await ensureDaemon(paths, config.engine, DAEMON_IDLE_SECONDS);
    const response = await callDaemon(
      paths,
      config.engine,
      {
        op: "transcribe",
        engine: config.engine,
        model: config.whisperModel,
        language: config.language,
        audioPath: args.audioPath,
        prompt: hint,
      },
      args.timeoutMs,
    );
    if (!response.ok) {
      throw new Error(response.message);
    }
    rawText = response.text ?? "";
  }

  rawText = suppressPromptEcho(rawText, hint);
  const text = await postprocess(rawText, config, () => {}, deadline);
  if (!text.trim()) {
    throw new Error("Речь не обнаружена (тишина на записи)");
  }
  const outcome: TranscriptionOutcome = {
    engine: config.engine,
    durationMs: Date.now() - started,
    rawText,
    text,
  };

  await appendHistory(paths, {
    id: randomUUID(),
    createdAt: new Date().toISOString(),
    engine: outcome.engine,
    durationMs: outcome.durationMs,
    audioPath: args.savedPath,
    rawText,
    text,
  });
  await pruneRecordings(paths, config.keepDays);
  return outcome;
}

async function executeAudioTranscription(
  paths: HostPaths,
  input: {
    audioBase64: string;
    mimeType: string;
    filename?: string;
    prompt?: string | null;
    timeoutMs?: number;
  },
  tempDir: string,
): Promise<{ ok: true; model: string; text: string } | { ok: false; code: "timeout" | "request_failed"; message: string }> {
  const config = await readConfigWithSecrets(paths);
  const audio = Buffer.from(input.audioBase64, "base64");
  const extension = extensionFor(input.filename ?? "", input.mimeType);

  const savedPath = config.saveRecordings
    ? await keepRecording(paths, audio, extension)
    : null;

  let audioPath = savedPath;
  if (!audioPath) {
    await mkdir(tempDir, { recursive: true });
    audioPath = join(tempDir, `${randomUUID()}${extension}`);
    await writeFile(audioPath, audio);
  }

  try {
    const outcome = await runTranscription(paths, config, {
      audioPath,
      savedPath,
      prompt: input.prompt ?? null,
      timeoutMs: input.timeoutMs ?? TRANSCRIBE_TIMEOUT_MS,
    });
    return { ok: true as const, model: outcome.engine, text: outcome.text };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false as const,
      code: message === "timeout" ? ("timeout" as const) : ("request_failed" as const),
      message:
        message === "timeout" ? "Распознавание не уложилось в отведённое время" : message,
    };
  }
}

export default experimental_defineHostEntry({
  contract: hostContract,
  experimental_signals: hostSignals,
  handlers: {
    identify: async () => ({ hostname: hostname(), platform: platform() }),

    setSecret: async ({ name, value }, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      await mkdir(paths.dataDir, { recursive: true });
      await writeSecret(paths, name, value);
      return { saved: true as const };
    },

    secretsStatus: async (_input, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      const config = await readConfigWithSecrets(paths);
      return {
        openaiApiKey: resolveProviderKey(config, "openai").key.length > 0,
        googleApiKey: resolveProviderKey(config, "google").key.length > 0,
        groqApiKey: resolveProviderKey(config, "groq").key.length > 0,
        aiPassApiKey: resolveAiPassKey(config).key.length > 0,
        aiPassKeySource: resolveAiPassKey(config).source,
      };
    },

    envKeys: async () => {
      // Только имена и только похожие на ключи: значения наружу не отдаём.
      const names = Object.keys(process.env)
        .filter((name) => /(?:KEY|TOKEN|SECRET|API)$/u.test(name) && (process.env[name] ?? "").length > 0)
        .sort();
      return { names };
    },

    applyConfig: async ({ config }, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      await mkdir(paths.dataDir, { recursive: true });
      const previous = await readFile(paths.configFile, "utf8")
        .then((raw) => JSON.parse(raw) as Partial<VoiceConfig>)
        .catch(() => null);
      await writeFile(paths.configFile, JSON.stringify(config, null, 2), "utf8");
      // Сменили движок или модель Whisper — прежняя модель больше не нужна в памяти.
      for (const engine of LOCAL_ENGINES) {
        const stillUsed =
          engine === config.engine &&
          (engine !== "whisper" || previous?.whisperModel === config.whisperModel);
        if (!stillUsed) {
          await stopDaemon(paths, engine);
        }
      }
      return { applied: true as const };
    },

    setup: async ({ engine, reinstall }, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      // Пакеты движка меняются под демоном — его нужно остановить.
      await stopDaemon(paths, engine);
      return await setupEngine(paths, engine, reinstall);
    },

    status: async (_input, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      const config = await readConfigWithSecrets(paths);
      const engines: { engine: EngineId; ready: boolean; detail: string; modelBytes: number }[] = [];
      const cloudKeys = {
        openai: resolveKey(resolveProviderKey(config, "openai").key, config.cloud.openaiKeyEnv).length > 0,
        google: resolveKey(resolveProviderKey(config, "google").key, config.cloud.googleKeyEnv).length > 0,
        groq: resolveKey(resolveProviderKey(config, "groq").key, config.cloud.groqKeyEnv).length > 0,
      };
      for (const engine of ["whisper", "gigaam", "openai", "google", "groq"] as const) {
        const detail = await engineReady(paths, engine, cloudKeys);
        engines.push({
          engine,
          ready: detail === "ready",
          detail,
          modelBytes: await modelSize(engine, config.whisperModel),
        });
      }
      let recordings = 0;
      if (existsSync(paths.recordingsDir)) {
        for (const day of await readdir(paths.recordingsDir)) {
          const directory = join(paths.recordingsDir, day);
          if ((await stat(directory)).isDirectory()) {
            recordings += (await readdir(directory)).filter((name) => !name.endsWith(".txt")).length;
          }
        }
      }
      return {
        pythonPath: await findPython(),
        ffmpegPath: await findFfmpeg(),
        dataDir: paths.dataDir,
        engines,
        recordings,
      };
    },

    historyList: async ({ limit }, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      if (!existsSync(paths.historyFile)) {
        return { entries: [] };
      }
      const lines = (await readFile(paths.historyFile, "utf8")).trim().split("\n").filter(Boolean);
      const entries = lines
        .slice(-limit)
        .reverse()
        .flatMap((line) => {
          try {
            return [JSON.parse(line)];
          } catch {
            return [];
          }
        });
      return { entries };
    },

    // Контракт SDK описывает оба вида сервиса, но плагин заявлен только как
    // голосовой: текстовую генерацию он честно отклоняет.
    "ai.inference.complete": async () => ({
      ok: false as const,
      code: "request_failed" as const,
      message: "Плагин «Голосовой ввод» обслуживает только транскрипцию.",
    }),

    "ai.voice.transcribe": async (input, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      return await executeAudioTranscription(paths, input, context.experimental_paths.tempDir);
    },

    transcribeDirect: async (input, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      const res = await executeAudioTranscription(paths, input, context.experimental_paths.tempDir);
      if (res.ok) {
        return { ok: true as const, text: res.text, error: null };
      }
      return { ok: false as const, text: "", error: res.message };
    },

    downloadModel: async ({ engine }, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      const config = await readConfig(paths);
      if (engine === "openai" || engine === "google" || engine === "groq") {
        return { ready: true, detail: "Облачный движок моделей не хранит." };
      }
      // Скрипт загрузки живёт в бандле хоста: раскладываем его рядом с окружением.
      await materializePython(paths);
      // Пока идёт загрузка, воркер не должен уснуть по таймеру простоя.
      const lease = context.experimental_retainWorker();
      try {
        await downloadModel(paths, engine, engine === "whisper" ? config.whisperModel : "", (event) => {
          void context.experimental_emitSignal("modelProgress", {
            engine,
            percent: event.event === "progress" ? event.percent : event.event === "done" ? 100 : null,
            downloadedBytes: event.event === "progress" ? event.downloadedBytes : null,
            totalBytes: event.event === "progress" ? event.totalBytes : null,
            state: event.event === "progress" ? "downloading" : event.event === "done" ? "done" : "error",
            message: event.event === "error" ? event.message : null,
          });
        });
      } finally {
        lease.dispose();
      }
      return { ready: true, detail: `Модель движка «${engine}» готова.` };
    },

    listModels: async (_input, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      const config = await readConfig(paths);
      const models = await listDiskModels();
      return {
        models: models.map((item) => ({
          ...item,
          active:
            (item.engine === "whisper" && item.model === config.whisperModel) ||
            (item.engine === "gigaam" && config.engine === "gigaam"),
        })),
      };
    },

    deleteModel: async ({ engine, model }, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      // Модель может быть загружена в память демона — иначе место не вернётся.
      await stopDaemon(paths, engine);
      return { freedBytes: await removeModel(engine, model) };
    },

    transcribeFile: async ({ path, timeoutMs }, context) => {
      const paths = hostPaths(context.experimental_paths.dataDir);
      const config = await readConfigWithSecrets(paths);
      if (!existsSync(path)) {
        throw new Error(`Файл не найден на машине распознавания: ${path}`);
      }
      return await runTranscription(paths, config, {
        audioPath: path,
        savedPath: null,
        prompt: null,
        timeoutMs,
      });
    },
  },
  dispose: async () => {
    // Демон намеренно переживает воркер: следующая запись не должна ждать модель.
  },
});
