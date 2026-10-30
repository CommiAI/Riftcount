/**
 * Standalone OpenRift board read against the two research stills.
 * Uses public scan bank/models under /tmp/openrift-scan and on-demand
 * https://openrift.app/media/cards/{uuid}-400w.webp references.
 */
import fs from "node:fs";
import path from "node:path";

import * as ort from "onnxruntime-node";
import sharp from "sharp";

import {
  BOARD_DETECTOR_SIZE,
  boardDetectorCandidates,
  fillBoardDetectorInput,
} from "/tmp/openrift-src/packages/shared/src/scan/board-detector.ts";
import { boardOptionsFor, readBoard } from "/tmp/openrift-src/packages/shared/src/scan/board.ts";
import { decodeEmbedBank } from "/tmp/openrift-src/packages/shared/src/scan/embed-format.ts";
import {
  normalizeEmbeddings,
  preprocessCardInto,
  type CardEmbedder,
  type EmbedBank,
} from "/tmp/openrift-src/packages/shared/src/scan/embed.ts";
import { gatesForEmbedDim } from "/tmp/openrift-src/packages/shared/src/scan/session-options.ts";
import type { RgbaImage } from "/tmp/openrift-src/packages/shared/src/scan/types.ts";

const ROOT = path.resolve(import.meta.dirname, "../..");
const ASSETS = process.env.OPENRIFT_SCAN_DIR ?? "/tmp/openrift-scan";
const BANK_FILE = path.join(ASSETS, "scan-bank-50d08713bd8b8fa4.bin");
const LABELS_FILE = path.join(ASSETS, "scan-labels-50d08713bd8b8fa4.json");
const ENCODER_FILE = path.join(ASSETS, "scan-encoder-v2.onnx");
const BOARD_FILE = path.join(ASSETS, "scan-board-detector-v2.onnx");
const REF_CACHE = process.env.OPENRIFT_REF_CACHE ?? "/tmp/or-spike/ref-cache";
const EMBED_SIZE = 192;
const MEDIA_BASE = "https://openrift.app/media/cards";

const PHOTOS = [
  {
    id: "photo1-runes",
    file: path.join(ROOT, "examples/photos/photo1-runes.jpg"),
  },
  {
    id: "photo2-fabric",
    file: path.join(ROOT, "examples/photos/photo2-fabric.jpg"),
  },
];

type Label = {
  name: string;
  code: string;
  language: string;
  type: string;
  markers: string;
};

async function loadImage(
  file: string,
  options: { maxSide?: number; rotate?: boolean } = {},
): Promise<RgbaImage> {
  let pipeline = sharp(file);
  if (options.rotate) {
    pipeline = pipeline.rotate();
  }
  pipeline = pipeline.flatten({ background: { r: 128, g: 128, b: 128 } });
  if (options.maxSide) {
    pipeline = pipeline.resize({
      width: options.maxSide,
      height: options.maxSide,
      fit: "inside",
      withoutEnlargement: true,
    });
  }
  const { data, info } = await pipeline.raw().toColourspace("srgb").ensureAlpha().toBuffer({
    resolveWithObject: true,
  });
  return {
    data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
    width: info.width,
    height: info.height,
  };
}

async function loadImageFromBuffer(buf: Buffer): Promise<RgbaImage> {
  const { data, info } = await sharp(buf)
    .flatten({ background: { r: 128, g: 128, b: 128 } })
    .raw()
    .toColourspace("srgb")
    .ensureAlpha()
    .toBuffer({ resolveWithObject: true });
  return {
    data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
    width: info.width,
    height: info.height,
  };
}

function createBoardDetector(modelFile: string): (frame: RgbaImage) => Promise<
  ReturnType<typeof boardDetectorCandidates>
> {
  const sessionPromise = ort.InferenceSession.create(modelFile);
  const input = new Float32Array(3 * BOARD_DETECTOR_SIZE * BOARD_DETECTOR_SIZE);
  return async (frame) => {
    const letterbox = fillBoardDetectorInput(frame, input);
    const session = await sessionPromise;
    const output = await session.run({
      image: new ort.Tensor("float32", input, [1, 3, BOARD_DETECTOR_SIZE, BOARD_DETECTOR_SIZE]),
    });
    const center = output.center?.data;
    const corner = output.corner?.data;
    const offsets = output.offsets?.data;
    if (
      !(center instanceof Float32Array) ||
      !(corner instanceof Float32Array) ||
      !(offsets instanceof Float32Array)
    ) {
      throw new TypeError("board detector missing float outputs");
    }
    return boardDetectorCandidates(frame, letterbox, center, corner, offsets);
  };
}

function createEmbedder(modelFile: string): CardEmbedder {
  const sessionPromise = ort.InferenceSession.create(modelFile);
  return async (pixels, count) => {
    const session = await sessionPromise;
    const slice = pixels.subarray(0, count * 3 * EMBED_SIZE * EMBED_SIZE);
    const output = await session.run({
      pixel_values: new ort.Tensor("float32", slice, [count, 3, EMBED_SIZE, EMBED_SIZE]),
    });
    return output.image_embeds.data as Float32Array;
  };
}

async function main(): Promise<void> {
  fs.mkdirSync(REF_CACHE, { recursive: true });
  const bankBuf = fs.readFileSync(BANK_FILE);
  const { bank, artKeys, canonical } = decodeEmbedBank(
    bankBuf.buffer.slice(bankBuf.byteOffset, bankBuf.byteOffset + bankBuf.byteLength),
  );
  const labels = JSON.parse(fs.readFileSync(LABELS_FILE, "utf-8")) as Record<string, Label>;
  const dim = bank.vectors.length / bank.keys.length;
  const gates = gatesForEmbedDim(dim);
  console.error(
    `bank keys=${bank.keys.length} dim=${dim} canonical=${canonical} gates=${JSON.stringify(gates)}`,
  );

  const detect = createBoardDetector(BOARD_FILE);
  const embedder = createEmbedder(ENCODER_FILE);
  const refMemo = new Map<string, Promise<RgbaImage | null>>();

  const fetchReference = (key: string): Promise<RgbaImage | null> => {
    const hit = refMemo.get(key);
    if (hit) {
      return hit;
    }
    const pending = (async () => {
      const cachePath = path.join(REF_CACHE, `${key}-400w.webp`);
      try {
        if (fs.existsSync(cachePath)) {
          return await loadImage(cachePath);
        }
        const url = `${MEDIA_BASE}/${key}-400w.webp`;
        const res = await fetch(url);
        if (!res.ok) {
          console.error(`ref miss ${key} status=${res.status}`);
          return null;
        }
        const buf = Buffer.from(await res.arrayBuffer());
        fs.writeFileSync(cachePath, buf);
        return await loadImageFromBuffer(buf);
      } catch (err) {
        console.error(`ref error ${key}`, err);
        return null;
      }
    })();
    refMemo.set(key, pending);
    return pending;
  };

  const deps = {
    embedder,
    bank: bank as EmbedBank,
    embedImageSize: EMBED_SIZE,
    artKeyOf: (key: string) => artKeys.get(key) ?? key,
    labelOf: (key: string) => {
      const label = labels[key];
      return label ? `${label.name} [${label.code}]` : key;
    },
    identityOf: (key: string) => {
      const label = labels[key];
      if (!label) {
        return undefined;
      }
      return {
        type: label.type,
        code: label.code,
        markers: label.markers || undefined,
        language: label.language,
      };
    },
    fetchReference,
  };

  const options = boardOptionsFor(gates, canonical);
  const maxSide = Number(process.env.MAX_SIDE ?? 2048);
  const results: unknown[] = [];

  for (const photo of PHOTOS) {
    const started = performance.now();
    console.error(`\n=== ${photo.id} ===`);
    const image = await loadImage(photo.file, { maxSide, rotate: true });
    console.error(`loaded ${image.width}x${image.height}`);
    const cards = await readBoard(image, detect, deps, options);
    const ms = performance.now() - started;
    const summary = cards.map((card) => {
      const label = labels[card.key];
      return {
        key: card.key,
        name: label?.name ?? "?",
        code: label?.code ?? "?",
        score: Number(card.score.toFixed(4)),
        rivalScore: Number(card.rivalScore.toFixed(4)),
        distance: Number(card.distance.toFixed(4)),
        confident: card.confident,
        printingResolved: card.printingResolved,
        alternatives: card.alternatives.map((alt) => {
          const altLabel = labels[alt];
          return {
            key: alt,
            name: altLabel?.name ?? "?",
            code: altLabel?.code ?? "?",
          };
        }),
      };
    });
    console.error(
      `cards=${cards.length} confident=${cards.filter((c) => c.confident).length} ${(ms / 1000).toFixed(1)}s`,
    );
    for (const row of summary) {
      const mark = row.confident ? "OK" : "??";
      console.error(
        `  [${mark}] ${row.name} ${row.code} score=${row.score} dist=${row.distance} rival=${row.rivalScore}`,
      );
    }
    results.push({
      photo: photo.id,
      file: photo.file,
      width: image.width,
      height: image.height,
      ms,
      cards: summary,
    });
  }

  const outPath = process.argv[2] ?? "/tmp/or-spike/results.json";
  fs.writeFileSync(
    outPath,
    `${JSON.stringify(
      {
        engine: "openrift-readBoard",
        bank: path.basename(BANK_FILE),
        encoder: path.basename(ENCODER_FILE),
        boardDetector: path.basename(BOARD_FILE),
        embedDim: dim,
        embedImageSize: EMBED_SIZE,
        maxSide,
        gates,
        canonical,
        results,
      },
      null,
      2,
    )}\n`,
  );
  console.error(`\nwrote ${outPath}`);
}

await main();
