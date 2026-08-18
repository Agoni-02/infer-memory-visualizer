import type { KvCacheProfile } from "./models";

export type CachePrecision = "bf16_fp16" | "fp8_int8" | "fp4_int4";

export const CACHE_PRECISIONS: Record<CachePrecision, { label: string; bytes: number }> = {
  bf16_fp16: { label: "BF16 / FP16", bytes: 2 },
  fp8_int8: { label: "FP8 / INT8", bytes: 1 },
  fp4_int4: { label: "FP4 / INT4", bytes: 0.5 },
};

type KvCacheInput = {
  profile: KvCacheProfile;
  tokens: number;
  sequences: number;
  kvPrecision: CachePrecision;
  indexPrecision: CachePrecision;
  mtpLayers?: number;
};

export type KvCacheBreakdown = {
  kvCache: number;
  indexCache: number;
  total: number;
  kvBytesPerElement: number;
  indexBytesPerElement: number;
  kvCopies: number;
  effectiveKvLayers: number;
  effectiveIndexLayers: number;
};

export function calculateKvCache({
  profile,
  tokens,
  sequences,
  kvPrecision,
  indexPrecision,
  mtpLayers = 0,
}: KvCacheInput): KvCacheBreakdown {
  const safeTokens = Math.max(0, Math.floor(tokens));
  const safeSequences = Math.max(0, Math.floor(sequences));
  const safeMtpLayers = Math.max(0, Math.floor(mtpLayers));
  const effectiveKvLayers = profile.layers + safeMtpLayers;
  const effectiveIndexLayers = profile.sparseLayers + safeMtpLayers;
  const kvBytesPerElement = (CACHE_PRECISIONS[kvPrecision] ?? CACHE_PRECISIONS.bf16_fp16).bytes;
  const indexBytesPerElement = (CACHE_PRECISIONS[indexPrecision] ?? CACHE_PRECISIONS.fp4_int4).bytes;
  // Standard GQA stores K and V separately; some architectures share one K/V tensor.
  const kvCopies = profile.sharedKv ? 1 : 2;
  const kvCache = safeTokens * safeSequences * effectiveKvLayers * kvCopies
    * profile.kvHeads * profile.headDim * kvBytesPerElement;
  // The MSA index cache stores one key-only vector per sparse layer and token.
  const indexCache = safeTokens * safeSequences * effectiveIndexLayers
    * profile.indexHeadDim * indexBytesPerElement;

  return {
    kvCache,
    indexCache,
    total: kvCache + indexCache,
    kvBytesPerElement,
    indexBytesPerElement,
    kvCopies,
    effectiveKvLayers,
    effectiveIndexLayers,
  };
}
