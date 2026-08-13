import type { WeightProfile } from "./models";

export type Mxfp8MatrixMemory = {
  payload: number;
  scales: number;
  total: number;
};

export function mxfp8MatrixMemory(out: number, input: number): Mxfp8MatrixMemory {
  const payload = out * input;
  const scales = Math.ceil(out) * Math.ceil(input / 32);
  return { payload, scales, total: payload + scales };
}

export type MiniMaxM3WeightBreakdown = {
  routedExpertPayload: number;
  routedExpertScales: number;
  attentionPayload: number;
  attentionScales: number;
  attentionMetadata: number;
  attention: number;
  denseMlpPayload: number;
  denseMlpScales: number;
  denseMlp: number;
  sharedExpertPayload: number;
  sharedExpertScales: number;
  sharedExperts: number;
  router: number;
  norms: number;
  embedding: number;
  lmHead: number;
  misc: number;
  paddedVocab: number;
  total: number;
};

type MiniMaxM3WeightInput = {
  profile: WeightProfile;
  hiddenSize: number;
  expertCount: number;
  tpSize: number;
  epSize: number;
  enableSharedExpertTp: boolean;
};

const align = (value: number, boundary: number) =>
  Math.ceil(value / boundary) * boundary;

const sumMatrices = (matrices: Mxfp8MatrixMemory[]) => matrices.reduce(
  (sum, matrix) => ({
    payload: sum.payload + matrix.payload,
    scales: sum.scales + matrix.scales,
    total: sum.total + matrix.total,
  }),
  { payload: 0, scales: 0, total: 0 },
);

export function calculateMiniMaxM3Weight({
  profile,
  hiddenSize: H,
  expertCount,
  tpSize: tp,
  epSize: ep,
  enableSharedExpertTp,
}: MiniMaxM3WeightInput): MiniMaxM3WeightBreakdown {
  const I = profile.expertIntermediateSize;
  const localExperts = expertCount / ep;

  const routedPerExpert = sumMatrices([
    mxfp8MatrixMemory(2 * I, H),
    mxfp8MatrixMemory(H, I),
  ]);
  const routedFactor = profile.moeLayers * localExperts;
  const routedExpertPayload = routedFactor * routedPerExpert.payload;
  const routedExpertScales = routedFactor * routedPerExpert.scales;

  const qRank = profile.attentionHeads / tp * profile.headDim;
  const kvRank = profile.kvHeads / tp * profile.headDim;
  const indexQRank = profile.indexerHeads / tp * profile.indexerHeadDim;
  // The index K projection produces one full index head and is replicated.
  const indexKRank = profile.indexerHeadDim;
  const attentionPerLayer = sumMatrices([
    mxfp8MatrixMemory(qRank, H),
    mxfp8MatrixMemory(kvRank, H),
    mxfp8MatrixMemory(kvRank, H),
    mxfp8MatrixMemory(H, qRank),
  ]);
  const indexerPerLayer = sumMatrices([
    mxfp8MatrixMemory(indexQRank, H),
    mxfp8MatrixMemory(indexKRank, H),
  ]);
  const attentionPayload = profile.totalLayers * attentionPerLayer.payload
    + profile.moeLayers * indexerPerLayer.payload;
  const attentionScales = profile.totalLayers * attentionPerLayer.scales
    + profile.moeLayers * indexerPerLayer.scales;
  const attentionMetadata = 119_808;
  const attention = attentionPayload + attentionScales + attentionMetadata;

  const denseRank = profile.denseIntermediateSize / tp;
  const densePerLayer = sumMatrices([
    mxfp8MatrixMemory(2 * denseRank, H),
    mxfp8MatrixMemory(H, denseRank),
  ]);
  const denseMlpPayload = profile.denseLayers * densePerLayer.payload;
  const denseMlpScales = profile.denseLayers * densePerLayer.scales;
  const denseMlp = denseMlpPayload + denseMlpScales;

  const sharedTp = enableSharedExpertTp ? tp : 1;
  const sharedRank = I / sharedTp;
  const sharedPerExpert = sumMatrices([
    mxfp8MatrixMemory(2 * sharedRank, H),
    mxfp8MatrixMemory(H, sharedRank),
  ]);
  const sharedFactor = profile.moeLayers * profile.sharedExperts;
  const sharedExpertPayload = sharedFactor * sharedPerExpert.payload;
  const sharedExpertScales = sharedFactor * sharedPerExpert.scales;
  const sharedExperts = sharedExpertPayload + sharedExpertScales;

  const router = profile.moeLayers * expertCount * (H + 1) * 4;
  const norms = 2_961_408;
  const paddedVocab = align(profile.vocabSize, profile.vocabPaddingSize);
  const embedding = paddedVocab / tp * H * 2;
  const lmHead = embedding;
  const misc = 256;
  const total = routedExpertPayload + routedExpertScales + attention + denseMlp
    + sharedExperts + router + norms + embedding + lmHead + misc;

  return {
    routedExpertPayload,
    routedExpertScales,
    attentionPayload,
    attentionScales,
    attentionMetadata,
    attention,
    denseMlpPayload,
    denseMlpScales,
    denseMlp,
    sharedExpertPayload,
    sharedExpertScales,
    sharedExperts,
    router,
    norms,
    embedding,
    lmHead,
    misc,
    paddedVocab,
    total,
  };
}
