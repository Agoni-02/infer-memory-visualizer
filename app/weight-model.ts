import type { MxWeightFormat, WeightProfile } from "./models";

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

export function mxMatrixMemory(
  out: number,
  input: number,
  format: MxWeightFormat,
): Mxfp8MatrixMemory {
  const payload = format === "mxfp4"
    ? Math.ceil(out * input / 2)
    : out * input;
  const scales = Math.ceil(out) * Math.ceil(input / 32);
  return { payload, scales, total: payload + scales };
}

export type MiniMaxWeightBreakdown = {
  routedExpertPayload: number;
  routedExpertScales: number;
  attentionQkvPayload: number;
  attentionQkvScales: number;
  attentionQkv: number;
  attentionOprojPayload: number;
  attentionOprojScales: number;
  attentionOproj: number;
  attentionIndexerPayload: number;
  attentionIndexerScales: number;
  attentionIndexer: number;
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
  mtpPerLayer: number;
  mtpWeight: number;
  misc: number;
  paddedVocab: number;
  total: number;
};

type MiniMaxWeightInput = {
  profile: WeightProfile;
  hiddenSize: number;
  expertCount: number;
  tpSize: number;
  epSize: number;
  attentionTpSize?: number;
  oprojTpSize?: number;
  embeddingTpSize?: number;
  lmHeadTpSize?: number;
  mtpLayers?: number;
  sharedExpertTpSize?: number;
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

export function calculateMiniMaxWeight({
  profile,
  hiddenSize: H,
  expertCount,
  tpSize: tp,
  epSize: ep,
  attentionTpSize,
  oprojTpSize,
  embeddingTpSize,
  lmHeadTpSize,
  mtpLayers = 0,
  sharedExpertTpSize = 1,
}: MiniMaxWeightInput): MiniMaxWeightBreakdown {
  const I = profile.expertIntermediateSize;
  const localExperts = expertCount / ep;
  const attentionTp = attentionTpSize ?? tp;
  const oprojTp = oprojTpSize ?? tp;
  const embeddingTp = embeddingTpSize ?? tp;
  const lmHeadTp = lmHeadTpSize ?? tp;
  const moeWeightFormat = profile.moeWeightFormat ?? "mxfp8";
  const sharedExpertWeightFormat = profile.sharedExpertWeightFormat
    ?? moeWeightFormat;

  const routedPerExpert = sumMatrices([
    mxMatrixMemory(2 * I, H, moeWeightFormat),
    mxMatrixMemory(H, I, moeWeightFormat),
  ]);
  const routedFactor = profile.moeLayers * localExperts;
  const routedExpertPayload = routedFactor * routedPerExpert.payload;
  const routedExpertScales = routedFactor * routedPerExpert.scales;

  const qRank = profile.attentionHeads / attentionTp * profile.headDim;
  const oRank = profile.attentionHeads / oprojTp
    * (profile.valueHeadDim ?? profile.headDim);
  const kvRank = (profile.sharedKv ? profile.kvHeads : profile.kvHeads / attentionTp)
    * profile.headDim;
  const indexQRank = profile.indexerHeads / attentionTp * profile.indexerHeadDim;
  // The index K projection produces one full index head and is replicated.
  const indexKRank = profile.indexerHeadDim;
  const qkvPerLayer = profile.qkvProjection === "qk"
    ? sumMatrices([
      mxfp8MatrixMemory(qRank + kvRank, H),
    ])
    : sumMatrices([
      mxfp8MatrixMemory(qRank, H),
      mxfp8MatrixMemory(kvRank, H),
      mxfp8MatrixMemory(kvRank, H),
    ]);
  const oprojPerLayer = mxfp8MatrixMemory(H, oRank);
  const indexerPerLayer = sumMatrices([
    mxfp8MatrixMemory(indexQRank, H),
    mxfp8MatrixMemory(indexKRank, H),
  ]);
  const attentionQkvPayload = profile.totalLayers * qkvPerLayer.payload;
  const attentionQkvScales = profile.totalLayers * qkvPerLayer.scales;
  const attentionQkv = attentionQkvPayload + attentionQkvScales;
  const attentionOprojPayload = profile.totalLayers * oprojPerLayer.payload;
  const attentionOprojScales = profile.totalLayers * oprojPerLayer.scales;
  const attentionOproj = attentionOprojPayload + attentionOprojScales;
  const sparseAttentionLayers = profile.sparseAttentionLayers ?? profile.moeLayers;
  const attentionIndexerPayload = sparseAttentionLayers * indexerPerLayer.payload;
  const attentionIndexerScales = sparseAttentionLayers * indexerPerLayer.scales;
  const attentionIndexer = attentionIndexerPayload + attentionIndexerScales;
  const attentionPayload = attentionQkvPayload + attentionOprojPayload
    + attentionIndexerPayload;
  const attentionScales = attentionQkvScales + attentionOprojScales
    + attentionIndexerScales;
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

  const sharedTp = sharedExpertTpSize;
  const sharedRank = I / sharedTp;
  const sharedPerExpert = sumMatrices([
    mxMatrixMemory(2 * sharedRank, H, sharedExpertWeightFormat),
    mxMatrixMemory(H, sharedRank, sharedExpertWeightFormat),
  ]);
  const sharedFactor = profile.moeLayers * profile.sharedExperts;
  const sharedExpertPayload = sharedFactor * sharedPerExpert.payload;
  const sharedExpertScales = sharedFactor * sharedPerExpert.scales;
  const sharedExperts = sharedExpertPayload + sharedExpertScales;

  const router = profile.moeLayers * expertCount * (H + 1) * 4;
  const norms = (4 * profile.totalLayers + 1) * H * 2;
  const paddedVocab = align(profile.vocabSize, profile.vocabPaddingSize);
  const embedding = paddedVocab / embeddingTp * H * 2;
  const lmHead = paddedVocab / lmHeadTp * H * 2;
  const mtpProjection = mxfp8MatrixMemory(H, 2 * H);
  const mtpNorms = 2 * H * 2;
  const mtpDecoderNorms = 4 * H * 2;
  const mtpPerLayer = routedPerExpert.total * localExperts
    + qkvPerLayer.total
    + oprojPerLayer.total
    + indexerPerLayer.total
    + sharedPerExpert.total * profile.sharedExperts
    + expertCount * (H + 1) * 4
    + mtpDecoderNorms
    + mtpProjection.total
    + mtpNorms;
  const mtpWeight = Math.max(0, Math.floor(mtpLayers)) * mtpPerLayer;
  const misc = 256;
  const total = routedExpertPayload + routedExpertScales + attention + denseMlp
    + sharedExperts + router + norms + embedding + lmHead + mtpWeight + misc;

  return {
    routedExpertPayload,
    routedExpertScales,
    attentionQkvPayload,
    attentionQkvScales,
    attentionQkv,
    attentionOprojPayload,
    attentionOprojScales,
    attentionOproj,
    attentionIndexerPayload,
    attentionIndexerScales,
    attentionIndexer,
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
    mtpPerLayer,
    mtpWeight,
    misc,
    paddedVocab,
    total,
  };
}
