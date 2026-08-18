import assert from "node:assert/strict";
import test from "node:test";

import { calculateMiniMaxWeight, mxfp8MatrixMemory, mxMatrixMemory } from "../app/weight-model.ts";
import { MODELS } from "../app/models.ts";

const profile = {
  vocabSize: 200064,
  totalLayers: 60,
  denseLayers: 3,
  moeLayers: 57,
  expertIntermediateSize: 3072,
  denseIntermediateSize: 12288,
  attentionHeads: 64,
  kvHeads: 4,
  headDim: 128,
  indexerHeads: 4,
  indexerHeadDim: 128,
  sharedExperts: 1,
  vocabPaddingSize: 64,
};

test("MXFP8 matrix memory adds one byte per [1, 32] block", () => {
  assert.deepEqual(mxfp8MatrixMemory(3, 33), {
    payload: 99,
    scales: 6,
    total: 105,
  });
});

test("TP4/DP8 shards routed, attention, dense and shared scales correctly", () => {
  const common = {
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 4,
    epSize: 32,
  };
  const replicatedShared = calculateMiniMaxWeight({
    ...common,
    sharedExpertTpSize: 1,
  });
  const shardedShared = calculateMiniMaxWeight({
    ...common,
    sharedExpertTpSize: 4,
  });

  assert.equal(replicatedShared.routedExpertScales, 403_439_616);
  assert.equal(replicatedShared.attentionScales, 52_936_704);
  assert.equal(replicatedShared.denseMlpScales, 5_308_416);
  assert.equal(replicatedShared.sharedExpertScales, 100_859_904);
  assert.equal(shardedShared.sharedExpertScales, 25_214_976);
  assert.equal(replicatedShared.sharedExpertScales / shardedShared.sharedExpertScales, 4);
  assert.equal(replicatedShared.total, 19_975_583_488);
  assert.equal(shardedShared.total, 17_479_300_864);
  assert.ok(Math.abs(replicatedShared.total / 1024 ** 3 - 18.60371) < 0.00001);
  assert.ok(Math.abs(shardedShared.total / 1024 ** 3 - 16.27887) < 0.00001);
});

test("index K stays replicated while index Q follows TP", () => {
  const tp1 = calculateMiniMaxWeight({
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 1,
    epSize: 2,
    sharedExpertTpSize: 1,
  });
  const tp2 = calculateMiniMaxWeight({
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 2,
    epSize: 4,
    sharedExpertTpSize: 1,
  });

  const mainAttentionScaleTp1 = 60 * 3_342_336;
  const mainAttentionScaleTp2 = 60 * 1_671_168;
  const indexScaleTp1 = tp1.attentionScales - mainAttentionScaleTp1;
  const indexScaleTp2 = tp2.attentionScales - mainAttentionScaleTp2;
  assert.equal(indexScaleTp1, 57 * (98_304 + 24_576));
  assert.equal(indexScaleTp2, 57 * (49_152 + 24_576));
});

test("MXFP4 packs two weights per byte and keeps one scale per 32 values", () => {
  assert.deepEqual(mxMatrixMemory(3, 33, "mxfp4"), {
    payload: 50,
    scales: 6,
    total: 56,
  });
});

test("MiniMax M3 full-model and TP4/DP8/EP32 per-device weights", () => {
  const m3 = MODELS.find((model) => model.id === "minimax-m3");
  assert.ok(m3?.weightProfile);
  assert.equal(m3.weightProfile.totalLayers, 60);
  assert.equal(m3.weightProfile.denseLayers, 3);
  assert.equal(m3.weightProfile.moeLayers, 57);
  assert.equal(m3.weightProfile.attentionHeads, 64);
  assert.equal(m3.weightProfile.kvHeads, 4);
  assert.equal(m3.weightProfile.indexerHeads, 4);
  assert.equal(m3.weightProfile.indexerHeadDim, 128);
  assert.equal(m3.weightProfile.moeWeightFormat, "mxfp8");
  assert.equal(m3.weightProfile.qkvProjection, "qkvo");
  const common = {
    profile: m3.weightProfile,
    hiddenSize: m3.hiddenSize,
    expertCount: m3.expertCount,
    sharedExpertTpSize: 1,
  };
  const full = calculateMiniMaxWeight({ ...common, tpSize: 1, epSize: 1 });
  const tp4Ep32 = calculateMiniMaxWeight({ ...common, tpSize: 4, epSize: 32 });

  assert.equal(full.total, 442_009_474_816);
  assert.equal(tp4Ep32.total, 19_975_583_488);
  assert.equal(tp4Ep32.routedExpertPayload, 12_910_067_712);
  assert.equal(tp4Ep32.routedExpertScales, 403_439_616);
  assert.equal(tp4Ep32.sharedExpertPayload, 3_227_516_928);
  assert.equal(tp4Ep32.sharedExpertScales, 100_859_904);
  assert.equal(
    tp4Ep32.attentionQkv,
    60 * (
      mxfp8MatrixMemory(64 / 4 * 128, m3.hiddenSize).total
      + 2 * mxfp8MatrixMemory(4 / 4 * 128, m3.hiddenSize).total
    ),
  );
  assert.equal(
    tp4Ep32.attentionOproj,
    60 * mxfp8MatrixMemory(m3.hiddenSize, 64 / 4 * 128).total,
  );
  assert.equal(
    tp4Ep32.attentionIndexer,
    57 * (
      mxfp8MatrixMemory(4 / 4 * 128, m3.hiddenSize).total
      + mxfp8MatrixMemory(128, m3.hiddenSize).total
    ),
  );
});

test("O-proj, Embedding and LM Head use independent TP sizes", () => {
  const m3 = MODELS.find((model) => model.id === "minimax-m3");
  assert.ok(m3?.weightProfile);
  const common = {
    profile: m3.weightProfile,
    hiddenSize: m3.hiddenSize,
    expertCount: m3.expertCount,
    tpSize: 4,
    epSize: 32,
    sharedExpertTpSize: 1,
  };
  const baseline = calculateMiniMaxWeight(common);
  const independentlySharded = calculateMiniMaxWeight({
    ...common,
    oprojTpSize: 8,
    embeddingTpSize: 8,
    lmHeadTpSize: 16,
  });

  assert.equal(independentlySharded.routedExpertPayload, baseline.routedExpertPayload);
  assert.equal(independentlySharded.attentionQkv, baseline.attentionQkv);
  assert.equal(independentlySharded.attentionIndexer, baseline.attentionIndexer);
  assert.equal(independentlySharded.attentionOproj, baseline.attentionOproj / 2);
  assert.ok(independentlySharded.attention < baseline.attention);
  assert.equal(independentlySharded.embedding, baseline.embedding / 2);
  assert.equal(independentlySharded.lmHead, baseline.lmHead / 4);
});

test("Shared Expert uses an independent configurable TP size", () => {
  const m3 = MODELS.find((model) => model.id === "minimax-m3");
  assert.ok(m3?.weightProfile);
  const common = {
    profile: m3.weightProfile,
    hiddenSize: m3.hiddenSize,
    expertCount: m3.expertCount,
    tpSize: 4,
    epSize: 32,
  };
  const tp1 = calculateMiniMaxWeight({ ...common, sharedExpertTpSize: 1 });
  const tp4 = calculateMiniMaxWeight({ ...common, sharedExpertTpSize: 4 });

  assert.equal(tp4.sharedExpertPayload, tp1.sharedExpertPayload / 4);
  assert.equal(tp4.sharedExpertScales, tp1.sharedExpertScales / 4);
  assert.equal(tp4.routedExpertPayload, tp1.routedExpertPayload);
  assert.equal(tp4.attention, tp1.attention);
});

test("QKV and Indexer Q share an independent Attention TP size", () => {
  const m3 = MODELS.find((model) => model.id === "minimax-m3");
  assert.ok(m3?.weightProfile);
  const common = {
    profile: m3.weightProfile,
    hiddenSize: m3.hiddenSize,
    expertCount: m3.expertCount,
    tpSize: 4,
    epSize: 32,
    sharedExpertTpSize: 1,
  };
  const tp2 = calculateMiniMaxWeight({ ...common, attentionTpSize: 2 });
  const tp4 = calculateMiniMaxWeight({ ...common, attentionTpSize: 4 });
  const qkvPerLayer = mxfp8MatrixMemory(64 / 4 * 128, m3.hiddenSize).total
    + 2 * mxfp8MatrixMemory(4 / 4 * 128, m3.hiddenSize).total;
  const indexerPerLayer = mxfp8MatrixMemory(4 / 4 * 128, m3.hiddenSize).total
    + mxfp8MatrixMemory(128, m3.hiddenSize).total;

  assert.equal(tp4.attentionQkv, 60 * qkvPerLayer);
  assert.equal(tp4.attentionIndexer, 57 * indexerPerLayer);
  assert.equal(tp4.attentionQkv, tp2.attentionQkv / 2);
  assert.equal(tp4.attentionOproj, tp2.attentionOproj);
  assert.ok(tp4.attentionIndexer < tp2.attentionIndexer);
});

test("MTP layer weight scales linearly", () => {
  const m3 = MODELS.find((model) => model.id === "minimax-m3");
  assert.ok(m3?.weightProfile);
  const common = {
    profile: m3.weightProfile,
    hiddenSize: m3.hiddenSize,
    expertCount: m3.expertCount,
    tpSize: 4,
    epSize: 32,
    oprojTpSize: 4,
    embeddingTpSize: 4,
    lmHeadTpSize: 4,
    sharedExpertTpSize: 1,
  };
  const noMtp = calculateMiniMaxWeight({ ...common, mtpLayers: 0 });
  const oneMtp = calculateMiniMaxWeight({ ...common, mtpLayers: 1 });
  const threeMtp = calculateMiniMaxWeight({ ...common, mtpLayers: 3 });

  assert.equal(noMtp.mtpWeight, 0);
  assert.equal(oneMtp.mtpWeight, oneMtp.mtpPerLayer);
  assert.equal(threeMtp.mtpWeight, 3 * oneMtp.mtpWeight);
  assert.equal(threeMtp.total - noMtp.total, 3 * (oneMtp.total - noMtp.total));
});
