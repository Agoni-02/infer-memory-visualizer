import assert from "node:assert/strict";
import test from "node:test";

import { calculateMiniMaxM3Weight, mxfp8MatrixMemory } from "../app/weight-model.ts";

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
  const replicatedShared = calculateMiniMaxM3Weight({
    ...common,
    enableSharedExpertTp: false,
  });
  const shardedShared = calculateMiniMaxM3Weight({
    ...common,
    enableSharedExpertTp: true,
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
  const tp1 = calculateMiniMaxM3Weight({
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 1,
    epSize: 2,
    enableSharedExpertTp: false,
  });
  const tp2 = calculateMiniMaxM3Weight({
    profile,
    hiddenSize: 6144,
    expertCount: 128,
    tpSize: 2,
    epSize: 4,
    enableSharedExpertTp: false,
  });

  const mainAttentionScaleTp1 = 60 * 3_342_336;
  const mainAttentionScaleTp2 = 60 * 1_671_168;
  const indexScaleTp1 = tp1.attentionScales - mainAttentionScaleTp1;
  const indexScaleTp2 = tp2.attentionScales - mainAttentionScaleTp2;
  assert.equal(indexScaleTp1, 57 * (98_304 + 24_576));
  assert.equal(indexScaleTp2, 57 * (49_152 + 24_576));
});
