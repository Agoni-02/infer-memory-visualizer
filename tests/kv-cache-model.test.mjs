import assert from "node:assert/strict";
import test from "node:test";

import { calculateKvCache } from "../app/kv-cache-model.ts";
import { MODELS } from "../app/models.ts";

test("MiniMax M3 stores K and V separately and supports FP4 index cache", () => {
  const model = MODELS.find((item) => item.id === "minimax-m3");
  assert.ok(model?.kvCacheProfile);
  assert.equal(model.kvCacheProfile.layers, 60);
  assert.equal(model.kvCacheProfile.sparseLayers, 57);
  const result = calculateKvCache({
    profile: model.kvCacheProfile,
    tokens: 1024,
    sequences: 1,
    kvPrecision: "bf16_fp16",
    indexPrecision: "fp4_int4",
  });

  assert.equal(result.kvCopies, 2);
  assert.equal(result.kvCache, 125_829_120);
  assert.equal(result.indexCache, 3_735_552);
  assert.equal(result.total, 129_564_672);
});

test("KV and index precision can be selected independently", () => {
  const model = MODELS.find((item) => item.id === "minimax-m3");
  assert.ok(model?.kvCacheProfile);
  const bf16 = calculateKvCache({
    profile: model.kvCacheProfile,
    tokens: 4096,
    sequences: 8,
    kvPrecision: "bf16_fp16",
    indexPrecision: "bf16_fp16",
  });
  const compressed = calculateKvCache({
    profile: model.kvCacheProfile,
    tokens: 4096,
    sequences: 8,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp4_int4",
  });

  assert.equal(compressed.kvCache, bf16.kvCache / 2);
  assert.equal(compressed.indexCache, bf16.indexCache / 4);
});

test("each MTP layer adds one KV layer and one sparse Index layer", () => {
  const model = MODELS.find((item) => item.id === "minimax-m3");
  assert.ok(model?.kvCacheProfile);
  const common = {
    profile: model.kvCacheProfile,
    tokens: 1024,
    sequences: 1,
    kvPrecision: "fp8_int8",
    indexPrecision: "fp4_int4",
  };
  const baseline = calculateKvCache({ ...common, mtpLayers: 0 });
  const threeMtp = calculateKvCache({ ...common, mtpLayers: 3 });

  assert.equal(threeMtp.effectiveKvLayers, 63);
  assert.equal(threeMtp.effectiveIndexLayers, 60);
  assert.equal(
    threeMtp.kvCache - baseline.kvCache,
    3 * 1024 * 2 * model.kvCacheProfile.kvHeads * model.kvCacheProfile.headDim,
  );
  assert.equal(
    threeMtp.indexCache - baseline.indexCache,
    3 * 1024 * model.kvCacheProfile.indexHeadDim * 0.5,
  );
});
