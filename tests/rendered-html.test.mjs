import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the inference memory planner", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /推理显存建模/);
  assert.match(html, /模型与负载/);
  assert.match(html, /模型配置/);
  assert.match(html, /本地专家数/);
  assert.match(html, /Hidden size/);
  assert.match(html, /专家总数/);
  assert.match(html, /HCCL buffer/);
  assert.match(html, /CANN \+ PTA \+ 算子/);
  assert.match(html, /单卡非权重显存预估/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("includes accessible numeric controls and a live result region", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /type="number"/);
  assert.match(html, /aria-live="polite"/);
  assert.match(html, /切换到深色模式/);
});
