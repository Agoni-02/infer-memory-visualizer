"use client";

import { useMemo, useState } from "react";

type Inputs = {
  hiddenSize: number;
  maxBatchedTokens: number;
  dpSize: number;
  tpSize: number;
  epSize: number;
  topK: number;
  localExpertNum: number;
  maxBS: number;
  graphCount: number;
  cannGB: number;
};

const DEFAULTS: Inputs = {
  hiddenSize: 7168,
  maxBatchedTokens: 4096,
  dpSize: 8,
  tpSize: 8,
  epSize: 8,
  topK: 8,
  localExpertNum: 32,
  maxBS: 128,
  graphCount: 5,
  cannGB: 1,
};

const GB = 1_000_000_000;
const MB = 1_000_000;
const align = (value: number, boundary: number) =>
  Math.ceil(value / boundary) * boundary;
const align480To512 = (value: number) => Math.ceil(value / 480) * 512;

function formatGB(bytes: number) {
  return `${(bytes / GB).toLocaleString("zh-CN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} GB`;
}

function formatMB(bytes: number) {
  return `${(bytes / MB).toLocaleString("zh-CN", {
    maximumFractionDigits: 1,
  })} MB`;
}

function safe(value: number, fallback = 0) {
  return Number.isFinite(value) ? Math.max(0, value) : fallback;
}

export default function Home() {
  const [inputs, setInputs] = useState(DEFAULTS);
  const [dark, setDark] = useState(false);

  const result = useMemo(() => {
    const H = safe(inputs.hiddenSize);
    const T = safe(inputs.maxBatchedTokens);
    const dp = Math.max(1, safe(inputs.dpSize, 1));
    const ep = Math.max(1, safe(inputs.epSize, 1));
    const K = safe(inputs.topK);
    const localExperts = safe(inputs.localExpertNum);
    const maxBS = safe(inputs.maxBS);

    const hiddenResidual = 2 * 2 * T * H;
    const moeBuffers = 4 * 2 * dp * T * K / ep * H;
    const activation = hiddenResidual + moeBuffers;

    const hcclDP = Math.max(Math.ceil(((dp + 1) * 4) / 1024 ** 2), 50) * 2 * MB;
    const hcclTP = 200 * 2 * MB;
    const alignedDispatch = align480To512(align(2 * H, 32) + 64);
    const alignedCombine = align(2 * H, 512);
    const epDispatch = localExperts * maxBS * ep * alignedDispatch;
    const epCombine = K * maxBS * alignedCombine;
    const hcclEP = 2 * (epDispatch + epCombine);
    const hccl = hcclDP + hcclTP + hcclEP;

    const graph = (safe(inputs.graphCount) / 5) * 0.27 * GB;
    const cann = safe(inputs.cannGB) * GB;
    const total = activation + hccl + graph + cann;

    return {
      activation,
      hiddenResidual,
      moeBuffers,
      hccl,
      hcclDP,
      hcclTP,
      hcclEP,
      epDispatch: 2 * epDispatch,
      epCombine: 2 * epCombine,
      alignedDispatch,
      alignedCombine,
      graph,
      cann,
      total,
    };
  }, [inputs]);

  const update = (key: keyof Inputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: Number(value) }));
  };

  const categories = [
    { label: "激活占用", value: result.activation, color: "var(--coral)" },
    { label: "HCCL buffer", value: result.hccl, color: "var(--blue)" },
    { label: "图占用", value: result.graph, color: "var(--violet)" },
    { label: "CANN + PTA + 算子", value: result.cann, color: "var(--green)" },
  ];

  return (
    <main className={dark ? "app dark" : "app"}>
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="shell">
        <header className="topbar">
          <div className="brand-block">
            <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
            <div>
              <h1>推理显存建模</h1>
              <p>Memory planner for distributed MoE inference</p>
            </div>
          </div>
          <div className="top-actions">
            <span className="live"><i />实时估算</span>
            <button
              className="theme-toggle"
              type="button"
              onClick={() => setDark((value) => !value)}
              aria-label={dark ? "切换到浅色模式" : "切换到深色模式"}
            >
              <span>{dark ? "☀" : "☾"}</span>{dark ? "浅色" : "深色"}
            </button>
          </div>
        </header>

        <section className="workspace">
          <aside className="control-card">
            <div className="section-title">
              <div>
                <span className="eyebrow">CONFIGURATION</span>
                <h2>模型与负载</h2>
              </div>
              <button className="reset" type="button" onClick={() => setInputs(DEFAULTS)}>重置</button>
            </div>

            <fieldset>
              <legend>核心参数</legend>
              <div className="field-grid">
                <NumberField label="Hidden size" value={inputs.hiddenSize} onChange={(v) => update("hiddenSize", v)} />
                <NumberField label="Max batched tokens" value={inputs.maxBatchedTokens} onChange={(v) => update("maxBatchedTokens", v)} />
              </div>
            </fieldset>

            <fieldset>
              <legend>并行策略</legend>
              <div className="field-grid three">
                <NumberField label="DP size" value={inputs.dpSize} onChange={(v) => update("dpSize", v)} />
                <NumberField label="TP size" value={inputs.tpSize} onChange={(v) => update("tpSize", v)} />
                <NumberField label="EP size" value={inputs.epSize} onChange={(v) => update("epSize", v)} />
              </div>
              <div className="field-grid">
                <NumberField label="TopK 专家" value={inputs.topK} onChange={(v) => update("topK", v)} />
                <NumberField label="本地专家数" value={inputs.localExpertNum} onChange={(v) => update("localExpertNum", v)} />
              </div>
            </fieldset>

            <fieldset>
              <legend>运行时</legend>
              <div className="field-grid">
                <NumberField label="Max BS" value={inputs.maxBS} onChange={(v) => update("maxBS", v)} />
                <NumberField label="图个数" value={inputs.graphCount} onChange={(v) => update("graphCount", v)} />
              </div>
              <NumberField label="CANN + PTA + 算子预估（GB）" value={inputs.cannGB} step="0.1" onChange={(v) => update("cannGB", v)} />
              <p className="field-note">该项默认按 1 GB 预留，实际占用通常低于此值。</p>
            </fieldset>
          </aside>

          <section className="results" aria-live="polite">
            <article className="hero-card">
              <div className="hero-copy">
                <span className="eyebrow">ESTIMATED PER DEVICE</span>
                <div className="total-line"><strong>{formatGB(result.total).replace(" GB", "")}</strong><span>GB</span></div>
                <p>单卡非权重显存预估</p>
              </div>
              <div className="ring" style={{ "--progress": `${Math.min(100, result.total / (8 * GB) * 100)}%` } as React.CSSProperties}>
                <div><strong>{Math.ceil(result.total / GB)}</strong><span>GB 档位</span></div>
              </div>
            </article>

            <div className="metric-grid">
              {categories.map((item) => (
                <article className="metric-card" key={item.label}>
                  <div className="metric-label"><i style={{ background: item.color }} />{item.label}</div>
                  <strong>{formatGB(item.value)}</strong>
                  <span>{result.total ? `${(item.value / result.total * 100).toFixed(1)}%` : "0%"} of total</span>
                </article>
              ))}
            </div>

            <article className="breakdown-card">
              <div className="panel-heading">
                <div><span className="eyebrow">MEMORY MAP</span><h2>显存构成</h2></div>
                <span className="unit-pill">十进制 GB</span>
              </div>
              <div className="stack" aria-label="显存构成比例图">
                {categories.map((item) => (
                  <div
                    key={item.label}
                    title={`${item.label}: ${formatGB(item.value)}`}
                    style={{ width: `${result.total ? item.value / result.total * 100 : 0}%`, background: item.color }}
                  />
                ))}
              </div>
              <div className="legend">
                {categories.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}
              </div>

              <div className="detail-sections">
                <DetailSection title="激活占用" value={result.activation} tone="coral">
                  <DetailRow label="Hidden states + residual" value={result.hiddenResidual} formula={`2 × 2 B × ${inputs.maxBatchedTokens} × ${inputs.hiddenSize}`} />
                  <DetailRow label="4 份 MoE 激活 buffer" value={result.moeBuffers} formula={`4 × 2 B × ${inputs.dpSize} × ${inputs.maxBatchedTokens} × ${inputs.topK} ÷ ${inputs.epSize} × ${inputs.hiddenSize}`} />
                </DetailSection>

                <DetailSection title="HCCL buffer" value={result.hccl} tone="blue">
                  <DetailRow label="DP buffer" value={result.hcclDP} formula={`max(ceil((${inputs.dpSize} + 1) × 4 ÷ 1024²), 50) × 2 MB`} />
                  <DetailRow label="TP buffer" value={result.hcclTP} formula="200 MB × 2" />
                  <DetailRow label="EP buffer" value={result.hcclEP} formula="2 × (dispatch + combine)" />
                  <div className="sub-detail">
                    <span>Dispatch {formatMB(result.epDispatch)}</span>
                    <span>Combine {formatMB(result.epCombine)}</span>
                    <span>480Align512 = {result.alignedDispatch.toLocaleString("zh-CN")} B</span>
                    <span>Align512 = {result.alignedCombine.toLocaleString("zh-CN")} B</span>
                  </div>
                </DetailSection>

                <DetailSection title="其他运行时" value={result.graph + result.cann} tone="violet">
                  <DetailRow label={`图占用（${inputs.graphCount} 张）`} value={result.graph} formula={`${inputs.graphCount} ÷ 5 × 0.27 GB`} />
                  <DetailRow label="CANN + PTA + 算子" value={result.cann} formula={`${inputs.cannGB} GB 预估值`} />
                </DetailSection>
              </div>
            </article>

            <p className="method-note"><strong>口径说明</strong> 所有结果均为单卡估算；GB 按 10⁹ bytes 计算。TP size 作为部署配置展示，TP HCCL buffer 按给定公式固定计入 400 MB。</p>
          </section>
        </section>
      </div>
    </main>
  );
}

function NumberField({ label, value, onChange, step = "1" }: { label: string; value: number; onChange: (value: string) => void; step?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" min="0" step={step} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function DetailSection({ title, value, tone, children }: { title: string; value: number; tone: string; children: React.ReactNode }) {
  return (
    <section className={`detail-section ${tone}`}>
      <div className="detail-title"><span>{title}</span><strong>{formatGB(value)}</strong></div>
      {children}
    </section>
  );
}

function DetailRow({ label, value, formula }: { label: string; value: number; formula: string }) {
  return (
    <div className="detail-row">
      <div><span>{label}</span><code>{formula}</code></div>
      <strong>{formatMB(value)}</strong>
    </div>
  );
}
