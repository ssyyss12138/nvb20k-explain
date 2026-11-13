/* nvb20k 深度解剖 — 交互：档位切换 / 模块讲解 / 导航高亮 */
"use strict";

const TIERS = {
  "3e17": {
    id: "3e17", h: 640, L: 18, heads: 5, kv: 5, inter: 1728, veN: 9,
    front: 4, mid: 10, back: 4, params: 86657522,
    flopsPerTok: "1.5277e8", steps: 3260, tokens: "854,589,440",
    trainSeq: 2048, evalSeq: 3072, wd: "0.03"
  },
  "1e18": {
    id: "1e18", h: 768, L: 26, heads: 6, kv: 6, inter: 2048, veN: 13,
    front: 5, mid: 16, back: 5, params: 171652698,
    flopsPerTok: "2.3209e8", steps: 6694, tokens: "1,754,791,936",
    trainSeq: 2048, evalSeq: 3072, wd: "0.015"
  },
  "3e18": {
    id: "3e18", h: 1024, L: 29, heads: 8, kv: 8, inter: 2752, veN: 15,
    front: 6, mid: 17, back: 6, params: 336784918,
    flopsPerTok: "4.4029e8", steps: 6602, tokens: "2,596,012,032",
    trainSeq: 3072, evalSeq: 3072, wd: "0.015"
  },
  "1e19": {
    id: "1e19", h: 1152, L: 39, heads: 9, kv: 9, inter: 3072, veN: 20,
    front: 8, mid: 23, back: 8, params: 556503290,
    flopsPerTok: "6.6886e8", steps: 13951, tokens: "5,485,756,416",
    trainSeq: 3072, evalSeq: 3072, wd: "0.015"
  }
};

const NUMFMT = n => n.toLocaleString("en-US");
const CUR = { tier: "1e19" };
const tier = () => TIERS[CUR.tier];

function applyTier() {
  const t = tier();
  document.querySelectorAll("[data-tier]").forEach(el => {
    const f = el.getAttribute("data-tier");
    let v = t[f];
    if (f === "layers") v = `${t.L} 层（前 ${t.front} / 中 ${t.mid} / 后 ${t.back}）`;
    if (f === "heads") v = `${t.heads} × 128`;
    if (f === "fv") v = `${NUMFMT(t.params)}`;
    if (f === "tl") v = `${t.tensors} 个张量`;
    el.textContent = v;
  });
  document.querySelectorAll(".seg [data-set-tier]").forEach(b => {
    b.classList.toggle("on", b.getAttribute("data-set-tier") === CUR.tier);
  });
  document.querySelectorAll("#tierTable tbody tr").forEach(tr => {
    tr.classList.toggle("hi", tr.getAttribute("data-tier-row") === CUR.tier);
  });
  document.querySelectorAll("[data-tiermid]").forEach(el => {
    const f = el.getAttribute("data-tiermid");
    let v = "";
    if (f === "range") v = `L${t.front} … L${t.front + t.mid - 1}`;
    if (f === "shortenAt") v = `第 ${t.front} 层`;
    if (f === "expandAt") v = `第 ${t.front + t.mid} 层`;
    if (f === "midLen") v = `${Math.round(t.evalSeq / 16)}`;
    if (f === "seq") v = `${NUMFMT(t.evalSeq)}`;
    el.textContent = v;
  });
}


function init() {
  try {
    const q = new URLSearchParams(location.search).get("tier");
    if (q && TIERS[q]) CUR.tier = q;
  } catch (e) { /* ignore */ }

  document.querySelectorAll(".seg [data-set-tier]").forEach(b => {
    b.classList.toggle("on", b.getAttribute("data-set-tier") === CUR.tier);
    b.addEventListener("click", () => {
      CUR.tier = b.getAttribute("data-set-tier");
      applyTier();
    });
  });

  const links = [...document.querySelectorAll(".nav a")];
  const secs = links.map(a => document.querySelector(a.getAttribute("href")));
  const updateNav = () => {
    const y = window.scrollY + 150;
    let idx = -1;
    secs.forEach((s, i) => {
      if (!s) return;
      const top = s.getBoundingClientRect().top + window.scrollY;
      if (top <= y) idx = i;
    });
    links.forEach((a, i) => a.classList.toggle("on", i === idx));
  };
  window.addEventListener("scroll", updateNav, { passive: true });
  updateNav();

  applyTier();
}

document.addEventListener("DOMContentLoaded", init);
