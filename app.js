// Replays recorded lookups. Every number on the page comes from data.json.
const SYS = "KuroANN";
const BASELINES = { pf: "PipeANN-Filter", faiss: "Faiss-SSD" };
const TARGETS = ["0.9", "0.95", "0.99"];

const EN = {
  tag: "Filtered vector search demo",
  h1: "Filtered vector search that knows how much to read before its first SSD read",
  lede: "A filter can match a handful of vectors or most of them, and each case needs a different search. {sys} counts the matches of every query before it reads the SSD, then uses that count to set how widely to search and when to stop. Pick a query to replay a recorded lookup and compare it with other SSD systems at the same recall.",
  tabSearch: "Semantic search",
  tabRec: "Recommendation",
  tabRag: "RAG",
  ledeSearch: "23.9 million PubMed abstracts, embedded with BGE-M3. Filters come from the National Library of Medicine records: publication type, year, and MeSH terms.",
  ledeRec: "{n} Amazon products, embedded with BLaIR. Pick a shopper and filter as on a store page. The query is the shopper's recent purchases, averaged, and every choice of filters replays its own recorded lookup.",
  shopAs: "Shopping as",
  facetDept: "Department",
  allDepts: "All departments",
  facetSub: "Subcategory",
  allSub: "All of {d}",
  facetPrice: "Price",
  anyPrice: "Any price",
  under: "Under ${p}",
  facetRating: "Customer rating",
  anyRating: "Any rating",
  stars4: "4 stars and up",
  matchBar: "{c} of {n} products match",
  sortNote: "Ranked by how close each product is to the shopper's recent purchases",
  speedH: "Search time for these filters",
  howShop: "{sys} counted {c} matches before its first read, then {route} and read {r} vectors from the SSD.",
  routeAllShop: "ranked every match by its code",
  routePartsShop: "ranked the matches in the {p} nearest partitions",
  tooFew: "fewer than 10",
  speedNote: "Times are of this one lookup, alone on the server. QPS is the throughput of each system on all filters that match {b} of the products, with 28 threads.",
  loadLabel: "Server load, queries per second",
  loadIdle: "Idle",
  loadNoteIdle: "The lookup alone on the server.",
  loadNote: "The same lookup, recorded while the server also answered other queries arriving at {r} per second.",
  thSearchLoad: "Search",
  thWait: "of which waiting",
  loadStopped: "not run at this load: p99 over 10 s from {r} QPS",
  stepSearchWait: "{c} of {n} abstracts match, {r} SSD reads, recall {recall}, {w} waiting in the queue",
  ledeRag: "A question with a scope, such as randomized trials since 2020, goes to the same PubMed index. The model answers only from the abstracts that the search returns.",
  pickSearch: "Questions",
  pickRec: "Shoppers",
  pickRag: "Questions",
  targetLabel: "Recall target",
  thSystem: "System",
  thTime: "Time",
  thReads: "SSD reads",
  thRecall: "Recall",
  notReached: "not reached, reaches {r} at most",
  stepCount: "Count the matches",
  stepCountNote: "{c} of {n} match, known before the first read",
  stepPlan: "Pick a plan row",
  routeAll: "rank every match by its code",
  routeParts: "rank the matches in the {p} nearest partitions",
  planNote: "{route}, keep the best {pool}",
  planStop: ", stop early with the calibrated table",
  stepRead: "Read from the SSD",
  stepReadNote: "{r} of {pool} candidates read, recall {recall}",
  mapRead: "read {r}",
  mapSkip: "not read {u}",
  mapPer: "one cell is {p} candidates",
  raceH: "The same lookup on each system",
  raceNote: "Replayed {f}\u00d7 slower than recorded, so the times keep their ratio.",
  readsVal: "{r} SSD reads",
  noPicture: "No product picture",
  mCount: "{c} of {n} vectors match, counted in memory before the first SSD read.",
  mRow: "The count selects the plan row for up to {v} matches, with a pool of {pool} candidates{stop}.",
  mRowStop: " and early stopping",
  mLegend: "Plan for recall {t}. Each bar is a row, as wide as its range of counts and as tall as its pool.",
  mAll: "All matches",
  mParts: "Nearest partitions",
  mRankAll: "Rank all {c} matches by their codes in memory and keep the best {pool}.",
  mRankParts: "Rank the matches in the {p} nearest of {n} partitions by their codes in memory and keep the best {pool}.",
  mGrid: "one cell is {k} partitions",
  mReadStop: "Read {r} of the {pool} candidates from the SSD, best estimate first, and stop once no unread candidate can beat the 10th best.",
  mReadAll: "Read the {pool} candidates from the SSD, best estimate first.",
  times: "{x}\u00d7",
  glanceQps: "more queries per second than the fastest other system",
  glanceSearch: "PubMed semantic search, recall {t}, filters that match {b}",
  glanceRec: "Amazon recommendation, recall {t}, filters that match {b}",
  glanceRag: "lower p99 latency than {base} under load",
  glanceRagC: "RAG retrieval at {r} queries per second",
  leadBadge: "faster than {base} on this lookup",
  leadAlone: "Only {sys} reaches this recall on this lookup.",
  capWhy: "more queries per second than the faster other system, averaged over filter widths. Up to {m} on one width.",
  stepEmbed: "BGE-M3 embeds the question",
  stepSearch: "{sys} searches",
  stepSearchNote: "{c} of {n} abstracts match, {r} SSD reads, recall {recall}",
  stepPrefill: "The model reads the prompt",
  stepPrefillNote: "{t} tokens from {k} abstracts",
  stepAnswer: "The model answers",
  stepAnswerNote: "{t} tokens",
  resultsSearch: "Nearest abstracts that pass the filter",
  resultsRec: "Recommended products that pass the filter",
  resultsRecNote: "Names, prices, and ratings come from the dataset. The images are generated illustrations, not seller photos.",
  resultsRag: "Abstracts the answer draws on",
  capH: "Throughput by filter width",
  capLede: "Queries per second on the whole workload, in bands by the share of vectors that a filter matches. {sys} runs its plan for the recall target, and each other system runs at its fastest setting that reaches the recall {sys} achieves on the band.",
  capFoot: "Queries per second, on a log scale. All systems run on the same machine with the same queries and 28 threads, and they use the same SSD for their files.",
  ragFoot: "The answer comes from Llama-3.1-8B-Instruct. Its times are computed from rates measured with vLLM on an NVIDIA A30: {a} ms plus {b} ms per prompt token, and {d} tokens per second. Embedding and search times are measured.",
  footer: "This page replays recorded lookups. Recall is measured against the exact filtered answer, and vectors tied with the 10th nearest count as found.",
  themeDark: "Dark",
  themeLight: "Light",
  band0: "under 0.01%",
  band1: "0.01–0.1%",
  band2: "0.1–1%",
  band3: "1–5%",
  band4: "5–20%",
  band5: "over 20%",
  langOther: "한국어",
};
let KO = {};
let lang = "en";
const t = (k, v = {}) => {
  let s = (lang === "ko" && KO[k]) || EN[k] || k;
  s = s.replaceAll("{sys}", SYS);
  for (const [a, b] of Object.entries(v)) s = s.replaceAll(`{${a}}`, b);
  return s;
};
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const fmt = (x) => Math.round(x).toLocaleString("en-US");
const ms = (us) => { const m = us / 1000; return (m >= 100 ? fmt(m) : m >= 10 ? m.toFixed(1) : m.toFixed(2)) + " ms"; };
const sec = (m) => (m >= 1000 ? (m / 1000).toFixed(2) + " s" : ms(m * 1000));
const millions = (n) => (lang === "ko" ? fmt(Math.round(n / 1e4 / 10) * 10) + "만" : (n / 1e6).toFixed(1) + " million");

let DATA = null;
let ragLoad = null;   // the RAG tab opens at the load of its headline tile
const state = { tab: "search", index: { search: 0, rec: 0, rag: 0 }, target: "0.99", timers: [], shop: null, load: null };

let runId = 0;
function clearTimers() { runId++; state.timers.forEach(clearTimeout); state.timers = []; }
function later(f, wait) { state.timers.push(setTimeout(f, wait)); }
const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const easeOut = (p) => 1 - (1 - p) ** 3;
const times = (x) => t("times", { x: x >= 10 ? x.toFixed(0) : x.toFixed(1) });
// Counts a multiplier up from 1x.
function countTo(node, x, dur) {
  if (still()) { node.textContent = times(x); return; }
  let t0 = null, done = false;
  const end = () => { if (!done) { done = true; node.textContent = times(x); } };
  const tick = (now) => {
    if (done || !node.isConnected) return;
    if (t0 === null) t0 = now;
    const p = Math.min(1, (now - t0) / dur);
    if (p >= 1) end(); else { node.textContent = times(1 + (x - 1) * easeOut(p)); requestAnimationFrame(tick); }
  };
  node.textContent = times(1);
  requestAnimationFrame(tick);
  setTimeout(end, dur + 60);
}

// KuroANN over the fastest other system that reached the same recall.
function lookupLead(sys) {
  const others = ["pf", "faiss"].filter((b) => sys[b]);
  if (!sys.kuro) return null;
  if (!others.length) return { alone: true };
  const next = others.reduce((a, b) => (sys[a].us <= sys[b].us ? a : b));
  return { x: sys[next].us / sys.kuro.us, next };
}
function bandLead(cap, b, target) {
  const k = cap.kuro?.[b]?.[target];
  if (!k?.reached) return null;
  const qps = ["pf", "faiss"].map((x) => cap[x]?.[b]?.[target]).filter((v) => v?.reached).map((v) => v.qps);
  return qps.length ? k.qps / Math.max(...qps) : null;
}
function showLead(box, lead) {
  box.replaceChildren();
  box.hidden = !lead || (!lead.alone && lead.x <= 1);
  if (box.hidden) return;
  if (lead.alone) { box.append(el("span", "why", t("leadAlone", { sys: SYS }))); return; }
  const x = el("span", "x");
  box.append(x, el("span", "why", t("leadBadge", { base: BASELINES[lead.next] })));
  countTo(x, lead.x, 700);
}

// One tile per tab: the largest lead in its recorded data, with the two systems as bars.
function renderGlance() {
  const box = $("glance"); box.replaceChildren();
  const tile = (x, name, what, bars, cond) => {
    const s = el("div", "stat"), big = el("span", "big"), rows = el("div", "bars");
    const top = Math.max(...bars.map((b) => b.v));
    bars.forEach((b) => {
      const r = el("div", b.me ? "bar-row me" : "bar-row"), track = el("span", "bt"), fill = document.createElement("i");
      fill.style.width = `${Math.max(2, (100 * b.v) / top)}%`;
      track.append(fill);
      r.append(el("span", "bn", b.n), track, el("span", "bv", b.label));
      rows.append(r);
    });
    s.append(el("span", "name", name), big, el("span", "what", what), rows, el("span", "cond", cond));
    box.append(s);
    countTo(big, x, 1100);
  };
  [["search", "glanceSearch"], ["rec", "glanceRec"]].forEach(([tab, key]) => {
    const cap = DATA[tab]?.capacity;
    if (!cap) return;
    let best = null;
    TARGETS.forEach((tg) => Object.keys(cap.kuro).forEach((b) => {
      const x = bandLead(cap, b, tg);
      if (x && (!best || x > best.x)) best = { x, tg, b };
    }));
    if (!best) return;
    const at = (s) => cap[s]?.[best.b]?.[best.tg];
    const other = ["pf", "faiss"].filter((s) => at(s)?.reached).sort((a, b) => at(b).qps - at(a).qps)[0];
    const bars = [{ n: SYS, v: at("kuro").qps, me: true }, { n: BASELINES[other], v: at(other).qps }];
    bars.forEach((b) => (b.label = `${fmt(b.v)}\u00a0QPS`));
    tile(best.x, t(tab === "search" ? "tabSearch" : "tabRec"), t("glanceQps"), bars,
      t(key, { t: (+best.tg).toFixed(2), b: t(`band${best.b}`) }));
  });
  // RAG: rates that Faiss still serves (at least 95% of the offered rate).
  const L = DATA.rag?.load?.summary;
  if (L?.kuro && L?.faiss) {
    let best = null;
    Object.keys(L.faiss).forEach((r) => {
      const f = L.faiss[r], k = L.kuro[r];
      if (k && f.achieved >= 0.95 * +r && (!best || f.p99 / k.p99 > best.x)) best = { x: f.p99 / k.p99, r };
    });
    if (best) {
      ragLoad = +best.r;
      const bars = [{ n: SYS, v: L.kuro[best.r].p99, me: true }, { n: BASELINES.faiss, v: L.faiss[best.r].p99 }];
      bars.forEach((b) => (b.label = `${fmt(b.v / 1000)}\u00a0ms`));
      tile(best.x, t("tabRag"), t("glanceRag", { base: BASELINES.faiss }), bars, t("glanceRagC", { r: fmt(+best.r) }));
    }
  }
  box.hidden = !box.children.length;
}
// Calls f(p) for p from 0 to 1 over dur ms after delay; a newer replay stops it.
function tween(dur, delay, f) {
  const id = runId;
  let t0 = null, ended = false;
  const end = () => { if (id === runId && !ended) { ended = true; f(1); } };
  later(() => {
    const tick = (now) => {
      if (id !== runId || ended) return;
      if (t0 === null) t0 = now;
      const p = dur > 0 ? Math.min(1, (now - t0) / dur) : 1;
      if (p >= 1) end(); else { f(p); requestAnimationFrame(tick); }
    };
    requestAnimationFrame(tick);
  }, delay);
  later(end, delay + dur + 60);
}

function applyStatic() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-sys]").forEach((e) => { e.textContent = SYS; });
  document.querySelectorAll("[data-i18n]").forEach((e) => { e.textContent = t(e.dataset.i18n); });
  document.title = `${SYS} demo`;
  $("lang").textContent = lang === "ko" ? "English" : t("langOther");
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  $("theme").textContent = dark ? t("themeLight") : t("themeDark");
  $("footer").textContent = t("footer");
  const leg = $("cap-legend"); leg.replaceChildren();
  [["kuro", SYS], ["pf", BASELINES.pf], ["faiss", BASELINES.faiss]].forEach(([c, name]) => {
    const s = el("span"); const i = el("i", `cap-fill ${c}`); i.style.height = "10px"; s.append(i, document.createTextNode(name)); leg.append(s);
  });
}

function chips(parts) { const c = el("span", "chips"); parts.forEach((p) => c.append(el("span", "chip", p))); return c; }

function renderExamples() {
  const box = $("examples"); box.replaceChildren();
  const set = current();
  set.questions.forEach((q, i) => {
    const b = el("button"); b.type = "button";
    if (i === state.index[state.tab]) b.setAttribute("aria-current", "true");
    if (state.tab === "rec") {
      const h = el("span", "history");
      q.history.slice(0, 5).forEach((x) => { const im = el(x.image ? "img" : "span", x.image ? "" : "noimg"); if (x.image) { im.src = x.image; im.alt = x.title; } im.title = x.title; h.append(im); });
      b.append(h, chips(q.filter));
    } else {
      b.append(el("span", "ex-q", q.q), chips(q.filter));
    }
    b.addEventListener("click", () => { state.index[state.tab] = i; renderExamples(); replay(); });
    box.append(b);
  });
}

function current() { return state.tab === "search" ? DATA.search : state.tab === "rec" ? DATA.rec : DATA.rag; }

function planText(row) {
  const f = Object.fromEntries(row.split(/\s+/).map((kv) => kv.split("=")));
  const route = f.np === "0" ? t("routeAll") : t("routeParts", { p: fmt(+f.np) });
  return { text: t("planNote", { route, pool: fmt(+f.L) }) + (+f.alpha > 0 ? t("planStop") : ""), pool: +f.L };
}

function step(name, time, note) {
  const li = el("li");
  li.append(el("span", "name", name), el("span", "time", time || ""));
  if (note) li.append(el("span", "note", note));
  return li;
}

function reveal(items, gaps) {
  let wait = 0;
  items.forEach((li, i) => {
    wait += gaps[i] || 0;
    later(() => li.classList.add("done"), wait);
  });
}

function replay() {
  clearTimers();
  if (state.tab === "rec" && DATA.rec.shop) renderShop();
  else if (state.tab === "rag") replayRag();
  else replayLookup();
  renderCapacity();
}

// Recommendation tab: one recorded lookup per department, subcategory, price limit and rating.
const shopKey = (s, f) => [s.q, f.dept || "", f.sub || "", f.price || "", f.rating || ""].join("|");
// A shopper's page opens on the subcategory of their purchases, at a price limit and rating whose lookup KuroANN wins at
// every target (fewest missing pictures, then fewest filters, then the largest lead).
function shopStart(i) {
  const shop = DATA.rec.shop, docs = DATA.rec.docs, s = shop.shoppers[i];
  const base = { shopper: i, dept: s.dept, sub: s.subs[0] || null, price: null, rating: null };
  const before = (a, b) => { for (let j = 0; j < a.length; j++) if (a[j] !== b[j]) return a[j] < b[j]; return false; };
  let best = base, bestRank = null;
  for (const price of [null, 200, 100, 50, 25, 10]) {
    for (const rating of [null, 4]) {
      const f = { ...base, price, rating }, L = shop.lookups[shopKey(s, f)];
      if (!L) continue;
      const runs = Object.values(L.systems);
      if (!runs.every((x) => ["pf", "faiss"].every((b) => !x[b] || x.kuro.us < x[b].us))) continue;
      const bare = new Set(runs.flatMap((x) => x.kuro.hits.slice(0, 10)).filter((id) => !docs[id]?.image)).size;
      const x = L.systems[state.target], lead = Math.min(...["pf", "faiss"].filter((b) => x[b]).map((b) => x[b].us / x.kuro.us));
      const rank = [bare, (price ? 1 : 0) + (rating ? 1 : 0), -lead];
      if (!bestRank || before(rank, bestRank)) { best = f; bestRank = rank; }
    }
  }
  return best;
}

function shopFacets(shop, sh, cur) {
  const box = $("facets"); box.replaceChildren();
  const group = (title, opts) => {
    const g = el("div", "facet"); g.append(el("span", "label", title));
    opts.forEach(([label, patch]) => {
      const next = { ...cur, ...patch }, hit = shop.lookups[shopKey(sh, next)];
      const on = Object.entries(patch).every(([k, v]) => (cur[k] || null) === (v || null));
      const b = el("button", "opt"); b.type = "button"; b.setAttribute("aria-pressed", on ? "true" : "false");
      b.disabled = !hit && !on;
      b.append(el("span", "opt-l", label), el("span", "opt-n", hit ? fmt(hit.count) : t("tooFew")));
      b.addEventListener("click", () => { state.shop = next; replay(); });
      g.append(b);
    });
    box.append(g);
  };
  group(t("facetDept"), [[t("allDepts"), { dept: null, sub: null }], [sh.dept, { dept: sh.dept }]]);
  if (cur.dept) group(t("facetSub"), [[t("allSub", { d: sh.dept }), { sub: null }], ...sh.subs.map((x) => [x, { sub: x }])]);
  group(t("facetPrice"), [[t("anyPrice"), { price: null }], ...[10, 25, 50, 100, 200].map((p) => [t("under", { p }), { price: p }])]);
  group(t("facetRating"), [[t("anyRating"), { rating: null }], [t("stars4"), { rating: 4 }]]);
}

function shopSpeed(L) {
  const sys = L.systems[state.target] || {};
  const rows = $("speed-rows"); rows.replaceChildren();
  const times = ["kuro", "pf", "faiss"].map((x) => sys[x]?.us).filter((x) => x > 0);
  const lo = Math.log10(Math.min(...times) / 3), hi = Math.log10(Math.max(...times) * 1.05);
  [["kuro", SYS], ["pf", BASELINES.pf], ["faiss", BASELINES.faiss]].forEach(([x, name]) => {
    const r = el("div", `speed-row ${x}`), s = sys[x];
    r.append(el("span", "speed-name", name));
    if (s) {
      const track = el("span", "speed-track"), fill = el("i", `cap-fill ${x}`);
      fill.style.width = "0%"; track.append(fill);
      later(() => { fill.style.width = `${(100 * (Math.log10(s.us) - lo)) / (hi - lo)}%`; }, 30);
      const cap = DATA.rec.capacity?.[x]?.[L.band]?.[state.target];
      r.append(track, el("span", "speed-val", [ms(s.us), cap?.reached ? `${fmt(cap.qps)} QPS` : null, `${t("thRecall")} ${s.recall.toFixed(2)}`].filter(Boolean).join(" · ")));
    } else {
      const atMost = DATA.rec.capacity?.[x]?.[L.band]?.[state.target]?.at_most;
      r.append(el("span", "speed-none", t("notReached", { r: atMost === undefined ? "–" : atMost.toFixed(3) })));
    }
    rows.append(r);
  });
  const leadBox = $("shop-lead"); leadBox.hidden = true;
  if (still()) showLead(leadBox, lookupLead(sys)); else later(() => showLead(leadBox, lookupLead(sys)), 450);
  const k = sys.kuro, plan = Object.fromEntries(k.plan.split(/\s+/).map((kv) => kv.split("=")));
  const route = plan.np === "0" ? t("routeAllShop") : t("routePartsShop", { p: fmt(+plan.np) });
  $("speed-how").textContent = t("howShop", { c: fmt(L.count), route, r: fmt(k.reads) }) + " " + t("speedNote", { b: t(`band${L.band}`) });
}

function renderShop() {
  const shop = DATA.rec.shop;
  if (!state.shop || !shop.lookups[shopKey(shop.shoppers[state.shop.shopper], state.shop)]) state.shop = shopStart(state.shop ? state.shop.shopper : 0);
  const cur = state.shop, sh = shop.shoppers[cur.shopper], L = shop.lookups[shopKey(sh, cur)];

  const who = $("shoppers"); who.replaceChildren();
  shop.shoppers.forEach((s, i) => {
    const b = el("button", "shopper"); b.type = "button";
    if (i === cur.shopper) b.setAttribute("aria-current", "true");
    const h = el("span", "history");
    s.history.slice(0, 5).forEach((x) => { const im = el(x.image ? "img" : "span", x.image ? "" : "noimg"); if (x.image) { im.src = x.image; im.alt = ""; } im.title = x.title; h.append(im); });
    b.append(el("span", "who", s.q), h, el("span", "dept", s.dept));
    b.addEventListener("click", () => { state.shop = shopStart(i); replay(); });
    who.append(b);
  });
  shopFacets(shop, sh, cur);

  const parts = [cur.dept ? cur.sub || cur.dept : t("allDepts"), cur.price ? t("under", { p: cur.price }) : null, cur.rating ? t("stars4") : null].filter(Boolean);
  $("shop-bar").replaceChildren(el("span", "match", t("matchBar", { c: fmt(L.count), n: millions(DATA.rec.size) })), chips(parts), el("span", "sort", t("sortNote")));
  shopSpeed(L);
  renderTargets();

  const k = L.systems[state.target].kuro, list = $("shop-results");
  list.replaceChildren(); list.classList.remove("shown");
  k.hits.slice(0, 10).forEach((id) => {
    const d = DATA.rec.docs[id]; if (!d) return;
    const li = el("li", "item");
    if (d.image) { const im = el("img"); im.src = d.image; im.alt = ""; li.append(im); } else li.append(el("div", "noimg", t("noPicture")));
    li.append(el("div", "title", d.title), el("div", "meta", [d.price !== undefined ? `$${d.price.toFixed(2)}` : "", d.rating !== undefined ? `${d.rating.toFixed(1)} ★` : ""].filter(Boolean).join(" · ")));
    list.append(li);
  });
  later(() => list.classList.add("shown"), Math.min(400, Math.max(60, k.us / 1000)));
}

// The candidates of the plan row as cells in rank order, filled as they are read; the rest stay unread.
function readMap(reads, pool) {
  const total = Math.max(pool, reads), per = Math.max(1, Math.ceil(total / 384));
  const n = Math.ceil(total / per), r = Math.min(n, Math.ceil(reads / per));
  const grid = el("div", "readmap"), cells = [];
  for (let i = 0; i < n; i++) { const x = el("i"); grid.append(x); cells.push(x); }
  const key = el("div", "readkey");
  const item = (cls, text) => { const s = el("span", cls); s.append(el("b"), document.createTextNode(text)); return s; };
  key.append(item("r", t("mapRead", { r: fmt(reads) })));
  if (total > reads) key.append(item("s", t("mapSkip", { u: fmt(total - reads) })));
  if (per > 1) key.append(el("span", "", t("mapPer", { p: fmt(per) })));
  let shown = 0;
  const fill = (p) => {
    for (const upto = Math.round(r * p); shown < upto; shown++) cells[shown].classList.add("r");
    if (p >= 1) for (let i = r; i < n; i++) cells[i].classList.add("s");
  };
  return { grid, key, fill };
}

const SVG_NS = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

// The four stages of one lookup: count, plan row, ranking in memory, reads from the SSD.
function renderMechanism(q, s, set, factor) {
  const box = $("mech"); box.replaceChildren(); box.hidden = false;
  const N = set.size, rows = set.plans[state.target], nlist = set.nlist;
  const f = Object.fromEntries(s.plan.split(/\s+/).map((kv) => kv.split("=")));
  const np = +f.np, pool = +f.L, stop = +f.alpha > 0;
  const chosen = rows.find((r) => q.count <= r.vmax);
  const cap = (num, text) => { const p = el("p", "mcap"); p.append(el("span", "mnum", num), el("span", "mtext", text)); return p; };

  // Stages 1 and 2: one log axis of counts, with the plan rows below it.
  const a = el("div", "mcard");
  const W = 600, pad = 16, top = 46, base = 122;
  const x = (c) => pad + ((W - 2 * pad) * Math.log10(Math.max(1, Math.min(c, N)))) / Math.log10(N);
  const g = svgEl("svg", { viewBox: `0 0 ${W} 132`, class: "maxis", "aria-hidden": "true" });
  [1e-4, 1e-3, 1e-2, 0.05, 0.2].forEach((share) => {
    const cx = x(share * N);
    g.append(svgEl("line", { x1: cx, x2: cx, y1: 26, y2: top, class: "mtick" }));
    const label = svgEl("text", { x: cx, y: 18, class: "mlab", "text-anchor": "middle" });
    label.textContent = `${+(share * 100).toFixed(2)}%`;
    g.append(label);
  });
  g.append(svgEl("line", { x1: pad, x2: W - pad, y1: top, y2: top, class: "maxisline" }));
  const maxPool = Math.max(...rows.map((r) => r.L));
  let low = 0, seg = null;
  rows.forEach((r) => {
    const x0 = x(low + 1), x1 = x(r.vmax);
    low = r.vmax;
    if (x1 - x0 < 0.5) return;
    const h = 10 + (36 * Math.log2(r.L)) / Math.log2(maxPool);
    const rect = svgEl("rect", { x: x0 + 0.5, y: base - h, width: Math.max(1, x1 - x0 - 1), height: h, rx: 2, class: `mseg ${r.np ? "part" : "all"}` });
    g.append(rect);
    if (r === chosen) seg = rect;
  });
  const needle = svgEl("line", { x1: pad, x2: pad, y1: top - 14, y2: base, class: "mneedle" });
  const dot = svgEl("circle", { cx: pad, cy: top, r: 5, class: "mdot" });
  g.append(needle, dot);
  const legend = el("div", "mlegend");
  legend.append(el("span", "mlegend-t", t("mLegend", { t: (+state.target).toFixed(2) })));
  [["all", "mAll"], ["part", "mParts"]].forEach(([k, key]) => { const s2 = el("span", `msw ${k}`); s2.append(el("i"), document.createTextNode(t(key))); legend.append(s2); });
  const count = (c) => t("mCount", { c: fmt(c), n: millions(N) });
  const c1 = cap("1", count(q.count)), c2 = cap("2", t("mRow", { v: fmt(Math.min(chosen.vmax, N)), pool: fmt(pool), stop: stop ? t("mRowStop") : "" }));
  a.append(g, legend, c1, c2);

  // Stage 3: the ranking in memory, over every match or over the nearest partitions.
  const b = el("div", "mcard");
  let rank, c3;
  if (np === 0) {
    const track = el("div", "mfunnel"), fill = el("i"), label = el("span", "mfunnel-n");
    track.append(fill, label);
    const kept = Math.max(4, (100 * Math.log10(pool)) / Math.log10(Math.max(q.count, pool + 1)));
    c3 = cap("3", t("mRankAll", { c: fmt(q.count), pool: fmt(pool) }));
    b.append(c3, track);
    rank = (p) => { fill.style.width = `${100 - (100 - kept) * p}%`; label.textContent = fmt(q.count - (q.count - pool) * p); };
  } else {
    const cells = 256, per = nlist / cells, lit = Math.max(1, Math.round(np / per));
    const grid = el("div", "mgrid"), items = [];
    for (let i = 0; i < cells; i++) { const c = el("i"); grid.append(c); items.push(c); }
    const cx = 19 + (q.count % 7), cy = 3;
    const order = items.map((c, i) => ({ c, d: Math.hypot((i % 32) - cx, Math.floor(i / 32) - cy) * 1.0 + ((i * 37) % 11) / 20 })).sort((u, v) => u.d - v.d);
    c3 = cap("3", t("mRankParts", { p: fmt(np), n: fmt(nlist), pool: fmt(pool) }));
    b.append(c3, grid, el("p", "mnote", t("mGrid", { k: fmt(per) })));
    let shown = 0;
    rank = (p) => { for (const upto = Math.round(lit * p); shown < upto; shown++) order[shown].c.classList.add("on"); };
  }

  // Stage 4: the reads, one cell per candidate in the order of the estimates.
  const c = el("div", "mcard");
  const map = readMap(s.reads, pool), time = el("span", "mtime", ms(s.us));
  const c4 = cap("4", stop && s.reads < pool ? t("mReadStop", { r: fmt(s.reads), pool: fmt(pool) }) : t("mReadAll", { pool: fmt(pool) }));
  c4.append(time);
  c.append(c4, map.grid, map.key);
  box.append(a, b, c);

  const caps = [c1, c2, c3, c4];
  const finish = () => {
    needle.setAttribute("x1", x(q.count)); needle.setAttribute("x2", x(q.count)); dot.setAttribute("cx", x(q.count));
    if (seg) seg.classList.add("on");
    rank(1); map.fill(1); time.textContent = ms(s.us);
    caps.forEach((e) => e.classList.add("lit"));
  };
  const play = () => {
    caps.forEach((e) => e.classList.remove("lit"));
    rank(0);
    time.textContent = "";
    c1.classList.add("lit");
    tween(800, 0, (p) => {
      const v = 10 ** (Math.log10(Math.max(1, q.count)) * easeOut(p)), px = x(v);
      needle.setAttribute("x1", px); needle.setAttribute("x2", px); dot.setAttribute("cx", px);
      c1.querySelector(".mtext").textContent = count(p >= 1 ? q.count : v);
    });
    later(() => { if (seg) seg.classList.add("on"); c2.classList.add("lit"); }, 900);
    later(() => caps[2].classList.add("lit"), 1400);
    tween(800, 1400, (p) => rank(easeOut(p)));
    const go = 2300, dk = (s.us / 1000) * factor;
    later(() => c4.classList.add("lit"), go);
    tween(dk, go, (p) => { map.fill(p); time.textContent = ms(s.us * p); });
    return go;
  };
  return { play, finish };
}

const RACE_MS = 2400;
// The same lookup on each system, on one slowed time axis.
function renderRace(q) {
  const box = $("race"); box.replaceChildren(); box.hidden = false;
  const sys = q.systems[state.target] || {}, cap = current().capacity || {};
  const slow = Math.max(...Object.values(sys).filter(Boolean).map((v) => v.us));
  const factor = RACE_MS / (slow / 1000);
  const lanes = [];
  box.append(el("span", "label", t("raceH")));
  [["kuro", SYS], ["pf", BASELINES.pf], ["faiss", BASELINES.faiss]].forEach(([k, name]) => {
    const lane = el("div", `lane ${k}`); lane.append(el("span", "lane-name", name));
    const s = sys[k];
    if (s) {
      const track = el("div", "lane-track"), fill = el("i", "lane-fill"), val = el("span", "lane-val");
      track.append(fill); lane.append(track, val);
      lanes.push({ s, lane, fill, val, w: (100 * s.us) / slow, dur: (s.us / 1000) * factor });
    } else {
      const atMost = cap[k]?.[q.band]?.[state.target]?.at_most;
      lane.append(el("span", "lane-none", t("notReached", { r: atMost === undefined ? "–" : atMost.toFixed(3) })));
    }
    box.append(lane);
  });
  const leadBox = el("p", "lead"); leadBox.hidden = true;
  box.append(leadBox, el("p", "race-note", t("raceNote", { f: fmt(factor) })));
  const lead = lookupLead(sys), fastest = Math.min(...lanes.map((x) => x.s.us));
  const leadAt = lead && !lead.alone ? (sys[lead.next].us / 1000) * factor : (sys.kuro.us / 1000) * factor;
  const finish = (x) => {
    x.fill.style.width = `${x.w}%`;
    x.val.textContent = [ms(x.s.us), t("readsVal", { r: fmt(x.s.reads) }), `${t("thRecall")} ${x.s.recall.toFixed(2)}`].join(" · ");
    if (x.s.us === fastest) x.lane.classList.add("won");
  };
  const run = (delay) => {
    lanes.forEach((x) => {
      x.val.textContent = ms(0);
      tween(x.dur, delay, (p) => { x.fill.style.width = `${x.w * p}%`; x.val.textContent = ms(x.s.us * p); if (p >= 1) finish(x); });
    });
    later(() => showLead(leadBox, lead), delay + leadAt + 80);
  };
  return { factor, run, finishAll: () => { lanes.forEach(finish); showLead(leadBox, lead); } };
}

// Replays one lookup.
function replayLookup() {
  const set = current(), q = set.questions[state.index[state.tab]];
  const s = q.systems[state.target].kuro, plan = planText(s.plan);
  $("run-head").replaceChildren(state.tab === "rec" ? chips(q.filter) : document.createTextNode(q.q));
  $("load-control").hidden = true;
  $("target-control").hidden = false; $("compare").parentElement.hidden = true; $("answer").hidden = true;
  renderTargets();
  const race = renderRace(q);
  renderResults(set, s.hits);
  const list = $("results"), papers = [...list.children];
  const reveal = () => papers.forEach((li, i) => later(() => li.classList.add("in"), i * 70));
  const steps = $("steps"); steps.replaceChildren();
  if (set.plans) {
    steps.hidden = true;
    const m = renderMechanism(q, s, set, race.factor);
    if (still()) { m.finish(); race.finishAll(); return; }
    list.classList.add("wait");
    const go = m.play();
    race.run(go);
    later(reveal, go + (s.us / 1000) * race.factor);
    return;
  }
  $("mech").hidden = true; steps.hidden = false;
  const countNote = (c) => t("stepCountNote", { c: fmt(c), n: millions(set.size) });
  const a = step(t("stepCount"), "", countNote(q.count));
  const b = step(t("stepPlan"), "", plan.text);
  const c = step(t("stepRead"), ms(s.us), t("stepReadNote", { r: fmt(s.reads), pool: fmt(plan.pool), recall: s.recall.toFixed(2) }));
  const map = readMap(s.reads, plan.pool); c.append(map.grid, map.key);
  steps.append(a, b, c);
  if (still()) { [a, b, c].forEach((li) => li.classList.add("done")); map.fill(1); race.finishAll(); return; }
  list.classList.add("wait");
  const note = a.querySelector(".note"), time = c.querySelector(".time");
  a.classList.add("active");
  tween(450, 0, (p) => { note.textContent = countNote(q.count * easeOut(p)); });
  later(() => { a.classList.add("done"); b.classList.add("active"); }, 470);
  later(() => { b.classList.add("done"); c.classList.add("active"); }, 760);
  const go = 800, dk = (s.us / 1000) * race.factor;
  race.run(go);
  tween(dk, go, (p) => { map.fill(p); time.textContent = ms(s.us * p); });
  later(() => { c.classList.add("done"); reveal(); }, go + dk);
}

function renderResults(set, ids) {
  const list = $("results"); list.replaceChildren(); list.classList.remove("wait");
  $("results-h").textContent = t(state.tab === "search" ? "resultsSearch" : state.tab === "rec" ? "resultsRec" : "resultsRag");
  list.className = "results-list " + (state.tab === "rec" ? "items" : "papers");
  const note = $("results-note");
  note.hidden = state.tab !== "rec";
  note.textContent = state.tab === "rec" ? t("resultsRecNote") : "";
  ids.slice(0, state.tab === "rec" ? 10 : 5).forEach((id) => {
    const d = set.docs[id]; if (!d) return;
    const li = el("li");
    if (state.tab === "rec") {
      li.className = "item";
      if (d.image) { const im = el("img"); im.src = d.image; im.alt = ""; li.append(im); } else li.append(el("div", "noimg", t("noPicture")));
      li.append(el("div", "title", d.title), el("div", "meta", [d.price !== undefined ? `$${d.price.toFixed(2)}` : "", d.rating !== undefined ? `${d.rating.toFixed(1)} ★` : ""].filter(Boolean).join(" · ")));
    } else {
      const a = el("a", "title", d.title); a.href = `https://pubmed.ncbi.nlm.nih.gov/${d.pmid}/`; a.target = "_blank"; a.rel = "noopener";
      li.append(a, el("div", "meta", [d.year, d.journal].filter(Boolean).join(" · ")), el("p", "snippet", d.snippet));
    }
    list.append(li);
  });
}

// Recorded time of each question per system and rate in the load test.
function renderLoads(L) {
  const box = $("loads"); box.replaceChildren();
  [null, ...L.rates].forEach((r) => {
    const b = el("button", "", r === null ? t("loadIdle") : `${fmt(r)}`); b.type = "button"; b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", r === state.load ? "true" : "false");
    b.addEventListener("click", () => { state.load = r; replay(); });
    box.append(b);
  });
  $("load-note").textContent = state.load === null ? t("loadNoteIdle") : t("loadNote", { r: fmt(state.load) });
}

function renderLoadCompare(L, q) {
  const tb = $("compare"); tb.replaceChildren();
  const head = el("tr"); [t("thSystem"), t("thSearchLoad"), t("thWait")].forEach((x) => head.append(el("th", "", x))); tb.append(head);
  [["kuro", SYS], ["pf", BASELINES.pf], ["faiss", BASELINES.faiss]].forEach(([k, name]) => {
    const r = el("tr", k === "kuro" ? "me" : ""); r.append(el("td", "", name));
    const x = L.q[q.q]?.[k]?.[state.load];
    if (x) r.append(el("td", "", sec(x.lat / 1000)), el("td", "", sec(x.wait / 1000)));
    else { const td = el("td", "none", t("loadStopped", { r: fmt(L.stopped?.[k] ?? 0) })); td.colSpan = 2; r.append(td); }
    tb.append(r);
  });
}

function replayRag() {
  const set = DATA.rag, q = set.questions[state.index.rag], L = set.load;
  $("run-head").replaceChildren(document.createTextNode(q.q));
  $("load-control").hidden = !L;
  if (L) renderLoads(L);
  const at = L && state.load !== null ? L.q[q.q]?.kuro?.[state.load] : null;
  const steps = $("steps"); steps.replaceChildren();
  const pre = q.prefill_ms, dec = q.decode_ms, srch = at ? at.lat / 1000 : q.search_us / 1000, emb = q.embed_ms;
  const a = step(t("stepEmbed"), sec(emb));
  const v = { c: fmt(q.count), n: millions(set.size), r: fmt(q.reads), recall: q.recall.toFixed(2) };
  const b = step(t("stepSearch"), sec(srch), at ? t("stepSearchWait", { ...v, w: sec(at.wait / 1000) }) : t("stepSearchNote", v));
  const c = step(t("stepPrefill"), sec(pre), t("stepPrefillNote", { t: fmt(q.prompt_tokens), k: q.cited.length }));
  const d = step(t("stepAnswer"), sec(dec), t("stepAnswerNote", { t: fmt(q.answer_tokens) }));
  steps.append(a, b, c, d);
  $("target-control").hidden = true; $("race").hidden = true; $("mech").hidden = true; $("steps").hidden = false;
  $("compare").parentElement.hidden = !(L && state.load !== null);
  if (L && state.load !== null) renderLoadCompare(L, q);
  const box = $("answer"); box.hidden = false; box.replaceChildren();
  const stack = el("div", "stack"); const total = emb + srch + pre + dec;
  [["s-embed", emb], ["s-search", srch], ["s-prefill", pre], ["s-answer", dec]].forEach(([c2, v]) => { const i = el("i", c2); i.style.width = `${(100 * v) / total}%`; stack.append(i); });
  const text = el("p"); box.append(stack, text);
  reveal([a, b, c], [emb, srch, pre]);
  const words = q.answer.split(/(\s+)/);
  const perWord = dec / Math.max(1, words.length);
  const start = emb + srch + pre;
  words.forEach((w, i) => later(() => { text.textContent += w; if (i === words.length - 1) d.classList.add("done"); }, start + i * perWord));
  later(() => d.classList.add("active"), start);
  renderResults(set, q.cited);
}

function renderTargets() {
  ["targets", "targets2", "targets3"].forEach((id) => {
    const box = $(id); if (!box) return; box.replaceChildren();
    TARGETS.forEach((x) => {
      const b = el("button", "", (+x).toFixed(2)); b.type = "button"; b.setAttribute("role", "radio"); b.setAttribute("aria-checked", x === state.target ? "true" : "false");
      b.addEventListener("click", () => { state.target = x; replay(); });
      box.append(b);
    });
  });
}

function renderCapacity() {
  const cap = state.tab === "rag" ? null : current().capacity;
  const chart = $("cap-chart"); chart.replaceChildren();
  $("capacity").hidden = !cap;
  if (!cap) return;
  const all = [];
  Object.values(cap).forEach((sys) => Object.values(sys).forEach((b) => { const v = b[state.target]; if (v?.reached) all.push(v.qps); }));
  const lo = Math.log10(Math.min(...all) / 2), hi = Math.log10(Math.max(...all));
  const width = (q) => (100 * (Math.log10(q) - lo)) / (hi - lo);
  const leads = [];
  [0, 1, 2, 3, 4, 5].forEach((b) => {
    const name = t(`band${b}`);
    if (!cap.kuro?.[b]) return;
    const row = el("div", "cap-band"); row.append(el("span", "", name));
    const rows = el("div", "cap-rows");
    ["kuro", "pf", "faiss"].forEach((k) => {
      const v = cap[k]?.[b]?.[state.target]; if (!cap[k]?.[b]) return;
      const r = el("div", `cap-row ${k}`);
      if (v?.reached) {
        const f = el("div", `cap-fill ${k}`), w = `calc(${width(v.qps)}% * 0.78)`;
        if (still()) f.style.width = w; else { f.style.width = "0"; later(() => { f.style.width = w; }, 40 + 70 * b); }
        r.append(f, el("span", "cap-val", fmt(v.qps)));
        const x = k === "kuro" ? bandLead(cap, b, state.target) : null;
        if (x) { leads.push(x); r.append(el("span", "cap-x", times(x))); }
      } else {
        r.append(el("span", "cap-none", `${k === "pf" ? BASELINES.pf : k === "faiss" ? BASELINES.faiss : SYS}: ${t("notReached", { r: v ? v.at_most.toFixed(3) : "–" })}`));
      }
      rows.append(r);
    });
    row.append(rows); chart.append(row);
  });
  const head = $("cap-head"); head.replaceChildren();
  head.hidden = !leads.length;
  if (leads.length) {
    const geo = Math.exp(leads.reduce((s, x) => s + Math.log(x), 0) / leads.length), big = el("span", "x");
    head.append(big, el("span", "why", t("capWhy", { m: times(Math.max(...leads)) })));
    countTo(big, geo, 900);
  }
  $("cap-foot").textContent = t("capFoot");
}

function selectTab(tab) {
  clearTimers();
  state.tab = tab;
  try { if (location.hash.slice(1) !== tab) history.replaceState(null, "", `#${tab}`); } catch (e) {}
  document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === tab ? "true" : "false"));
  $("pick-label").textContent = t(tab === "search" ? "pickSearch" : tab === "rec" ? "pickRec" : "pickRag");
  const set = current();
  const shop = tab === "rec" && !!DATA.rec?.shop;
  $("shop").hidden = !shop; $("grid").hidden = shop; $("results-section").hidden = shop;
  $("tab-lede").textContent = t(tab === "search" ? "ledeSearch" : tab === "rec" ? "ledeRec" : "ledeRag", { n: tab === "rec" ? millions(set.size) : "" });
  if (!shop) renderExamples();
  replay();
  const foot = $("rag-foot");
  foot.hidden = tab !== "rag";
  if (tab === "rag") foot.textContent = t("ragFoot", { a: DATA.rag.rates.prefill_base_ms, b: DATA.rag.rates.prefill_per_token_ms, d: DATA.rag.rates.decode_tok_s });
}

async function start() {
  const params = new URLSearchParams(location.search);
  lang = params.get("lang") === "ko" ? "ko" : "en";
  DATA = window.DEMO_DATA || (await (await fetch("data.json?v=b08d3eb252")).json());
  KO = DATA.ko || {};
  applyStatic();
  renderGlance();
  state.load = ragLoad;
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.hidden = !DATA[b.dataset.tab];
    b.addEventListener("click", () => selectTab(b.dataset.tab));
  });
  $("lang").hidden = Object.keys(KO).length === 0;
  $("lang").addEventListener("click", () => { lang = lang === "ko" ? "en" : "ko"; applyStatic(); renderGlance(); selectTab(state.tab); });
  $("theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
    applyStatic();
  });
  const wanted = location.hash.slice(1);
  selectTab(DATA[wanted] ? wanted : "search");
}
start();
