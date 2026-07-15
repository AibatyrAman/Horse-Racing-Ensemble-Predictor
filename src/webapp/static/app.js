/* TJK Tahmin Paneli — vanilla JS (fetch + SSE, framework yok) */
"use strict";

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
const api = (p) => fetch(p).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
const pct = (v, nd = 1) => v == null ? "—" : (v * 100).toFixed(nd) + "%";
const esc = (s) => s == null ? "" : String(s)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

const SERIES = {
  full:  { label: "full (oranlı)",  cssVar: "--series-full" },
  abl:   { label: "abl (oransız)",  cssVar: "--series-abl" },
  blend: { label: "blend (harman)", cssVar: "--series-blend" },
};
const cssColor = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/* ── Sekmeler ─────────────────────────────────────────────────────────── */
const loaded = {};
$("#tabs").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  if (!btn) return;
  $$("#tabs button").forEach(b => b.classList.toggle("active", b === btn));
  $$(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + btn.dataset.tab));
  loadTab(btn.dataset.tab);
});

function loadTab(name) {
  // Veri sekmeleri her ziyarette tazelenir (sunucu tarafı mtime-önbellekli, ucuz).
  // Yalnız İşlemler bir kez kurulur: SSE bağlantısı yeniden kurulmamalı.
  if (name === "islemler") {
    if (loaded[name]) return;
    loaded[name] = true;
  }
  ({ bugun: loadBugun, performans: loadPerf, strateji: loadStrateji,
     modeller: loadModeller, islemler: loadJobs })[name]();
}

/* ── BUGÜN ────────────────────────────────────────────────────────────── */
async function loadBugun(date) {
  const q = date ? `api/predictions?date=${encodeURIComponent(date)}` : "api/predictions";
  let d;
  try { d = await api(q); }
  catch { $("#raceList").innerHTML = "<p class='muted'>Tahmin verisi yok — İşlemler'den 'Tahmin Üret' çalıştırın.</p>"; return; }

  const sel = $("#dateSel");
  sel.innerHTML = d.dates.map(x =>
    `<option value="${esc(x.value)}" ${x.value === d.date ? "selected" : ""}>${esc(x.label)}</option>`).join("");
  sel.onchange = () => { loadBugun(sel.value); };

  if (!d.dates.length) {
    $("#oddsTsInfo").textContent = "";
    $("#raceList").innerHTML = "<p class='muted'>Henüz tahmin üretilmedi. İşlemler sekmesinden " +
      "önce <b>Programı Çek</b>, ardından <b>Tahmin Üret</b> çalıştırın; işler bitince bu " +
      "sekmeye dönmeniz yeterli.</p>";
    return;
  }

  const ts = d.cities.flatMap(c => c.races.map(r => r.odds_ts)).filter(Boolean);
  $("#oddsTsInfo").textContent = ts.length
    ? `🕒 Oran çekimi: ${ts.sort()[0].slice(11, 16)}–${ts.sort().at(-1).slice(11, 16)} (TJK saati)` : "";

  if (!d.cities.length) { $("#raceList").innerHTML = "<p class='muted'>Bu tarihte tahmin yok.</p>"; return; }

  $("#raceList").innerHTML = d.cities.map(city => `
    <div class="city"><h2>📍 ${esc(city.name)}</h2>
      ${city.races.map(r => raceCard(r)).join("")}
    </div>`).join("");

  $("#raceList").onclick = (e) => {
    const head = e.target.closest(".race-head");
    if (head && !e.target.closest("a")) head.parentElement.classList.toggle("open");
  };
}

function raceCard(r) {
  const pickFull = r.runners.find(x => x.picks.includes("full"));
  const fav = r.runners.find(x => x.is_fav);
  const meta = [r.saat, r.yaris_turu, r.mesafe ? r.mesafe + "m" : null, r.n + " at"]
    .filter(Boolean).join(" · ");
  const res = r.result;
  const resBadge = res
    ? `<span class="chip ${res.hit_full ? "won" : "lost"}">🏁 Kazanan: ${esc(res.winner)} ${res.hit_full == null ? "" : (res.hit_full ? "✓" : "✗")}</span>`
    : "";
  return `<div class="race">
    <div class="race-head">
      <span class="no">${esc(r.kosu_id)}</span>
      <span class="meta">${esc(meta)}</span>
      <span class="pickline">
        Model: <b>${esc(pickFull ? pickFull.at : "—")}</b>
        ${r.surpriz ? '<span class="chip star">★ favori değil</span>' : ""}
        · Favori: ${esc(fav ? fav.at : "—")}
        ${resBadge}
      </span>
    </div>
    <div class="race-body"><div class="table-scroll"><table>
      <tr>${res ? '<th class="num">Snç</th>' : ""}<th>At</th><th>Jokey</th><th class="num">Gny</th>
          <th class="num">P full</th><th class="num">P abl</th><th class="num">P blend</th>
          <th class="num">Sk</th><th class="num">St</th></tr>
      ${r.runners.map(x => `
        <tr class="${x.sonuc === 1 ? "hl-winner" : (x.picks.includes("full") ? "hl-full" : "")}">
          ${res ? `<td class="num">${x.sonuc ?? "—"}</td>` : ""}
          <td>${x.url ? `<a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.at)}</a>` : esc(x.at)}
              ${x.is_fav ? '<span class="chip">favori</span>' : ""}
              ${x.picks.map(p => `<span class="chip">${p}</span>`).join("")}</td>
          <td>${esc(x.jokey || "—")}</td>
          <td class="num">${x.ganyan ?? "—"}</td>
          <td class="num">${pct(x.p_full)}</td>
          <td class="num">${pct(x.p_abl)}</td>
          <td class="num">${pct(x.p_blend)}</td>
          <td class="num">${esc(x.siklet ?? "—")}</td>
          <td class="num">${esc(x.start ?? "—")}</td>
        </tr>`).join("")}
    </table></div></div>
  </div>`;
}

/* ── PERFORMANS ───────────────────────────────────────────────────────── */
async function loadPerf() {
  let d;
  try { d = await api("api/performance"); }
  catch { $("#perfCards").innerHTML = "<p class='muted'>Henüz forward-test verisi yok.</p>"; return; }

  $("#perfCards").innerHTML = d.cumulative.map(r => `
    <div class="card">
      <h4><span class="sw" style="background:var(${SERIES[r.variant]?.cssVar || "--accent"});display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px"></span>${esc(SERIES[r.variant]?.label || r.variant)}</h4>
      <div class="big">${pct(r.p1_winner)}</div>
      <div class="sub">P@1 kazanan · ${r.n_races} yarış</div>
      <div class="sub">ROI: <b>${r.roi == null ? "—" : (r.roi * 100).toFixed(1) + "%"}</b> ·
                       P@3 (tabela): ${pct(r.p3_top3)}</div>
    </div>`).join("") || "<p class='muted'>Henüz kümülatif veri yok.</p>";

  drawPerfChart(d.daily);

  const rows = d.daily.slice().reverse();
  $("#perfTable").innerHTML = `
    <tr><th>Tarih</th><th>Varyant</th><th class="num">Yarış</th>
        <th class="num">P@1 kazanan</th><th class="num">ROI</th><th class="num">P@3 tabela</th></tr>` +
    rows.map(r => `<tr><td>${esc(r.tarih)}</td><td>${esc(r.variant)}</td>
      <td class="num">${r.n_races}</td><td class="num">${pct(r.p1_winner)}</td>
      <td class="num">${r.roi == null ? "—" : (r.roi * 100).toFixed(1) + "%"}</td>
      <td class="num">${pct(r.p3_top3)}</td></tr>`).join("");
}

function drawPerfChart(daily) {
  const wrap = $("#perfChart");
  const dates = [...new Set(daily.map(r => r.tarih))].sort(
    (a, b) => a.split(".").reverse().join("") < b.split(".").reverse().join("") ? -1 : 1);
  if (dates.length < 2) { wrap.innerHTML = "<p class='muted'>Trend için en az 2 gün gerekir.</p>"; return; }

  const variants = Object.keys(SERIES).filter(v => daily.some(r => r.variant === v && r.p1_winner != null));
  const W = 860, H = 300, m = { t: 16, r: 110, b: 34, l: 46 };
  const x = i => m.l + i * (W - m.l - m.r) / (dates.length - 1);
  const y = v => m.t + (1 - v) * (H - m.t - m.b);

  let g = "";
  for (const gy of [0, .25, .5, .75, 1])
    g += `<line x1="${m.l}" y1="${y(gy)}" x2="${W - m.r}" y2="${y(gy)}" stroke="var(--border)" stroke-width="1"/>
          <text x="${m.l - 8}" y="${y(gy) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${gy * 100}%</text>`;
  const step = Math.ceil(dates.length / 8);
  dates.forEach((dt, i) => {
    if (i % step === 0 || i === dates.length - 1)
      g += `<text x="${x(i)}" y="${H - 10}" text-anchor="middle" font-size="10.5" fill="var(--muted)">${dt.slice(0, 5)}</text>`;
  });

  let lines = "", labels = "";
  const byVar = {};
  for (const v of variants) {
    const pts = dates.map((dt, i) => {
      const row = daily.find(r => r.tarih === dt && r.variant === v);
      return row && row.p1_winner != null ? [i, row.p1_winner] : null;
    }).filter(Boolean);
    byVar[v] = new Map(pts);
    if (pts.length < 2) continue;
    const color = `var(${SERIES[v].cssVar})`;
    lines += `<polyline fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"
      points="${pts.map(([i, val]) => `${x(i)},${y(val)}`).join(" ")}"/>`;
    const last = pts.at(-1);
    labels += `<text x="${x(last[0]) + 8}" y="${y(last[1]) + 4}" font-size="12" font-weight="600"
      fill="var(--text-2)">${v} ${pct(last[1], 0)}</text>
      <circle cx="${x(last[0])}" cy="${y(last[1])}" r="3.5" fill="${color}" stroke="var(--panel)" stroke-width="2"/>`;
  }

  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Günlük P@1 trendi">
      ${g}${lines}${labels}
      <line id="xh" x1="0" y1="${m.t}" x2="0" y2="${H - m.b}" stroke="var(--muted)" stroke-width="1" opacity="0"/>
    </svg>
    <div class="viz-tooltip" id="perfTip"></div>
    <div class="legend">${variants.map(v =>
      `<span><span class="sw" style="background:var(${SERIES[v].cssVar})"></span>${SERIES[v].label}</span>`).join("")}</div>`;

  const svg = wrap.querySelector("svg"), tip = $("#perfTip"), xh = $("#xh", wrap);
  svg.addEventListener("mousemove", (e) => {
    const rect = svg.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * W / rect.width;
    const i = Math.max(0, Math.min(dates.length - 1,
      Math.round((sx - m.l) / ((W - m.l - m.r) / (dates.length - 1)))));
    xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i)); xh.setAttribute("opacity", ".5");
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX - rect.left + 14, rect.width - 150) + "px";
    tip.style.top = (e.clientY - rect.top - 10) + "px";
    tip.innerHTML = `<b>${dates[i]}</b><br>` + variants.map(v => {
      const val = byVar[v].get(i);
      return `<span class="sw" style="background:var(${SERIES[v].cssVar})"></span>${v}: ${val == null ? "—" : pct(val)}`;
    }).join("<br>");
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; xh.setAttribute("opacity", "0"); });
}

/* ── STRATEJİ ─────────────────────────────────────────────────────────── */
async function loadStrateji(betType) {
  const q = betType ? `api/strategy?bet_type=${encodeURIComponent(betType)}` : "api/strategy";
  const d = await api(q).catch(() => null);
  if (!d) return;

  if (d.bets_name) $("#betsTitle").textContent = "Günün önerileri — " + d.bets_name.replace("bets_", "").replace(".md", "");
  $("#betsMd").innerHTML = d.bets_md ? mdRender(d.bets_md) : "<p class='muted'>Henüz öneri üretilmedi (İşlemler → Strateji Üret).</p>";
  $("#stratSummary").innerHTML = d.summary_md ? mdRender(d.summary_md) : "<p class='muted'>Backtest özeti yok.</p>";

  // Önceki önerilerin sonuçları (bets_track.csv)
  if (d.track && d.track.length) {
    $("#trackPanel").hidden = false;
    const s = d.track_summary;
    $("#trackSummary").textContent = s
      ? `${s.n_total} öneri · ${s.n_resolved} sonuçlandı · ${s.n_won} tuttu` +
        (s.gany_n ? ` · Ganyan: ${s.gany_won}/${s.gany_n} tuttu` +
          (s.gany_roi == null ? "" : `, gerçek ROI ${(s.gany_roi * 100).toFixed(1)}%`) : "")
      : "";
    $("#trackTable").innerHTML = `
      <tr><th>Tarih</th><th>Şehir</th><th>Koşu</th><th>Tür</th><th>Atlar</th>
          <th class="num">P model</th><th class="num">EV</th><th class="num">Pay</th>
          <th>Sonuç</th><th class="num">Kâr</th></tr>` +
      d.track.map(r => `<tr>
        <td>${esc(r.tarih)}</td><td>${esc(r.sehir)}</td><td>${esc(r.kosu)}</td>
        <td>${esc(r.bet_type)}</td><td>${esc(r.horses)}</td>
        <td class="num">${pct(r.p_model)}</td>
        <td class="num">${r.ev == null ? "—" : (r.ev * 100).toFixed(0) + "%"}</td>
        <td class="num">${r.stake == null ? "—" : r.stake + " TL"}</td>
        <td class="${r.status === "won" ? "hit-1" : r.status === "lost" ? "hit-0" : "muted"}">
            ${r.status === "won" ? "✓ tuttu" : r.status === "lost" ? "✗" : "bekliyor"}</td>
        <td class="num">${r.profit == null ? "—" : r.profit.toFixed(1) + " TL"}</td></tr>`).join("");
  }

  const sel = $("#betTypeSel");
  if (sel.options.length <= 1 && d.bet_types.length) {
    d.bet_types.forEach(t => sel.add(new Option(t, t)));
    sel.onchange = () => loadStrateji(sel.value || null);
  }

  $("#btTable").innerHTML = d.backtest.length ? `
    <tr><th>Tarih</th><th>Tür</th><th>Kombinasyon</th>
        <th class="num">P model</th><th class="num">P piyasa</th><th class="num">EV</th><th>Sonuç</th></tr>` +
    d.backtest.map(r => `<tr>
      <td>${esc(r.tarih)}</td><td>${esc(r.bet_type)}</td><td>${esc(r.combo)}</td>
      <td class="num">${pct(r.p_model)}</td><td class="num">${pct(r.p_market)}</td>
      <td class="num">${r.ev == null ? "—" : (r.ev * 100).toFixed(0) + "%"}</td>
      <td class="${r.hit ? "hit-1" : "hit-0"}">${r.hit == null ? "—" : r.hit ? "✓ tuttu" : "✗"}</td></tr>`).join("")
    : "<tr><td class='muted'>Backtest verisi yok.</td></tr>";

  if (d.bankroll_png) { $("#bankrollPanel").hidden = false; $("#bankrollImg").src = "reports/bankroll_curve.png"; }
}

/* ── MODELLER ─────────────────────────────────────────────────────────── */
async function loadModeller() {
  const d = await api("api/models").catch(() => null);
  if (!d) return;

  const regCard = (title, reg) => !reg ? "" : Object.entries(reg).map(([t, r]) => `
    <div class="card"><h4>${title} · ${t === "Is_Winner" ? "Kazanan" : "Tabela"}</h4>
      <div class="big">${esc(r.model_name)}</div>
      <div class="sub">AUC ${r.cv_auc} · P@1 ${pct(r.precision_at_1)} · P@3 ${pct(r.precision_at_3)}</div>
      <div class="sub muted">Terfi: ${esc((r.promoted_at || "").slice(0, 10))}</div>
    </div>`).join("");
  $("#regCards").innerHTML = regCard("Full", d.registry_full) + regCard("Ablation", d.registry_abl);

  $("#cmpTable").innerHTML = d.comparison.length ? `
    <tr><th>Hedef</th><th>Model</th><th class="num">AUC</th>
        <th class="num">P@1</th><th class="num">P@3</th><th class="num">Value ROI</th></tr>` +
    d.comparison.map(r => `<tr><td>${esc(r.target)}</td><td>${esc(r.model)}</td>
      <td class="num">${r.auc ?? "—"}</td><td class="num">${pct(r.p1)}</td>
      <td class="num">${pct(r.p3)}</td>
      <td class="num">${r.roi_value == null ? "—" : (r.roi_value * 100).toFixed(1) + "%"}</td></tr>`).join("")
    : "<tr><td class='muted'>Karşılaştırma verisi yok.</td></tr>";

  $("#modelImgs").innerHTML = d.images.map(n =>
    `<a href="reports/${n}" target="_blank"><img src="reports/${n}" alt="${n}" loading="lazy"></a>`).join("");
}

/* ── İŞLEMLER ─────────────────────────────────────────────────────────── */
let evtSource = null, jobTimer = null;

async function loadJobs() {
  const d = await api("api/jobs").catch(() => null);
  if (!d) return;
  const anyRunning = Object.values(d).some(j => j.status === "running");
  const st = { idle: "boşta", running: "çalışıyor…", done: "tamamlandı", failed: "HATA" };

  $("#jobCards").innerHTML = Object.entries(d).map(([k, j]) => `
    <div class="card">
      <h4>${esc(j.label)} <span class="status ${j.status}">${st[j.status]}</span></h4>
      <div class="sub">${esc(j.desc)}</div>
      <div class="sub muted">${j.last_run ? "Son çalışma: " + j.last_run : ""}</div>
      <div class="job-actions">
        <button class="primary" data-start="${k}" ${anyRunning ? "disabled" : ""}>Başlat</button>
        <button data-log="${k}">Log</button>
      </div>
    </div>`).join("");

  $("#jobCards").onclick = async (e) => {
    const start = e.target.dataset.start, log = e.target.dataset.log;
    if (start) {
      const r = await fetch(`api/jobs/${start}/start`, { method: "POST" });
      if (!r.ok) alert((await r.json()).detail || "Başlatılamadı");
      watchLog(start);
      loadJobs();
    } else if (log) {
      watchLog(log);
    }
  };

  // Koşan iş varken kartları periyodik tazele
  clearTimeout(jobTimer);
  if (anyRunning) jobTimer = setTimeout(loadJobs, 5000);
}

function watchLog(key) {
  if (evtSource) evtSource.close();
  const view = $("#logView");
  $("#logTitle").textContent = "Log — " + key;
  view.textContent = "";
  // Önce mevcut logu getir, sonra canlı akışa geç
  api(`api/jobs/${key}/log`).then(d => {
    view.textContent = d.log || "";
    view.scrollTop = view.scrollHeight;
    if (d.status !== "running") return; // bitmiş iş: akışa gerek yok
    evtSource = new EventSource(`api/jobs/${key}/stream`);
    evtSource.onmessage = (e) => {
      view.textContent += e.data + "\n";
      view.scrollTop = view.scrollHeight;
    };
    evtSource.addEventListener("done", (e) => {
      view.textContent += `\n── İş bitti (çıkış kodu ${e.data}) ──\n`;
      evtSource.close();
      loadJobs();
    });
  });
}

/* ── Mini markdown (başlık/tablo/bold/blockquote yeter) ───────────────── */
function mdRender(md) {
  const lines = md.split("\n");
  let html = "", tbl = [];
  const flushTbl = () => {
    if (!tbl.length) return;
    const rows = tbl.filter(l => !/^\s*\|[\s:-]+\|/.test(l));
    html += "<div class='table-scroll'><table>" + rows.map((l, i) => {
      const cells = l.split("|").slice(1, -1).map(c => inline(c.trim()));
      const tag = i === 0 ? "th" : "td";
      return "<tr>" + cells.map(c => `<${tag}>${c}</${tag}>`).join("") + "</tr>";
    }).join("") + "</table></div>";
    tbl = [];
  };
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
  for (const l of lines) {
    if (/^\s*\|/.test(l)) { tbl.push(l); continue; }
    flushTbl();
    if (/^#{1,3}\s/.test(l)) html += `<h3>${inline(l.replace(/^#+\s*/, ""))}</h3>`;
    else if (/^>\s?/.test(l)) html += `<blockquote>${inline(l.replace(/^>\s?/, ""))}</blockquote>`;
    else if (l.trim() === "" ) html += "";
    else html += `<p>${inline(l)}</p>`;
  }
  flushTbl();
  return html;
}

/* ── Başlat ───────────────────────────────────────────────────────────── */
loadTab("bugun");
