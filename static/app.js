const SkySentinel = (() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function splash() {
    const s = $("splash"); if (!s) return;
    let seen = false; try { seen = !!sessionStorage.getItem("ss_entered"); } catch {}
    if (seen) { s.remove(); return; }
    const bar = $("splashBar");
    requestAnimationFrame(() => { bar.style.transition = "width 2.2s linear"; bar.style.width = "100%"; });
    setTimeout(() => { s.classList.add("out"); try { sessionStorage.setItem("ss_entered", "1"); } catch {}
      setTimeout(() => s.remove(), 600); }, 2300);
  }

  // ---- detection-speed selector (swaps model weights) ----
  async function initEngine(onBusy) {
    const box = $("engine"); if (!box) return;
    let info; try { info = await (await fetch("/api/engine")).json(); } catch { return; }
    box.innerHTML = info.options.map((o) => `<button data-e="${o.id}">${esc(o.label)}</button>`).join("");
    const mark = (id) => [...box.children].forEach((b) => b.classList.toggle("active", b.dataset.e === id));
    mark(info.engine);
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
      mark(b.dataset.e);
      onBusy && onBusy(true, "Loading model…");
      try { await fetch("/api/engine", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: b.dataset.e }) }); } catch {}
      onBusy && onBusy(false, "");
    }));
  }

  async function initSensitivity(onChange) {
    const el = $("sens"); if (!el) return;
    try { el.value = (await (await fetch("/api/controls")).json()).confidence; } catch {}
    const push = () => fetch("/api/controls", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence: parseFloat(el.value) }) }).catch(() => {});
    el.addEventListener("change", push);
  }

  // ---- siren ----
  let soundOn = true, actx = null, osc = null, lfo = null, sirenOn = false;
  function siren(on) {
    if (on) {
      if (sirenOn || !soundOn || !actx) return;
      if (actx.state === "suspended") actx.resume();
      osc = actx.createOscillator(); osc.type = "sawtooth"; osc.frequency.value = 760;
      const g = actx.createGain(); g.gain.value = 0.1;
      lfo = actx.createOscillator(); lfo.frequency.value = 4;
      const lg = actx.createGain(); lg.gain.value = 200;
      lfo.connect(lg); lg.connect(osc.frequency); osc.connect(g); g.connect(actx.destination);
      osc.start(); lfo.start(); sirenOn = true;
    } else if (sirenOn) { try { osc.stop(); lfo.stop(); } catch {} sirenOn = false; }
  }

  // ===================================================== UPLOAD
  function initUpload() {
    splash();
    initEngine((busy, msg) => { if (busy) setBar(0, msg); });
    initSensitivity();
    const drop = $("drop"), file = $("file"), go = $("go"), fname = $("fname");
    let chosen = null;
    drop.addEventListener("click", () => file.click());
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files.length) pick(e.dataTransfer.files[0]); });
    file.addEventListener("change", () => file.files.length && pick(file.files[0]));
    function pick(f) { chosen = f; fname.textContent = `${f.name} (${(f.size / 1048576).toFixed(1)} MB)`; go.disabled = false; }

    go.addEventListener("click", async () => {
      if (!chosen) return;
      go.disabled = true; $("prog").classList.remove("hidden"); $("result").classList.add("hidden");
      const fd = new FormData(); fd.append("video", chosen);
      fd.append("engine", [...$("engine").children].find((b) => b.classList.contains("active"))?.dataset.e || "");
      fd.append("confidence", $("sens").value);
      let res; try { res = await (await fetch("/api/upload", { method: "POST", body: fd })).json(); } catch { return fail("Upload failed"); }
      if (res.error) return fail(res.error);
      poll(res.job_id);
    });
    function poll(id) {
      const t = setInterval(async () => {
        let j; try { j = await (await fetch(`/api/job/${id}`)).json(); } catch { return; }
        if (j.status === "processing") setBar(j.progress, `Checking… ${j.progress}%`);
        else if (j.status === "queued") setBar(3, "Starting…");
        else if (j.status === "transcoding") setBar(97, "Preparing video…");
        else if (j.status === "done") { clearInterval(t); setBar(100, "Done"); show(id, j.image); }
        else if (j.status === "error") { clearInterval(t); fail(j.error || "Could not process the file"); }
      }, 700);
    }
    function show(id, isImage) {
      $("result").classList.remove("hidden");
      const player = $("player"), photo = $("photo");
      if (isImage) {
        const url = `/image/${id}?t=${Date.now()}`;
        photo.src = url; photo.classList.remove("hidden");
        player.classList.add("hidden"); player.removeAttribute("src");
        $("dl").href = url;
      } else {
        const url = `/video/${id}?t=${Date.now()}`;
        player.src = url; player.classList.remove("hidden");
        photo.classList.add("hidden"); photo.removeAttribute("src");
        $("dl").href = url;
      }
    }
    function setBar(p, m) { $("fill").style.width = p + "%"; $("fill").style.background = ""; $("status").textContent = m; }
    function fail(m) { setBar(100, m); $("fill").style.background = "var(--alarm)"; go.disabled = false; }
    $("again").addEventListener("click", (e) => { e.preventDefault(); $("result").classList.add("hidden"); $("prog").classList.add("hidden"); go.disabled = false; });
  }

  // enlarge a live feed into a full-screen player
  function openPlayer(id, name) {
    const ov = document.createElement("div");
    ov.className = "player-ov";
    ov.innerHTML = `<div class="player-box"><div class="player-top"><span>${esc(name || "Camera")}</span>
      <span class="player-close">✕</span></div><img src="/stream/${id}?t=${Date.now()}" alt=""></div>`;
    const close = () => ov.remove();
    ov.addEventListener("click", (e) => { if (e.target === ov || e.target.classList.contains("player-close")) close(); });
    document.addEventListener("keydown", function esc2(e) { if (e.key === "Escape") { close(); document.removeEventListener("keydown", esc2); } });
    document.body.appendChild(ov);
  }

  // ===================================================== LIVE
  function initLive() {
    splash();
    initEngine((busy, msg) => { $("engineNote").textContent = busy ? msg : ""; });
    initSensitivity();

    $("sound").addEventListener("click", () => {
      soundOn = !soundOn; $("sound").textContent = soundOn ? "🔊" : "🔇";
      if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
      actx.resume(); if (!soundOn) siren(false);
    });

    // network discovery
    $("scan").addEventListener("click", async () => {
      if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
      $("found").innerHTML = "";
      $("scanState").textContent = "Scanning the network…";
      await fetch("/api/discover", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user: $("user").value.trim(), password: $("pass").value }) }).catch(() => {});
      pollDiscover();
    });
    function pollDiscover() {
      const t = setInterval(async () => {
        let s; try { s = await (await fetch("/api/discover")).json(); } catch { return; }
        $("found").innerHTML = s.cameras.map((c) => `
          <div class="finditem">
            <span><span class="dot ${c.ok ? "on" : "off"}"></span>${esc(c.ip)} · ${esc(c.name)}</span>
            ${c.ok ? `<button class="btn btn-sm" data-token="${c.token}" data-ip="${esc(c.ip)}">Add</button>`
                   : `<span class="muted">needs login</span>`}</div>`).join("");
        $("found").querySelectorAll("button[data-token]").forEach((b) => b.addEventListener("click", async () => {
          b.disabled = true; b.textContent = "Added";
          await fetch("/api/cameras", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token: b.dataset.token, name: b.dataset.ip }) });
          refresh();
        }));
        if (!s.scanning) { clearInterval(t); $("scanState").textContent = s.message || (s.cameras.length ? "" : "No cameras found."); }
      }, 900);
    }

    $("add").addEventListener("click", async () => {
      const source = $("camSrc").value.trim(); if (!source) return;
      if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
      await fetch("/api/cameras", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, name: $("camName").value.trim() }) });
      $("camSrc").value = ""; $("camName").value = ""; refresh();
    });

    const grid = $("grid");
    async function refresh() {
      let cams = []; try { cams = await (await fetch("/api/cameras")).json(); } catch { return; }
      if (!cams.length) { grid.innerHTML = '<div class="empty">No cameras connected yet.</div>'; siren(false); return; }
      const ids = cams.map((c) => c.id); let alarm = false;
      cams.forEach((c) => {
        if (c.targets > 0) alarm = true;
        let el = $("cam-" + c.id);
        if (!el) {
          el = document.createElement("div"); el.className = "cam"; el.id = "cam-" + c.id;
          el.innerHTML = `<img class="feed" src="/stream/${c.id}" alt="">
            <div class="meta"><span class="nm"></span><span class="st"></span>
            <span class="x" data-id="${c.id}">remove</span></div>`;
          el.querySelector(".feed").addEventListener("click", () => openPlayer(c.id, el.querySelector(".nm").textContent));
          el.querySelector(".x").addEventListener("click", async (e) => {
            e.stopPropagation();
            await fetch("/api/cameras/" + e.target.dataset.id, { method: "DELETE" }); el.remove(); });
          grid.appendChild(el);
        }
        el.classList.toggle("alarm", c.targets > 0);
        el.querySelector(".nm").textContent = c.name || c.id;
        const tg = c.targets > 0 ? `<span class="alarm-txt">${c.targets} drone${c.targets > 1 ? "s" : ""}</span>` : "clear";
        el.querySelector(".st").innerHTML = `<span class="dot ${c.online ? "on" : "off"}"></span>${c.fps} fps · ${tg}`;
      });
      [...grid.querySelectorAll(".cam")].forEach((el) => { if (!ids.includes(el.id.replace("cam-", ""))) el.remove(); });
      siren(alarm);
    }
    refresh(); setInterval(refresh, 1200);
  }

  return { initUpload, initLive };
})();
