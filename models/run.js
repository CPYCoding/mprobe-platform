/* ============================================================
   MProbe — shared behavior for model "Run" pages.
   Reads the `window.MODEL` config defined in each page's HTML.
   Colleagues: to change a page's CONTENT (title, samples, text),
   edit that page's .html — not this file.
   ============================================================ */
(function(){
  var M = window.MODEL || {};
  var app = document.getElementById("app");
  if(!app) return;
  var $ = function(s,c){return (c||document).querySelector(s);};
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Show the "Run via API" strip on run pages. Off for now — set to true to bring it back.
  var SHOW_API = false;

  var ICON = {
    detector:'<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2 3 7v6c0 5 4 8 9 9 5-1 9-4 9-9V7z"/><path d="m9 12 2 2 4-4"/></svg>',
    cleaning:'<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z"/><path d="M4 7v10c0 1.7 3.6 3 8 3s8-1.3 8-3V7"/><path d="m8 13 2.5 2.5L16 10"/></svg>',
    up:'<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>'
  };
  var isClean = M.kind === "cleaning";
  var icon = isClean ? ICON.cleaning : ICON.detector;
  var apiInput = isClean ? '-F dataset=@data.zip' : '-F input=@frame.jpg';

  var header = '<div class="run-hd"><div class="thumb th-'+(isClean?"cleaning":"detector")+'">'+icon+'</div>'+
    '<div><div class="run-title">Run · '+(M.title||"Model")+'</div><div class="run-sub">'+(M.arch||'')+' · '+(M.dataset||'')+' · your model</div></div>'+
    '<span class="run-owned">✓ usage rights</span></div>';
  var intro = M.intro ? '<p class="run-intro">'+M.intro+'</p>' : '';
  var apiStrip = '<div class="api-strip"><div><div class="k">Run via API</div>'+
    '<code>curl -X POST https://api.mprobe.ai/v1/models/'+(M.slug||"model")+'/run -H "Authorization: Bearer mpb_live_…" '+apiInput+'</code></div>'+
    '<div class="quota">340 / 1,000 runs this month · Team plan</div></div>';

  app.innerHTML = header + intro + (SHOW_API ? apiStrip : "") + '<div id="host"></div>';
  var host = $("#host");

  if(isClean){ renderCleaning(); }
  else { renderDetector("single"); }

  /* ===================== DETECTOR ===================== */
  function renderDetector(mode){
    var isBatch = mode === "batch";
    host.innerHTML =
      '<div class="run-wrap">'+
        '<div class="seg" id="dmode"><button data-m="single" class="'+(!isBatch?"on":"")+'">Single frame</button><button data-m="batch" class="'+(isBatch?"on":"")+'">Upload dataset</button></div>'+
        (isBatch ? batchBody() : singleBody())+
        '<div id="result"></div>'+
      '</div>';
    [].forEach.call($("#dmode").children, function(b){ b.onclick = function(){ renderDetector(b.dataset.m); }; });
    if(isBatch) wireBatch(); else wireSingle();
  }

  function singleBody(){
    return '<div class="k-lab">Pick a sample frame</div>'+
      '<div class="frames" id="frames">'+ (M.samples||[]).map(function(f,i){
        return '<div class="frame" data-i="'+i+'"><div class="scene" style="background:'+f.bg+'">'+(f.emoji||'')+'</div><div class="box" id="box-'+i+'"><span class="bl"></span></div><div class="name">'+f.name+'</div></div>';
      }).join("")+'</div>'+
      '<button class="btn btn-primary runbtn" id="go" disabled>▶ Run inference</button>';
  }

  function batchBody(){
    return '<div class="k-lab">Upload a dataset — the model runs inference on every image</div>'+
      '<div class="dropzone" id="ds-drop" style="margin-top:10px">'+ICON.up+'<div class="big">Drop a dataset (.zip / folder of images)</div><div class="sm">or click to load a sample set</div></div>'+
      '<button class="btn btn-primary runbtn" id="ds-go" disabled style="margin-top:16px">▶ Run on dataset</button>';
  }

  /* ---- single-frame inference ---- */
  var sel = -1;
  function wireSingle(){
    sel = -1;
    [].forEach.call($("#frames").children, function(fr){
      fr.onclick = function(){
        sel = +fr.dataset.i;
        [].forEach.call($("#frames").children, function(o){ o.classList.toggle("sel", o===fr); });
        $("#go").disabled = false;
      };
    });
    $("#go").onclick = runSingle;
  }
  function runSingle(){
    var f = (M.samples||[])[sel]; if(!f) return;
    var res = $("#result");
    res.innerHTML = '<div style="font-family:var(--mono);font-size:13px;color:var(--ink-3);margin-top:18px">▷ running inference on '+f.name+' …</div>';
    setTimeout(function(){
      if(f.hit && f.box){
        var box = $("#box-"+sel);
        if(box){ box.style.top=f.box.t+"%"; box.style.left=f.box.l+"%"; box.style.width=f.box.w+"%"; box.style.height=f.box.h+"%"; box.querySelector(".bl").textContent=M.label+" "+f.conf+"%"; box.classList.add("show"); }
      }
      var protos = f.hit ? '<div style="margin-top:16px"><div class="k-lab">Why — top matching prototypes</div><div class="protos">'+
        [[2,Math.min(99,f.conf)],[5,Math.max(40,f.conf-18)],[7,Math.max(30,f.conf-31)]].map(function(p){
          return '<div class="proto-row"><span class="pn">proto '+p[0]+'</span><span class="pb"><i style="width:'+p[1]+'%"></i></span><span class="pv">'+(p[1]/100).toFixed(2)+'</span></div>';
        }).join("")+'</div></div>' : '';
      var cap = M.label ? M.label.charAt(0).toUpperCase()+M.label.slice(1) : "Target";
      var v = f.hit
        ? '<div class="verdict pass"><div class="vi">'+(M.emoji||'✓')+'</div><div><div class="vt">'+cap+' detected · '+f.conf+'% confidence</div><div class="vs">The hardened model fired above threshold. A backdoor trigger planted in this frame would not flip the result.</div></div></div>'
        : '<div class="verdict neutral"><div class="vi">—</div><div><div class="vt">No '+(M.label||"target")+' detected</div><div class="vs">Confidence below threshold. Nothing flagged in this frame.</div></div></div>';
      res.innerHTML = v + protos + '<div class="res-actions"><button class="btn btn-ghost" onclick="location.reload()">Run another frame</button></div>';
    }, reduce?0:700);
  }

  /* ---- batch inference on an uploaded dataset ---- */
  var dsCount = 0;
  function wireBatch(){
    dsCount = 0;
    var dz = $("#ds-drop"), go = $("#ds-go");
    dz.onclick = function(){
      dsCount = 24;
      dz.classList.add("loaded");
      dz.querySelector(".big").textContent = "my_dataset.zip · " + dsCount + " images";
      dz.querySelector(".sm").textContent = "ready to run inference";
      go.disabled = false;
    };
    go.onclick = runBatch;
  }
  function runBatch(){
    var res = $("#result");
    var n = dsCount || 24;
    var hitBg = (M.samples && M.samples[0] && M.samples[0].bg) || "linear-gradient(160deg,#2a3f5f,#16263f)";
    var items = [];
    for(var i=0;i<n;i++){
      var hit = Math.random() < 0.62;
      items.push({ hit:hit, conf: hit ? Math.round(72+Math.random()*27) : Math.round(20+Math.random()*35) });
    }
    var detected = items.filter(function(x){return x.hit;}).length;
    res.innerHTML = '<div class="prog"><i id="bprog"></i></div>'+
      '<div style="font-family:var(--mono);font-size:12.5px;color:var(--ink-3);margin-top:8px" id="bstat">running inference on '+n+' images…</div>';
    var prog = $("#bprog"), done = 0;
    (function tick(){
      done++;
      prog.style.width = Math.round(done/n*100)+"%";
      $("#bstat").textContent = "running inference… "+done+"/"+n;
      if(done>=n) return batchDone(items, detected, n, hitBg);
      setTimeout(tick, reduce?0:Math.min(70, 900/n));
    })();
  }
  function batchDone(items, detected, n, hitBg){
    var cap = M.label ? M.label.charAt(0).toUpperCase()+M.label.slice(1) : "Target";
    var grid = items.map(function(x){
      return '<div class="btile" style="background:'+(x.hit?hitBg:"linear-gradient(160deg,#2a2d33,#191c20)")+'">'+
        '<div class="bd" style="background:'+(x.hit?"var(--clean)":"var(--ink-3)")+'">'+(x.hit?"✓":"–")+'</div>'+
        '<div class="cf">'+x.conf+'%</div></div>';
    }).join("");
    $("#result").innerHTML =
      '<div class="verdict pass"><div class="vi">'+(M.emoji||'✓')+'</div><div><div class="vt">'+cap+' detected in '+detected+' of '+n+' images</div>'+
      '<div class="vs">Batch inference complete — '+Math.round(detected/n*100)+'% of the uploaded set contained a '+(M.label||"target")+'. The hardened model held; nothing was flipped by a trigger.</div></div></div>'+
      '<div class="batch-grid">'+grid+'</div>'+
      '<div class="res-actions"><button class="btn btn-primary" onclick="location.reload()">Run again</button><button class="btn btn-ghost" id="csv">Download results (CSV)</button></div>';
    var c = $("#csv"); if(c) c.onclick = function(){ c.textContent = "✓ Exported (demo)"; };
  }

  /* ===================== CLEANING ===================== */
  function renderCleaning(){
    if(M.live){ return renderLiveCleaning(); }
    host.innerHTML = '<div class="run-wrap"><div class="k-lab">Input dataset</div>'+
      '<div class="dropzone" id="drop" style="margin-top:10px">'+ICON.up+'<div class="big">Drop a labeled dataset (.zip / .npz)</div><div class="sm">or click to load a sample</div></div>'+
      '<button class="btn btn-primary runbtn" id="go" disabled style="margin-top:16px">▶ Run cleaning</button>'+
      '<div id="result"></div></div>';
    var dz = $("#drop"), go = $("#go");
    dz.onclick = function(){
      dz.classList.add("loaded");
      var total = (M.cleanStats && M.cleanStats.total) || 12480;
      dz.querySelector(".big").textContent = "sample_dataset.zip · " + total.toLocaleString() + " samples";
      dz.querySelector(".sm").textContent = "ready to clean";
      go.disabled = false;
    };
    go.onclick = runCleaning;
  }
  function runCleaning(){
    var res = $("#result");
    res.innerHTML = '<div class="prog"><i id="prog"></i></div><div class="console" id="con"></div>';
    var con = $("#con"), prog = $("#prog");
    var st = M.cleanStats || {total:12480, removed:214, triggers:137, flips:77};
    var kept = st.total - st.removed;
    var logs = [
      {t:"$ mprobe clean --dataset sample_dataset.zip", c:"dim"},
      {t:"[scan] "+st.total.toLocaleString()+" samples loaded"},
      {t:"[scan] detecting label-flips & backdoor triggers…"},
      {t:"[flag] "+st.triggers+" samples carry a backdoor trigger pattern", c:"bad"},
      {t:"[flag] "+st.flips+" samples have inconsistent labels", c:"warn"},
      {t:"[clean] removed "+st.removed+" samples · "+kept.toLocaleString()+" kept", c:"ok"},
      {t:"[done] cleaned set + audit report signed", c:"ok"}
    ];
    var i = 0;
    (function step(){
      if(i>=logs.length) return cleanDone(st, kept);
      var d = document.createElement("div"); d.className="ln "+(logs[i].c||""); d.textContent=logs[i].t;
      con.appendChild(d); con.scrollTop = con.scrollHeight;
      prog.style.width = Math.round((i+1)/logs.length*100)+"%"; i++;
      setTimeout(step, reduce?0:300);
    })();
  }
  function cleanDone(st, kept){
    $("#result").innerHTML =
      '<div class="verdict flag"><div class="vi">🧹</div><div><div class="vt">'+st.removed+' poisoned / mislabeled samples removed</div><div class="vs">Cleaned '+st.total.toLocaleString()+' → '+kept.toLocaleString()+' samples. Backdoor triggers and label-flips filtered out before training.</div></div></div>'+
      '<div class="res-metrics"><div class="m"><div class="k">Samples in</div><div class="v">'+st.total.toLocaleString()+'</div></div>'+
        '<div class="m"><div class="k">Removed</div><div class="v" style="color:var(--threat)">'+st.removed+'</div></div>'+
        '<div class="m"><div class="k">Triggers</div><div class="v">'+st.triggers+'</div></div>'+
        '<div class="m"><div class="k">Label-flips</div><div class="v">'+st.flips+'</div></div></div>'+
      '<div class="res-actions"><button class="btn btn-primary" onclick="location.reload()">Run again</button></div>';
  }

  /* ---- live cleaning: talks to the real backend (server/main.py) ---- */
  var liveFile = null, liveUseSample = false;
  function renderLiveCleaning(){
    host.innerHTML =
      '<div class="run-wrap"><div class="k-lab">Input dataset</div>'+
      '<div class="dropzone" id="drop" style="margin-top:10px">'+ICON.up+'<div class="big">Drop a labeled dataset (.zip)</div><div class="sm">or click to choose a .zip file</div></div>'+
      '<input type="file" id="file-input" accept=".zip" style="display:none">'+
      '<div style="margin-top:10px;font-size:12.5px;color:var(--ink-3)">or <a href="#" id="load-sample" style="color:var(--teal-ink);font-weight:600">load a sample dataset</a> to try it without your own files</div>'+
      '<button class="btn btn-primary runbtn" id="go" disabled style="margin-top:16px">▶ Run cleaning</button>'+
      '<div id="result"></div></div>';

    liveFile = null; liveUseSample = false;
    var dz = $("#drop"), go = $("#go"), input = $("#file-input"), sampleLink = $("#load-sample");

    function selectFile(f){
      liveUseSample = false; liveFile = f;
      dz.classList.add("loaded");
      dz.querySelector(".big").textContent = f.name;
      dz.querySelector(".sm").textContent = "ready to clean";
      go.disabled = false;
    }

    dz.onclick = function(){ input.click(); };
    dz.ondragover = function(e){ e.preventDefault(); dz.classList.add("loaded"); };
    dz.ondrop = function(e){
      e.preventDefault();
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if(f) selectFile(f);
    };
    input.onchange = function(){ if(input.files && input.files[0]) selectFile(input.files[0]); };
    sampleLink.onclick = function(e){
      e.preventDefault();
      liveUseSample = true; liveFile = null;
      dz.classList.add("loaded");
      dz.querySelector(".big").textContent = "sample_dataset.zip (bundled demo set)";
      dz.querySelector(".sm").textContent = "ready to run — mix of clean, triggered, and anomalous images";
      go.disabled = false;
    };
    go.onclick = runLiveCleaning;
  }

  function logLine(con, text, cls){
    var d = document.createElement("div");
    d.className = "ln " + (cls||"");
    d.textContent = text;
    con.appendChild(d);
    con.scrollTop = con.scrollHeight;
    return d;
  }

  function runLiveCleaning(){
    var res = $("#result");
    res.innerHTML = '<div class="prog" id="live-prog"><i></i></div><div class="console" id="live-con"></div>';
    var con = $("#live-con"), prog = $("#live-prog").firstElementChild;
    var base = (M.backend || "").replace(/\/$/, "");

    logLine(con, "$ mprobe clean --dataset " + (liveUseSample ? "sample_dataset.zip" : liveFile.name), "dim");
    logLine(con, "[upload] sending dataset to detection service…");
    prog.style.width = "20%";

    var req;
    if(liveUseSample){
      req = fetch(base + "/v1/models/dataset-cleaning/run-sample", { method: "POST" });
    } else {
      var formData = new FormData();
      formData.append("dataset", liveFile);
      req = fetch(base + "/v1/models/dataset-cleaning/run", { method: "POST", body: formData });
    }

    logLine(con, "[scan] stage 1 — materialized trigger detector (known signatures)…");
    prog.style.width = "50%";

    req.then(function(response){
        if(!response.ok){
          return response.json().catch(function(){ return {}; }).then(function(body){
            throw new Error(body.detail || ("Request failed (HTTP " + response.status + ")"));
          });
        }
        return response.json();
      })
      .then(function(data){
        logLine(con, "[scan] stage 2 — blind feature detector (unknown / novel patterns)…");
        prog.style.width = "85%";
        setTimeout(function(){
          logLine(con, "[done] cleaned set + report ready", "ok");
          prog.style.width = "100%";
          setTimeout(function(){ liveCleanDone(data, base); }, reduce?0:250);
        }, reduce?0:350);
      })
      .catch(function(err){
        logLine(con, "[error] " + err.message, "bad");
        res.insertAdjacentHTML("beforeend",
          '<div class="verdict flag" style="margin-top:12px"><div class="vi">!</div><div>'+
          '<div class="vt">Couldn’t reach the detection service</div>'+
          '<div class="vs">Make sure the backend is running (uvicorn server.main:app --reload) and reachable — see the log above for the exact error.</div>'+
          '</div></div>'+
          '<div class="res-actions"><button class="btn btn-ghost" onclick="location.reload()">Try again</button></div>');
      });
  }

  function liveCleanDone(data, base){
    var report = data.report || {};
    var detectors = report.detectors || [];
    var total = report.total_samples || 0;
    var autoRemoved = report.auto_removed_samples || 0;
    var pendingReview = report.pending_review_samples || 0;
    var kept = report.kept_samples != null ? report.kept_samples : (total - autoRemoved);

    var overall = '<div class="res-metrics" style="grid-template-columns:repeat(4,1fr)">'+
      '<div class="m"><div class="k">Samples in</div><div class="v">'+total.toLocaleString()+'</div></div>'+
      '<div class="m"><div class="k">Auto-removed</div><div class="v" style="color:var(--threat)">'+autoRemoved+'</div></div>'+
      '<div class="m"><div class="k">Pending review</div><div class="v" style="color:var(--amber)">'+pendingReview+'</div></div>'+
      '<div class="m"><div class="k">In cleaned set</div><div class="v" style="color:var(--clean)">'+kept.toLocaleString()+'</div></div>'+
      '</div>';

    // NOTE: materialized_trigger_detector's flagged_sample_ids is a plain
    // string array; blind_feature_detector's is an array of
    // {sample_id,score,level,flagged_by} objects. The two shapes are
    // intentionally different (auto_remove vs review carry different
    // authority) — branch on d.action, never guess the shape.
    var detectorCards = detectors.map(function(d, idx){
      var triggers = d.known_triggers_checked
        ? '<div style="font-size:12px;color:var(--ink-3);margin-top:4px">known triggers checked: '+d.known_triggers_checked.join(", ")+'</div>'
        : "";
      var isReview = d.action === "review";
      var actionTag = isReview
        ? '<span style="font-size:11px;font-weight:700;color:var(--amber)">flags for review — stays in cleaned set</span>'
        : '<span style="font-size:11px;font-weight:700;color:var(--threat)">removes from cleaned set</span>';
      var listHost = (isReview && d.flagged_count)
        ? '<div class="review-list" id="review-list-'+idx+'" style="margin-top:8px;max-height:220px;overflow-y:auto"></div>'
        : "";
      return '<div class="m" style="text-align:left;margin-top:10px">'+
        '<div class="k">'+d.name+' · '+d.confidence+'</div>'+
        '<div class="v" style="font-size:16px">'+d.flagged_count+' flagged <span style="font-size:12px;font-weight:500">— '+actionTag+'</span></div>'+
        '<div style="font-size:12.5px;color:var(--ink-2);margin-top:4px">'+d.description+'</div>'+triggers+listHost+
        '</div>';
    }).join("");

    var downloadUrl = data.download_url ? (base + data.download_url) : null;

    $("#result").innerHTML =
      '<div class="verdict flag"><div class="vi">🧹</div><div><div class="vt">'+autoRemoved+' removed automatically'+(pendingReview ? ', '+pendingReview+' flagged for review' : '')+'</div>'+
      '<div class="vs">Only known-trigger matches (stage 1) are auto-removed. Stage 2 is a statistical signal, not a known-signature match — its hits stay in the downloadable set and are reported as pending review instead of being deleted automatically.</div></div></div>'+
      overall + detectorCards +
      '<div class="res-actions"><button class="btn btn-primary" onclick="location.reload()">Run again</button>'+
      (downloadUrl ? '<a class="btn btn-ghost" href="'+downloadUrl+'" download>Download cleaned dataset (.zip)</a>' : '')+
      '</div>';

    // Sample ids come from the buyer's own uploaded filenames — build
    // these rows with textContent, never interpolate them into innerHTML.
    detectors.forEach(function(d, idx){
      if(d.action !== "review" || !d.flagged_count) return;
      var host = $("#review-list-" + idx);
      if(!host) return;
      (d.flagged_sample_ids || []).forEach(function(entry){
        var row = document.createElement("div");
        row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;font-family:var(--mono);font-size:12px;padding:5px 0;border-top:1px solid var(--line-2)";
        var name = document.createElement("span");
        name.textContent = entry.sample_id;
        name.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink-2)";
        var badge = document.createElement("span");
        badge.textContent = entry.level;
        badge.style.cssText = "flex:none;font-weight:700;text-transform:uppercase;font-size:10px;padding:2px 7px;border-radius:20px;" +
          (entry.level === "high"
            ? "background:color-mix(in srgb,var(--threat) 16%,transparent);color:var(--threat)"
            : "background:color-mix(in srgb,var(--amber) 16%,transparent);color:var(--amber)");
        row.appendChild(name);
        row.appendChild(badge);
        host.appendChild(row);
      });
    });
  }
})();
