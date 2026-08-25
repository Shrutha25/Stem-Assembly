(function () {
  const stepLabel = document.getElementById("step-label");
  const stepName = document.getElementById("step-name");
  const checklist = document.getElementById("checklist");
  const subStepProgress = document.getElementById("sub-step-progress");
  const tierImage = document.getElementById("tier-image");
  const tierImageImg = document.getElementById("tier-image-img");
  const tierVideo = document.getElementById("tier-video");
  const tierVideoVid = document.getElementById("tier-video-vid");
  const tierTrainer = document.getElementById("tier-trainer");
  const statusBar = document.getElementById("status-bar");
  const completeBanner = document.getElementById("complete-banner");
  const main = document.querySelector("main");
  const resetBtn = document.getElementById("reset-btn");
  const resetBtnComplete = document.getElementById("reset-btn-complete");
  const mockBanner = document.getElementById("mock-banner");

  async function resetStation(btn) {
    btn.disabled = true;
    btn.textContent = "Resetting…";
    try {
      await fetch("/api/reset", { method: "POST" });
    } finally {
      btn.disabled = false;
      btn.textContent = "Start again";
    }
  }
  resetBtn.addEventListener("click", () => resetStation(resetBtn));
  resetBtnComplete.addEventListener("click", () => resetStation(resetBtnComplete));

  const TIER_RANK = { NONE: 0, REFERENCE_IMAGE: 1, REFERENCE_VIDEO: 2, TRAINER_ALERT: 3 };

  function setAssetOrHide(cardEl, mediaEl, url, kind) {
    if (!url) {
      cardEl.querySelector(".missing")?.remove();
      const note = document.createElement("div");
      note.className = "missing";
      note.textContent = "reference " + kind + " not configured for this step";
      cardEl.appendChild(note);
      mediaEl.style.display = "none";
      return;
    }
    mediaEl.style.display = "";
    if (mediaEl.getAttribute("src") !== url) {
      mediaEl.onerror = () => {
        mediaEl.style.display = "none";
        if (!cardEl.querySelector(".missing")) {
          const note = document.createElement("div");
          note.className = "missing";
          note.textContent = "reference " + kind + " not found yet (" + url + ")";
          cardEl.appendChild(note);
        }
      };
      mediaEl.setAttribute("src", url);
    }
  }

  function render(state) {
    if (state.status !== "running") {
      statusBar.textContent = state.status;
      return;
    }

    mockBanner.classList.toggle("visible", !!state.is_mock);

    if (state.completed) {
      main.style.display = "none";
      completeBanner.classList.add("visible");
      return;
    }
    main.style.display = "";
    completeBanner.classList.remove("visible");

    stepLabel.textContent = `Step ${state.step_id} of ${state.total_steps}`;
    stepName.textContent = state.step_name || "";

    checklist.innerHTML = "";
    const diagnosis = state.class_diagnosis || {};
    for (const [cls, ok] of Object.entries(state.class_status || {})) {
      const li = document.createElement("li");
      const dot = document.createElement("span");
      const reason = diagnosis[cls]?.reason;
      dot.className = "dot" + (ok ? " ok" : reason === "wrong_part" || reason === "wrong_orientation" ? " bad" : "");
      li.appendChild(dot);
      const text = document.createElement("span");
      text.className = "checklist-text";
      const label = document.createElement("span");
      label.textContent = cls.replaceAll("_", " ");
      text.appendChild(label);
      const message = diagnosis[cls]?.message;
      if (!ok && message) {
        const hint = document.createElement("span");
        hint.className = "checklist-hint";
        hint.textContent = message;
        text.appendChild(hint);
      }
      li.appendChild(text);
      checklist.appendChild(li);
    }

    if (state.sub_step_progress) {
      const [done, total] = state.sub_step_progress;
      subStepProgress.style.display = "";
      subStepProgress.textContent = state.waiting_for_reset
        ? `Step ${state.step_id} (${done}/${total}) — move away and show empty before continuing`
        : `Step ${state.step_id} (${done}/${total})`;
    } else {
      subStepProgress.style.display = "none";
    }

    const tier = TIER_RANK[state.tier] ?? 0;
    tierImage.classList.toggle("visible", tier >= 1);
    tierVideo.classList.toggle("visible", tier >= 2);
    tierTrainer.classList.toggle("visible", tier >= 3);
    if (tier >= 1) setAssetOrHide(tierImage, tierImageImg, state.reference_image, "image");
    if (tier >= 2) setAssetOrHide(tierVideo, tierVideoVid, state.reference_video, "video");

    statusBar.textContent = `elapsed on this step: ${state.elapsed_on_step}s`;
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/state`);
    ws.onmessage = (ev) => render(JSON.parse(ev.data));
    ws.onclose = () => {
      statusBar.textContent = "disconnected — retrying…";
      setTimeout(connect, 1500);
    };
    ws.onerror = () => ws.close();
  }

  connect();
})();
