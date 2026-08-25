(function () {
  const currentStep = document.getElementById("current-step");
  const currentTier = document.getElementById("current-tier");
  const flagsEl = document.getElementById("flags");
  const resetBtn = document.getElementById("reset-btn");
  const mockBanner = document.getElementById("mock-banner");

  resetBtn.addEventListener("click", async () => {
    if (!confirm("Reset the station back to step 1? This discards the current attempt.")) return;
    resetBtn.disabled = true;
    resetBtn.textContent = "Resetting…";
    try {
      await fetch("/api/reset", { method: "POST" });
    } finally {
      resetBtn.disabled = false;
      resetBtn.textContent = "Reset station";
    }
  });

  function renderState(state) {
    if (state.status !== "running") {
      currentStep.textContent = state.status;
      return;
    }
    mockBanner.classList.toggle("visible", !!state.is_mock);
    currentStep.textContent = state.completed
      ? "Assembly complete"
      : `Step ${state.step_id} of ${state.total_steps}: ${state.step_name}`;
    currentTier.textContent = state.completed
      ? ""
      : `tier=${state.tier} · elapsed ${state.elapsed_on_step}s`;
  }

  async function pollFlags() {
    try {
      const res = await fetch("/api/flags");
      const flags = await res.json();
      if (flags.length === 0) {
        flagsEl.innerHTML = '<li class="empty">no flags yet</li>';
        return;
      }
      flagsEl.innerHTML = "";
      for (const f of flags.slice().reverse()) {
        const li = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = `Step ${f.step_id}: ${f.step_name}`;
        const time = document.createElement("span");
        time.textContent = new Date(f.at * 1000).toLocaleTimeString();
        li.appendChild(label);
        li.appendChild(time);
        flagsEl.appendChild(li);
      }
    } catch (e) {
      // transient — next poll will retry
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/state`);
    ws.onmessage = (ev) => renderState(JSON.parse(ev.data));
    ws.onclose = () => setTimeout(connect, 1500);
    ws.onerror = () => ws.close();
  }

  connect();
  pollFlags();
  setInterval(pollFlags, 3000);
})();
