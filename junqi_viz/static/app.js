const $ = (selector) => document.querySelector(selector);
const state = { meta: null, step: 0, playing: false, timer: null };

function position(node, x, y) {
  node.style.left = `${(x + 0.5) / 17 * 100}%`;
  node.style.top = `${(y + 0.5) / 17 * 100}%`;
}

function renderCells(cells) {
  const layer = $("#cell-layer");
  layer.replaceChildren();
  for (const cell of cells) {
    const node = document.createElement("div");
    node.className = [
      "cell",
      cell.railway ? "railway" : "",
      cell.camp ? "camp" : "",
      cell.stronghold ? "stronghold" : "",
      cell.nine_grid ? "nine-grid" : "",
    ].filter(Boolean).join(" ");
    position(node, cell.x, cell.y);
    layer.append(node);
  }
}

function renderPieces(frame) {
  const layer = $("#piece-layer");
  const action = frame.action;
  layer.replaceChildren();
  for (const piece of frame.pieces) {
    const node = document.createElement("div");
    node.className = `piece ${piece.seat}`;
    node.textContent = piece.label;
    node.title = `${piece.seat} · ${piece.label} · #${piece.piece_id}`;
    if (action && action.src[0] === piece.x && action.src[1] === piece.y) node.classList.add("last-src");
    if (action && action.dst[0] === piece.x && action.dst[1] === piece.y) node.classList.add("last-dst");
    position(node, piece.x, piece.y);
    layer.append(node);
  }
}

function renderPolicy(policy) {
  const list = $("#policy-list");
  list.replaceChildren();
  if (!policy || !policy.top_actions?.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "普通对局没有策略数据";
    list.append(empty);
    return;
  }
  for (const item of policy.top_actions.slice(0, 8)) {
    const row = document.createElement("li");
    const move = document.createElement("span");
    move.textContent = `(${item.src.join(",")}) → (${item.dst.join(",")})`;
    const prob = document.createElement("span");
    prob.className = "prob";
    prob.textContent = `${(item.probability * 100).toFixed(1)}%`;
    row.append(move, prob);
    list.append(row);
  }
}

function renderAlive(alive) {
  const labels = { SOUTH: "南 · 橙", WEST: "西 · 蓝", NORTH: "北 · 绿", EAST: "东 · 紫" };
  const list = $("#alive-list");
  list.replaceChildren();
  for (const seat of ["SOUTH", "WEST", "NORTH", "EAST"]) {
    const node = document.createElement("span");
    node.textContent = `${labels[seat]}　${alive[seat]}`;
    list.append(node);
  }
}

async function loadFrame(step) {
  const response = await fetch(`/api/frame/${step}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  const frame = await response.json();
  state.step = frame.step;
  $("#step-number").textContent = frame.step;
  $("#turn-badge").textContent = frame.turn;
  $("#event").textContent = frame.result?.event ?? "开局";
  $("#dead-count").textContent = frame.dead_pieces;
  $("#value").textContent = frame.policy ? Number(frame.policy.value).toFixed(3) : "—";
  $("#action-source").textContent = frame.policy?.action_source ?? "—";
  $("#scrubber").value = frame.step;
  $("#progress").textContent = `${frame.step} / ${frame.length}`;
  renderPieces(frame);
  renderPolicy(frame.policy);
  renderAlive(frame.alive_pieces);
  if (state.playing && frame.step >= frame.length) stopPlayback();
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.timer);
  $("#play").textContent = "播放";
}

function schedulePlayback() {
  if (!state.playing) return;
  const delay = Number($("#speed").value);
  state.timer = setTimeout(async () => {
    if (state.step >= state.meta.length) return stopPlayback();
    await loadFrame(state.step + 1);
    schedulePlayback();
  }, delay);
}

function togglePlayback() {
  if (state.playing) return stopPlayback();
  if (state.step >= state.meta.length) loadFrame(0);
  state.playing = true;
  $("#play").textContent = "暂停";
  schedulePlayback();
}

async function boot() {
  const response = await fetch("/api/meta", { cache: "no-store" });
  state.meta = await response.json();
  $("#filename").textContent = state.meta.filename;
  $("#replay-kind").textContent = state.meta.kind === "policy" ? "强化学习轨迹" : "普通对局";
  $("#scrubber").max = state.meta.length;
  renderCells(state.meta.cells);

  $("#first").addEventListener("click", () => loadFrame(0));
  $("#previous").addEventListener("click", () => loadFrame(Math.max(0, state.step - 1)));
  $("#play").addEventListener("click", togglePlayback);
  $("#next").addEventListener("click", () => loadFrame(Math.min(state.meta.length, state.step + 1)));
  $("#last").addEventListener("click", () => loadFrame(state.meta.length));
  $("#scrubber").addEventListener("input", (event) => loadFrame(Number(event.target.value)));
  $("#label-toggle").addEventListener("change", (event) => {
    $("#board").classList.toggle("hide-labels", !event.target.checked);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") loadFrame(Math.max(0, state.step - 1));
    if (event.key === "ArrowRight") loadFrame(Math.min(state.meta.length, state.step + 1));
    if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  });
  await loadFrame(0);
}

boot().catch((error) => {
  console.error(error);
  $("#filename").textContent = "复盘加载失败";
});
