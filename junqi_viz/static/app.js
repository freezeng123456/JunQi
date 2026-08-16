const $ = (selector) => document.querySelector(selector);

const SEAT_LABELS = { SOUTH: "南", WEST: "西", NORTH: "北", EAST: "东" };
const EVENT_LABELS = { MOVE: "移动", EAT: "吃子", KILLED: "被击杀", BOMB: "炸弹同归" };

const state = {
  meta: null,
  step: 0,
  frame: null,
  playing: false,
  timer: null,
};

function position(node, x, y) {
  node.style.left = `${(x + 0.5) / 17 * 100}%`;
  node.style.top = `${(y + 0.5) / 17 * 100}%`;
}

function coordText(point) {
  return point ? `(${point[0]},${point[1]})` : "—";
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

function pieceText(piece) {
  if (!piece) return "空位";
  return `${piece.seat_label}${piece.label} · #${piece.piece_id}`;
}

function renderAction(frame) {
  const detail = frame.move_detail;
  const result = frame.result;
  const actionTeam = $("#action-team");
  const actionSummary = $("#action-summary");
  const srcPiece = $("#src-piece");
  const dstPiece = $("#dst-piece");
  const note = $("#event-note");

  if (!detail) {
    actionTeam.textContent = "初始局面";
    actionSummary.textContent = "开局布阵";
    srcPiece.textContent = "—";
    dstPiece.textContent = "—";
    note.textContent = "从第 1 步开始查看动作和策略解释。";
    return;
  }

  actionTeam.textContent = `Team ${detail.team} · ${detail.seat_label}方`;
  actionSummary.textContent = `${detail.seat_label}方  ${coordText(detail.src)} → ${coordText(detail.dst)}`;
  srcPiece.textContent = pieceText(detail.src_piece);
  dstPiece.textContent = pieceText(detail.dst_piece);

  const notes = [];
  if (result) {
    notes.push(`事件：${result.event_label || EVENT_LABELS[result.event] || result.event}`);
    if (result.flag_captured) notes.push("吃旗");
    if (result.flag_reveal_src || result.flag_reveal_dst) notes.push("军旗信息公开");
    if (result.seats_died_labels?.length) notes.push(`死亡：${result.seats_died_labels.join("、")}方`);
    if (result.terminated_after) {
      notes.push(result.draw_after ? "终局：和棋" : `终局：Team ${result.winner_team_after} 获胜`);
    }
  }
  note.textContent = notes.join(" · ") || "普通移动";
}

function renderState(frame) {
  const stateInfo = frame.state || {};
  const aliveTeams = frame.alive_teams || { team0: 0, team1: 0 };
  $("#move-counter").textContent = stateInfo.move_counter ?? frame.move_counter ?? "0";
  $("#moves-since-combat").textContent = stateInfo.moves_since_last_combat ?? "0";
  $("#team0-alive").textContent = aliveTeams.team0 ?? "0";
  $("#team1-alive").textContent = aliveTeams.team1 ?? "0";

  const seatState = $("#seat-state");
  seatState.replaceChildren();
  for (const seat of ["SOUTH", "WEST", "NORTH", "EAST"]) {
    const info = frame.seat_info?.[seat];
    if (!info) continue;
    const row = document.createElement("div");
    row.className = `seat-row ${info.dead ? "dead" : ""} ${info.flag_revealed ? "flag-revealed" : ""}`;
    const label = document.createElement("span");
    label.textContent = `${info.label} · T${info.team}`;
    const status = document.createElement("small");
    status.textContent = `${info.alive_pieces}枚${info.dead ? " · 死亡" : ""}${info.flag_revealed ? " · 旗已知" : ""}`;
    row.append(label, status);
    seatState.append(row);
  }

  const policy = frame.policy;
  if (state.meta?.has_beliefs) {
    $("#belief-state").textContent = "Belief 已记录";
  } else {
    $("#belief-state").textContent = "无 Belief 快照";
  }
}

function renderPolicy(policy) {
  const list = $("#policy-list");
  const summary = $("#policy-summary");
  const source = $("#policy-source");
  list.replaceChildren();
  if (!policy || !policy.top_actions?.length) {
    source.textContent = "Top-K";
    summary.textContent = "普通对局没有策略数据";
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "普通对局没有策略数据";
    list.append(empty);
    return;
  }

  source.textContent = policy.source_label || policy.action_source || "Top-K";
  const chosenProbability = policy.chosen_probability;
  const chosenRank = policy.chosen_rank;
  const chosenText = chosenProbability == null
    ? "不在 Top-K"
    : `Top-${chosenRank} · ${(chosenProbability * 100).toFixed(1)}%`;
  summary.textContent = `已选动作：${chosenText} · Value ${Number(policy.value).toFixed(4)}`;

  for (const item of policy.top_actions.slice(0, 8)) {
    const row = document.createElement("li");
    if (item.chosen) row.classList.add("chosen");
    const move = document.createElement("span");
    move.textContent = `${item.chosen ? "★ " : ""}(${item.src.join(",")}) → (${item.dst.join(",")})`;
    const prob = document.createElement("span");
    prob.className = "prob";
    prob.textContent = `${(item.probability * 100).toFixed(1)}%`;
    row.append(move, prob);
    list.append(row);
  }
}

function renderEvents() {
  const list = $("#event-list");
  const events = state.meta?.key_events || [];
  $("#event-count").textContent = `${events.length} 个`;
  list.replaceChildren();
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "没有关键战斗节点";
    list.append(empty);
    return;
  }
  for (const event of events) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `event-item ${event.step === state.step ? "active" : ""}`;
    const step = document.createElement("span");
    step.className = "event-step";
    step.textContent = `${event.step}`;
    const main = document.createElement("span");
    main.className = "event-main";
    const action = event.src && event.dst ? `${coordText(event.src)}→${coordText(event.dst)}` : "";
    const deaths = event.seats_died_labels?.length ? ` · ${event.seats_died_labels.join("、")}死` : "";
    main.textContent = `${event.seat_label || "—"} · ${event.event_label || event.event} ${action}${deaths}`;
    const value = document.createElement("span");
    value.className = "event-value";
    value.textContent = event.value == null ? "" : Number(event.value).toFixed(3);
    button.append(step, main, value);
    button.addEventListener("click", () => loadFrame(event.step));
    list.append(button);
  }
}

function renderSourceMeta() {
  const meta = state.meta?.meta || {};
  const checkpoint = meta.checkpoint ? String(meta.checkpoint).split("/").pop() : "—";
  const policyTeam = meta.policy_team == null ? "—" : `Team ${meta.policy_team}`;
  const lines = [
    `运行：${meta.run_id || "—"}`,
    `Checkpoint：${checkpoint}`,
    `策略方：${policyTeam}`,
    `规则：${state.meta?.rules_version || "—"} · 状态：${state.meta?.state_version || "—"}`,
  ];
  $("#run-meta").textContent = lines.join("\n");
  const shape = state.meta?.belief_shape;
  $("#belief-shape").textContent = shape ? `Belief ${shape.join("×")}` : "无 Belief";
}

async function loadFrame(step) {
  const response = await fetch(`/api/frame/${step}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  const frame = await response.json();
  state.frame = frame;
  state.step = frame.step;
  $("#step-number").textContent = frame.step;
  if (frame.terminated) {
    $("#turn-badge").textContent = frame.draw ? "和棋" : `Team ${frame.winner_team} 获胜`;
  } else {
    $("#turn-badge").textContent = `${SEAT_LABELS[frame.turn] || frame.turn}方`;
  }
  $("#event").textContent = frame.result?.event_label || EVENT_LABELS[frame.result?.event] || "开局";
  $("#dead-count").textContent = frame.dead_pieces;
  $("#value").textContent = frame.policy ? Number(frame.policy.value).toFixed(4) : "—";
  $("#action-source").textContent = frame.policy?.source_label || frame.policy?.action_source || "—";
  $("#scrubber").value = frame.step;
  $("#progress").textContent = `${frame.step} / ${frame.length}`;
  renderPieces(frame);
  renderAction(frame);
  renderState(frame);
  renderPolicy(frame.policy);
  renderEvents();
  if (state.playing && frame.step >= frame.length) stopPlayback();
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.timer);
  state.timer = null;
  $("#play").textContent = "播放";
}

function schedulePlayback() {
  if (!state.playing) return;
  const delay = Number($("#speed").value);
  state.timer = setTimeout(async () => {
    if (state.step >= state.meta.length) return stopPlayback();
    try {
      await loadFrame(state.step + 1);
      schedulePlayback();
    } catch (error) {
      stopPlayback();
      console.error(error);
    }
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
  $("#replay-kind").textContent = state.meta.kind === "policy"
    ? `强化学习轨迹 · ${state.meta.length} 步`
    : `普通对局 · ${state.meta.length} 步`;
  $("#scrubber").max = state.meta.length;
  renderCells(state.meta.cells);
  renderSourceMeta();

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
  $("#replay-kind").textContent = String(error);
});
