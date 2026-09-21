"""Deliver old HTTP responses last through the real browser application code."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('different_observer', [False, True])
def test_stale_frame_response_cannot_replace_current_player_view(different_observer):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the async UI regression')
    app = Path(__file__).resolve().parents[1] / 'junqi_viz/static/app.js'
    script = r'''
const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const nodes = new Map();
const requests = [];
const context = vm.createContext({console, setTimeout, clearTimeout,
  document: {querySelector: key => {
    if (!nodes.has(key)) nodes.set(key, {textContent:'', value:0});
    return nodes.get(key);
  }},
  fetch: url => new Promise(resolve => requests.push({url, resolve})),
});
const source = fs.readFileSync(process.argv[1], 'utf8').split('boot().catch')[0];
vm.runInContext(source, context);
const changeView = process.argv[2] === 'True';
context.changeView = changeView;
vm.runInContext(`
  state.meta = {length:3, key_events:[]};
  state.observer = changeView ? 'OMNISCIENT' : 'SOUTH';
  const drawn = [];
  renderPieces = frame => drawn.push(frame);
  renderAction = renderState = renderPolicy = renderEvents = () => {};
  const oldRequest = loadFrame(0);
  state.observer = 'SOUTH';
  const newRequest = loadFrame(1);
`, context);
function frame(step, observer) {
  return {step, observer, turn:'SOUTH', length:3, pieces:[], key_events:[],
    policy:null, dead_pieces:0, terminated:false};
}
(async () => {
  requests[1].resolve({ok:true, json:async()=>frame(1,'SOUTH')});
  await vm.runInContext('newRequest', context);
  requests[0].resolve({ok:true, json:async()=>frame(0,changeView?null:'SOUTH')});
  await vm.runInContext('oldRequest', context);
  assert.equal(vm.runInContext('state.step', context), 1);
  assert.equal(vm.runInContext('state.frame.observer', context), 'SOUTH');
  assert.equal(vm.runInContext('drawn.length', context), 1);
})().catch(err => {console.error(err);process.exitCode=1;});
'''
    subprocess.run([node, '-e', script, str(app), str(different_observer)], check=True,
                   capture_output=True, text=True, timeout=10)
