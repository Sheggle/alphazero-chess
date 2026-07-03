"""Local server to play 3D Connect 4 (4x4x4 "Score Four") against the trained net.

The bot plays PURE net (Gumbel MCTS with models/connect4/playnet.pt) — no hand-coded
tactics, no 1-ply safety nets. The point of 3D Connect 4 is to test whether the same
AlphaZero code transfers to a new game, so the bot's strength IS the net's strength.
If it plays badly, that's the honest signal to fix the training, not to patch the bot.

  PYTHONPATH=. uv run python chess_ui/connect4_play_server.py    # -> http://localhost:8099
Stateless: the browser holds the move list (columns 0..15); each /api/move reconstructs
the game, applies the human move, then the bot replies. The UI is a rotatable Three.js
board (pegs + stacked beads) with a separate 4x4 column selector for moves.
"""
import sys, json, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from alphazero.connect4_env import Connect4Game, LINES, _cell
from alphazero.connect4_net import Connect4Net, Connect4Evaluator
from alphazero.gumbel import GumbelMCTS

CKPT = ROOT / "models/connect4/playnet.pt"
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
_net = Connect4Net(channels=_ck["channels"], blocks=_ck["blocks"])
_net.load_state_dict(_ck["state_dict"]); _net.eval()
_DEV = "mps" if torch.backends.mps.is_available() else "cpu"
_ev = Connect4Evaluator(_net.to(_DEV), device=_DEV)
_lock = threading.Lock()
_rng = np.random.default_rng(0)


def game_from_moves(moves):
    g = Connect4Game()
    for c in moves:
        g = g.apply(int(c))
    return g


def bot_column(g, sims):
    """PURE net move — Gumbel MCTS with the trained net, no hand-coded tactics.

    The point of 3D Connect 4 is to test whether the same AlphaZero code transfers to a
    new game: the bot's strength IS the net's strength. No 1-ply safety nets, no
    alpha-beta patches — if it plays badly, that's the honest signal to fix training.
    """
    with _lock:
        a, _ = GumbelMCTS(_ev, n_sims=int(sims), max_considered=8, rng=_rng).run(g, add_noise=False)
    return int(a), "net"


def win_line_cells(g):
    if g.result() == 0:
        return None
    flat = g.board.reshape(-1)
    for line in LINES:
        cells = [_cell(*c) for c in line]
        v = flat[cells[0]]
        if v != 0 and all(flat[c] == v for c in cells):
            return cells
    return None


def state_payload(g, last_bot=None, bot_kind=None):
    over = g.is_terminal()
    res = g.result()
    return {
        "board": g.board.tolist(),
        "to_play": g.to_play,
        "ply": g.ply,
        "legal": g.legal_moves(),
        "over": over,
        "winner": res,
        "win_line": win_line_cells(g) if over else None,
        "last_bot": last_bot,
        "bot_kind": bot_kind,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/move":
            self._send(404, b"{}"); return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        moves = list(req.get("moves", []))
        sims = int(req.get("sims", 200))
        human_plays = req.get("human", 1)
        try:
            g = game_from_moves(moves)
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}).encode()); return
        last_bot = bot_kind = None
        if not g.is_terminal() and g.to_play != human_plays:
            c, kind = bot_column(g, sims)
            g = g.apply(c)
            moves = moves + [c]
            last_bot, bot_kind = c, kind
        payload = state_payload(g, last_bot, bot_kind)
        payload["moves"] = moves
        self._send(200, json.dumps(payload).encode())


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Play 3D Connect 4</title>
<style>
  body{font-family:-apple-system,system-ui,sans-serif;background:#1e1e22;color:#e8e8ea;
    margin:0;display:flex;flex-direction:column;align-items:center;padding:8px;gap:6px}
  h1{font-size:18px;margin:0}
  h1 span{font-size:12px;color:#9a9aa2}
  #status{font-size:16px;font-weight:600;min-height:22px}
  .thinking{color:#ffcf4a}
  #view{width:min(680px,96vw);height:min(46vh,420px);border-radius:12px;overflow:hidden;
    background:#17181c;touch-action:none;cursor:grab}
  #view:active{cursor:grabbing}
  .hint{color:#7a7f8a;font-size:11px}
  .selwrap{display:flex;gap:16px;align-items:center;flex-wrap:wrap;justify-content:center}
  .sel{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}
  .sel button{width:40px;height:40px;border-radius:8px;border:1px solid #44444c;background:#2b2f38;
    color:#c9ced8;font-size:12px;font-weight:600;cursor:pointer;transition:.1s;padding:0}
  .sel button:hover:not(:disabled){background:#4a7dff;border-color:#4a7dff;color:#fff}
  .sel button:disabled{opacity:.35;cursor:default}
  .sel .lab{font-size:9px;color:#6a6f7a;text-align:center;grid-column:span 4;margin-top:-2px}
  .legend{color:#9a9aa2;font-size:12px}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center}
  select,.btn{background:#2c2c33;color:#e8e8ea;border:1px solid #44444c;border-radius:6px;
    padding:6px 11px;font-size:13px;cursor:pointer}
  .btn.primary{background:#4a7dff;border-color:#4a7dff;font-weight:600}
</style></head><body>
<h1>3D Connect 4 <span>vs the net</span></h1>
<div id="status">Your move</div>
<div id="view"></div>
<div class="hint">drag to rotate · scroll to zoom · pick your column below →</div>
<div class="selwrap">
  <div>
    <div class="sel" id="sel"></div>
    <div class="hint" style="text-align:center;margin-top:4px">drop column</div>
  </div>
  <div class="legend">🔴 you &nbsp;·&nbsp; 🟡 net<br>bottom bead = level&nbsp;1</div>
</div>
<div class="controls">
  <label>You play
    <select id="side"><option value="1" selected>Red (first)</option><option value="-1">Yellow (second)</option></select>
  </label>
  <label>Strength
    <select id="sims"><option value="80">Fast</option><option value="200" selected>Normal</option><option value="600">Strong</option></select>
  </label>
  <button class="btn primary" onclick="newGame()">New game</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const N=4, SP=1.35, BEAD=0.9;   // spacing, per-level height
let moves=[], human=1, sims=200, board=null, over=false, winLine=[], busy=false;
const flatIdx=(x,y,z)=>x*16+y*4+z;
const px=i=>(i-1.5)*SP;

// ---------- three.js scene ----------
const view=document.getElementById('view');
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x17181c);
const camera=new THREE.PerspectiveCamera(42, view.clientWidth/view.clientHeight, 0.1, 100);
camera.position.set(6.5,6.5,8.5);
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(view.clientWidth, view.clientHeight);
view.appendChild(renderer.domElement);
const controls=new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0,1.4,0); controls.enablePan=false;
controls.minDistance=6; controls.maxDistance=18; controls.update();
scene.add(new THREE.AmbientLight(0xffffff,0.65));
const dl=new THREE.DirectionalLight(0xffffff,0.85); dl.position.set(6,12,8); scene.add(dl);
const dl2=new THREE.DirectionalLight(0x88aaff,0.25); dl2.position.set(-6,4,-8); scene.add(dl2);

const boardGrp=new THREE.Group(); scene.add(boardGrp);
// base plate
const base=new THREE.Mesh(new THREE.BoxGeometry(4*SP+0.5,0.35,4*SP+0.5),
  new THREE.MeshStandardMaterial({color:0x2a2320,roughness:.9}));
base.position.y=-0.18; boardGrp.add(base);
// pegs (one per column), kept for hover-highlight
const pegs={};
const pegH=(N-0.2)*BEAD;
for(let x=0;x<N;x++)for(let y=0;y<N;y++){
  const peg=new THREE.Mesh(new THREE.CylinderGeometry(0.06,0.06,pegH,12),
    new THREE.MeshStandardMaterial({color:0x6b5a48,roughness:.6}));
  peg.position.set(px(x), pegH/2, px(y));
  boardGrp.add(peg); pegs[x*N+y]=peg;
}
const beadGrp=new THREE.Group(); boardGrp.add(beadGrp);
const RED=0xe0402f, YEL=0xe8b400;
function rebuildBeads(){
  while(beadGrp.children.length) beadGrp.remove(beadGrp.children[0]);
  if(!board) return;
  for(let x=0;x<N;x++)for(let y=0;y<N;y++)for(let z=0;z<N;z++){
    const v=board[x][y][z]; if(v===0) continue;
    const win=winLine&&winLine.includes(flatIdx(x,y,z));
    const m=new THREE.Mesh(new THREE.SphereGeometry(0.42,24,24),
      new THREE.MeshStandardMaterial({color:v===1?RED:YEL, roughness:.35, metalness:.15,
        emissive:win?0x22aa44:0x000000, emissiveIntensity:win?0.9:0}));
    m.position.set(px(x), z*BEAD+0.45, px(y));
    beadGrp.add(m);
  }
}
function highlightPeg(col,on){
  for(const k in pegs) pegs[k].material.emissive.setHex(0x000000);
  if(on&&col!=null){ pegs[col].material.emissive.setHex(0x2a4d99); }
}
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene,camera); })();
window.addEventListener('resize',()=>{
  camera.aspect=view.clientWidth/view.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(view.clientWidth, view.clientHeight);
});

// ---------- column selector ----------
const sel=document.getElementById('sel');
function buildSel(){
  sel.innerHTML='';
  for(let x=0;x<N;x++)for(let y=0;y<N;y++){
    const col=x*N+y;
    const b=document.createElement('button');
    b.textContent=String.fromCharCode(65+x)+(y+1);
    b.onclick=()=>play(col);
    b.onmouseenter=()=>highlightPeg(col,true);
    b.onmouseleave=()=>highlightPeg(null,false);
    b.dataset.col=col;
    sel.appendChild(b);
  }
}
function refreshSel(){
  [...sel.children].forEach(b=>{
    const col=+b.dataset.col, x=Math.floor(col/N), y=col%N;
    const full=board && board[x][y][N-1]!==0;
    b.disabled=full||over||busy;
  });
}

// ---------- game flow ----------
function setStatus(msg,think){const el=document.getElementById('status');
  el.innerHTML=think?'<span class="thinking">'+msg+'</span>':msg;}
async function send(){
  busy=true; refreshSel();
  try{
    const r=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({moves,sims,human})});
    const d=await r.json();
    if(d.error){setStatus('('+d.error+')');busy=false;refreshSel();return;}
    moves=d.moves; board=d.board; over=d.over; winLine=d.win_line||[];
    rebuildBeads();
    if(d.over){const w=d.winner;
      setStatus(w===0?'Draw — cube full.':(w===human?'You win! 🎉':'The net wins.'));}
    else setStatus('Your move');
  }catch(e){setStatus('(server error)');}
  busy=false; refreshSel();
}
function play(col){
  if(busy||over||!board) return;
  const x=Math.floor(col/N), y=col%N;
  if(board[x][y][N-1]!==0) return;
  const z=board[x][y].findIndex(v=>v===0);
  board[x][y][z]=human; moves.push(col); rebuildBeads();   // record move + optimistic render
  setStatus('Net is thinking…',true); refreshSel();
  send();
}
function newGame(){
  human=parseInt(document.getElementById('side').value);
  sims=parseInt(document.getElementById('sims').value);
  moves=[]; over=false; winLine=[];
  board=Array.from({length:N},()=>Array.from({length:N},()=>[0,0,0,0]));
  rebuildBeads(); refreshSel();
  if(human===-1){ setStatus('Net is thinking…',true); send(); }
  else setStatus('Your move');
}
buildSel(); newGame();
</script></body></html>"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"3D Connect 4 ready -> http://localhost:{port}   (net iter {_ck.get('iter')}, {_DEV}, PURE net — no tactic patches)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
