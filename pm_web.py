#!/usr/bin/env python3
"""
pm_web.py — Visual PM Suite web interface.

Pure stdlib — no Flask, no React, no external dependencies.
Run alongside your agent: python pm_web.py
Open: http://localhost:7331

The agent uses pm_tool.py directly; this is for humans to visualise/edit.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── Point at the same DB as the agent tool ────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
# Works whether pm_web.py is placed in project root or project/tools/
_DB_PATH = _SCRIPT_DIR / "memdir" / "pm_suite.db"
if not _DB_PATH.exists():
    # Also try sibling directory
    _DB_PATH = _SCRIPT_DIR.parent / "memdir" / "pm_suite.db"

PORT = 7331

# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _jl(s):
    try:    return json.loads(s) if isinstance(s, str) else (s or [])
    except: return []

def _uid():  return str(uuid.uuid4())[:8]
def _ts():   return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def _fmt(s):
    if not s: return "—"
    try:    return datetime.fromisoformat(s.replace("Z","")).strftime("%d %b %Y")
    except: return s

# ── HTML page ─────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent PM Suite</title>
<style>
  :root{--bg:#0f172a;--sb:#1e293b;--bdr:#334155;--txt:#e2e8f0;--dim:#94a3b8;
        --muted:#475569;--blue:#3b82f6;--green:#10b981;--yellow:#f59e0b;
        --red:#ef4444;--purple:#8b5cf6;--cyan:#06b6d4;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;
       display:flex;height:100vh;overflow:hidden;}
  /* Sidebar */
  #sidebar{width:220px;background:var(--sb);border-right:1px solid var(--bdr);
            display:flex;flex-direction:column;flex-shrink:0;}
  #sb-head{padding:14px 16px;border-bottom:1px solid var(--bdr);}
  #sb-head h1{font-size:14px;font-weight:700;}
  #sb-head p{font-size:11px;color:var(--muted);margin-top:2px;}
  #proj-list{flex:1;overflow-y:auto;padding:6px 0;}
  .proj-btn{display:block;width:100%;text-align:left;padding:8px 14px;
             background:transparent;border:none;border-left:3px solid transparent;
             color:var(--dim);cursor:pointer;font-size:13px;transition:.1s;}
  .proj-btn:hover{background:#0f172a;color:var(--txt);}
  .proj-btn.active{background:#0f172a;border-left-color:var(--blue);color:var(--txt);}
  .proj-btn small{display:block;font-size:11px;color:var(--muted);margin-top:1px;}
  #sb-foot{padding:10px 14px;border-top:1px solid var(--bdr);}
  /* Main */
  #main{flex:1;display:flex;flex-direction:column;overflow:hidden;}
  #topbar{height:50px;background:var(--sb);border-bottom:1px solid var(--bdr);
           display:flex;align-items:center;gap:10px;padding:0 16px;flex-shrink:0;}
  #topbar h2{font-size:15px;font-weight:600;}
  #tabs{display:flex;gap:2px;padding:8px 14px;border-bottom:1px solid var(--bdr);
        background:var(--bg);flex-shrink:0;}
  .tab{padding:6px 12px;border-radius:6px;border:none;cursor:pointer;
       font-size:12px;font-weight:500;background:transparent;color:var(--muted);transition:.1s;}
  .tab:hover{color:var(--txt);}
  .tab.active{background:var(--sb);color:var(--txt);}
  #content{flex:1;overflow-y:auto;padding:14px;}
  /* Cards */
  .card{background:var(--sb);border:1px solid var(--bdr);border-radius:8px;
        padding:12px;margin-bottom:8px;}
  .card h3{font-size:13px;font-weight:600;margin-bottom:6px;}
  /* Kanban */
  #kanban{display:flex;gap:10px;height:100%;overflow-x:auto;}
  .kcol{min-width:200px;flex:0 0 200px;display:flex;flex-direction:column;}
  .kcol-head{display:flex;align-items:center;gap:6px;margin-bottom:8px;padding:4px 0;}
  .kcol-head span.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
  .kcol-head span.lbl{font-size:12px;font-weight:600;color:#cbd5e1;}
  .kcol-head span.cnt{margin-left:auto;font-size:11px;color:var(--muted);
                       background:var(--bg);border-radius:999px;padding:1px 6px;}
  .kcol-body{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:6px;}
  .kcard{background:var(--sb);border:1px solid var(--bdr);border-radius:6px;
          padding:10px;cursor:pointer;transition:.1s;}
  .kcard:hover{border-color:var(--blue);}
  .kcard .kt{font-size:13px;font-weight:500;margin-bottom:4px;}
  .kcard .kd{font-size:12px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  /* Log timeline */
  .log-line{position:relative;padding-left:24px;margin-bottom:10px;}
  .log-dot{position:absolute;left:0;top:2px;width:18px;height:18px;border-radius:50%;
            background:var(--bg);border:2px solid;display:flex;align-items:center;
            justify-content:center;font-size:9px;}
  /* Badges */
  .badge{font-size:11px;padding:2px 7px;border-radius:4px;display:inline-block;}
  /* Buttons */
  .btn{display:inline-flex;align-items:center;gap:4px;padding:6px 12px;
       border-radius:6px;border:none;cursor:pointer;font-size:13px;font-weight:500;}
  .btn-primary{background:var(--blue);color:#fff;}
  .btn-ghost{background:transparent;color:var(--dim);border:1px solid var(--bdr);}
  .btn-ghost:hover{color:var(--txt);}
  .btn-green{background:var(--green);color:#fff;}
  /* Forms */
  .form-group{margin-bottom:12px;}
  label{font-size:12px;color:var(--dim);display:block;margin-bottom:4px;}
  input,textarea,select{width:100%;background:var(--bg);border:1px solid var(--bdr);
                         border-radius:6px;padding:7px 10px;color:var(--txt);
                         font-size:13px;font-family:inherit;}
  textarea{resize:vertical;min-height:80px;}
  select{cursor:pointer;}
  /* Grid stats */
  .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin-bottom:14px;}
  .stat-card{background:var(--sb);border:1px solid var(--bdr);border-radius:8px;
              padding:12px 10px;text-align:center;}
  .stat-card .num{font-size:22px;font-weight:700;}
  .stat-card .lbl{font-size:11px;color:var(--muted);margin-top:2px;}
  /* Modal */
  .modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.7);display:flex;
             align-items:center;justify-content:center;z-index:100;}
  .modal{background:var(--sb);border:1px solid var(--bdr);border-radius:10px;
          padding:22px;width:500px;max-height:88vh;overflow-y:auto;}
  .modal h3{font-size:15px;font-weight:700;margin-bottom:16px;}
  /* Snippet code */
  pre{background:var(--bg);border-radius:6px;padding:10px;font-size:12px;
      color:#cbd5e1;overflow-x:auto;max-height:180px;margin:8px 0;}
  /* Misc */
  .empty{text-align:center;color:var(--muted);padding:40px 0;font-size:14px;}
  .row{display:flex;gap:8px;align-items:center;}
  .gap-top{margin-top:12px;}
  .text-sm{font-size:12px;color:var(--dim);}
  .chip{font-size:11px;padding:1px 6px;border-radius:3px;background:var(--bg);}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sb-head">
    <h1>🧠 Agent PM Suite</h1>
    <p>Persistent Project Management</p>
  </div>
  <div id="proj-list"></div>
  <div id="sb-foot">
    <button class="btn btn-ghost" style="width:100%;justify-content:center;font-size:12px"
      onclick="openModal('new-project')">+ New Project</button>
  </div>
</div>

<div id="main">
  <div id="topbar">
    <h2 id="proj-name">Select a project</h2>
    <span id="proj-status-badge" class="badge"></span>
    <div style="margin-left:auto;display:flex;gap:8px;">
      <button class="btn btn-ghost" id="btn-export" onclick="exportBrief()" style="display:none;font-size:12px">
        📋 Agent Brief
      </button>
    </div>
  </div>
  <div id="tabs">
    <button class="tab active" data-tab="dashboard">📊 Dashboard</button>
    <button class="tab" data-tab="kanban">📋 Kanban</button>
    <button class="tab" data-tab="sprints">🏃 Sprints</button>
    <button class="tab" data-tab="logs">📜 Logs</button>
    <button class="tab" data-tab="snippets">💾 Snippets</button>
    <button class="tab" data-tab="docs">📝 Docs</button>
  </div>
  <div id="content"><div class="empty">← Select or create a project</div></div>
</div>

<!-- Modal container -->
<div id="modal-area"></div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let state = { projects:[], project:null, tasks:[], logs:[], snippets:[], sprints:[], tab:'dashboard' };

// ── API ───────────────────────────────────────────────────────────────────────
async function api(method, path, body){
  const opts = { method, headers:{'Content-Type':'application/json'} };
  if(body) opts.body = JSON.stringify(body);
  const r = await fetch('/api'+path, opts);
  return r.json();
}
const GET  = p    => api('GET',    p);
const POST = (p,b)=> api('POST',   p, b);
const PUT  = (p,b)=> api('PUT',    p, b);

// ── Init ──────────────────────────────────────────────────────────────────────
async function init(){
  state.projects = await GET('/projects');
  renderSidebar();
  if(state.projects.length) await selectProject(state.projects[0].id);
}

async function selectProject(id){
  state.project = state.projects.find(p=>p.id===id);
  if(!state.project) return;
  [state.tasks, state.logs, state.snippets, state.sprints] = await Promise.all([
    GET('/projects/'+id+'/tasks'),
    GET('/projects/'+id+'/logs'),
    GET('/projects/'+id+'/snippets'),
    GET('/projects/'+id+'/sprints'),
  ]);
  document.getElementById('proj-name').textContent = state.project.name;
  const sb = document.getElementById('proj-status-badge');
  sb.textContent = state.project.status;
  const sc = {active:'#10b981',paused:'#f59e0b',completed:'#64748b',archived:'#334155'};
  sb.style.cssText = `background:${sc[state.project.status]||'#64748b'}22;color:${sc[state.project.status]||'#64748b'};border:1px solid ${sc[state.project.status]||'#64748b'}44`;
  document.getElementById('btn-export').style.display='';
  renderSidebar();
  renderTab(state.tab);
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function renderSidebar(){
  const el = document.getElementById('proj-list');
  if(!state.projects.length){
    el.innerHTML = '<div style="padding:12px 16px;font-size:12px;color:var(--muted)">No projects yet.</div>';
    return;
  }
  el.innerHTML = state.projects.map(p=>{
    const active = state.project?.id===p.id;
    const sc = {active:'#10b981',paused:'#f59e0b',completed:'#64748b',archived:'#334155'};
    return `<button class="proj-btn ${active?'active':''}" onclick="selectProject('${p.id}')">
      ${p.name}
      <small style="color:${sc[p.status]||'#64748b'}">${p.status}</small>
    </button>`;
  }).join('');
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click', ()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    state.tab = t.dataset.tab;
    renderTab(state.tab);
  });
});

function renderTab(tab){
  if(!state.project){ document.getElementById('content').innerHTML='<div class="empty">Select a project first.</div>'; return; }
  const fns = {dashboard:renderDashboard, kanban:renderKanban, sprints:renderSprints,
                logs:renderLogs, snippets:renderSnippets, docs:renderDocs};
  (fns[tab]||renderDashboard)();
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
function renderDashboard(){
  const t = state.tasks;
  const byStatus = s => t.filter(x=>x.status===s).length;
  const STATUSES = [
    {id:'done',color:'#10b981'},{id:'in-progress',color:'#f59e0b'},
    {id:'blocked',color:'#ef4444'},{id:'failed',color:'#8b5cf6'},
    {id:'todo',color:'#3b82f6'},{id:'backlog',color:'#64748b'}
  ];
  const activeSprint = state.sprints.find(s=>s.status==='active');
  const workingSnips = state.snippets.filter(s=>s.status==='working');
  const recentLogs   = [...state.logs].sort((a,b)=>b.timestamp.localeCompare(a.timestamp)).slice(0,5);
  const icons = {session:'🚀',attempt:'⚡',success:'✅',failure:'❌',solution:'💡',checkpoint:'🏁',note:'📝'};

  document.getElementById('content').innerHTML = `
    <div class="stats-grid">${STATUSES.map(s=>`
      <div class="stat-card">
        <div class="num" style="color:${s.color}">${byStatus(s.id)}</div>
        <div class="lbl">${s.id}</div>
      </div>`).join('')}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="card">
        <h3>🏃 Active Sprint</h3>
        ${activeSprint ? `
          <div style="font-size:14px;font-weight:500">${activeSprint.name}</div>
          <div class="text-sm" style="margin-top:3px">${activeSprint.goal||'No goal set'}</div>
          <div class="text-sm" style="margin-top:4px">${fmtDate(activeSprint.start_date)} → ${fmtDate(activeSprint.end_date)}</div>
        ` : '<div class="text-sm">No active sprint</div>'}
      </div>
      <div class="card">
        <h3>💡 Working Solutions</h3>
        ${workingSnips.length ? workingSnips.map(s=>`<div class="text-sm" style="margin-bottom:3px">💡 ${s.title}</div>`).join('') : '<div class="text-sm">None logged yet</div>'}
      </div>
      <div class="card" style="grid-column:1/-1">
        <h3>📜 Recent Activity</h3>
        ${recentLogs.length ? recentLogs.map(l=>`
          <div style="display:flex;gap:8px;padding:6px 0;border-bottom:1px solid var(--bdr)">
            <span>${icons[l.type]||'📝'}</span>
            <div>
              <div style="font-size:13px">${l.title||l.content?.slice(0,60)||'—'}</div>
              <div class="text-sm">${l.timestamp?.slice(0,16)}</div>
            </div>
          </div>`).join('') : '<div class="text-sm">No activity yet</div>'}
      </div>
    </div>
    <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
      <button class="btn btn-primary" onclick="openModal('new-task')">+ Task</button>
      <button class="btn btn-ghost" onclick="openModal('new-log')">+ Log Entry</button>
      <button class="btn btn-ghost" onclick="openModal('new-snippet')">+ Snippet</button>
      <button class="btn btn-ghost" onclick="openModal('new-sprint')">+ Sprint</button>
    </div>`;
}

// ── Kanban ────────────────────────────────────────────────────────────────────
const COLS = [
  {id:'backlog',  label:'Backlog',      dot:'#6b7280'},
  {id:'todo',     label:'To Do',        dot:'#3b82f6'},
  {id:'in-progress',label:'In Progress',dot:'#f59e0b'},
  {id:'blocked',  label:'Blocked',      dot:'#ef4444'},
  {id:'done',     label:'Done',         dot:'#10b981'},
  {id:'failed',   label:'Failed',       dot:'#8b5cf6'},
];
const PCOLORS = {critical:'#ef4444',high:'#f59e0b',medium:'#3b82f6',low:'#64748b'};

function renderKanban(){
  const colsHtml = COLS.map(col=>{
    const tasks = state.tasks.filter(t=>t.status===col.id);
    const cards = tasks.map(t=>{
      const cl  = Array.isArray(t.checklist) ? t.checklist : [];
      const done = cl.filter(c=>c.done).length;
      return `<div class="kcard" onclick="openTaskDetail('${t.id}')">
        <div style="display:flex;gap:6px">
          <div style="width:3px;border-radius:2px;background:${PCOLORS[t.priority]||'#64748b'};flex-shrink:0"></div>
          <div style="flex:1;min-width:0">
            <div class="kt">${esc(t.title)}</div>
            ${t.description?`<div class="kd">${esc(t.description)}</div>`:''}
            ${cl.length?`<div class="text-sm" style="margin-top:4px">✓ ${done}/${cl.length}</div>`:''}
          </div>
        </div>
      </div>`;
    }).join('');
    return `<div class="kcol">
      <div class="kcol-head">
        <span class="dot" style="background:${col.dot}"></span>
        <span class="lbl">${col.label}</span>
        <span class="cnt">${tasks.length}</span>
      </div>
      <div class="kcol-body">${cards}
        ${col.id==='backlog'?`<button class="btn btn-ghost" style="width:100%;justify-content:center;margin-top:4px;font-size:12px" onclick="openModal('new-task')">+ Add Task</button>`:''}
      </div>
    </div>`;
  }).join('');
  document.getElementById('content').innerHTML = `<div id="kanban">${colsHtml}</div>`;
}

function openTaskDetail(id){
  const t = state.tasks.find(x=>x.id===id);
  if(!t) return;
  const cl = Array.isArray(t.checklist) ? t.checklist : [];
  const statusOpts = ['backlog','todo','in-progress','blocked','done','failed']
    .map(s=>`<option value="${s}" ${t.status===s?'selected':''}>${s}</option>`).join('');
  const prioOpts = ['low','medium','high','critical']
    .map(p=>`<option value="${p}" ${t.priority===p?'selected':''}>${p}</option>`).join('');
  showModal(`
    <h3>Task: ${esc(t.title)}</h3>
    <div class="form-group"><label>Status</label>
      <select id="td-status">${statusOpts}</select></div>
    <div class="form-group"><label>Priority</label>
      <select id="td-priority">${prioOpts}</select></div>
    <div class="form-group"><label>Description</label>
      <textarea id="td-desc" rows="3">${esc(t.description||'')}</textarea></div>
    ${cl.length?`<div class="form-group"><label>Checklist</label>
      ${cl.map((c,i)=>`<div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
        <input type="checkbox" id="cl-${i}" ${c.done?'checked':''}>
        <label for="cl-${i}" style="font-size:13px;color:var(--txt)">${esc(c.text)}</label>
      </div>`).join('')}
    </div>`:''}
    <div class="row gap-top">
      <button class="btn btn-primary" onclick="saveTask('${t.id}', ${JSON.stringify(cl).replace(/"/g,'&quot;')})">Save</button>
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn" style="margin-left:auto;background:#ef444422;color:#ef4444;border:1px solid #ef444444" onclick="deleteTask('${t.id}')">Delete</button>
    </div>`);
}

async function saveTask(id, cl){
  const status   = document.getElementById('td-status').value;
  const priority = document.getElementById('td-priority').value;
  const desc     = document.getElementById('td-desc').value;
  const newCl    = cl.map((c,i)=>({...c, done: document.getElementById('cl-'+i)?.checked||false}));
  await PUT('/tasks/'+id, {status, priority, description:desc, checklist:newCl});
  await reloadProject(); closeModal();
}

async function deleteTask(id){
  if(!confirm('Delete this task?')) return;
  await api('DELETE', '/tasks/'+id);
  await reloadProject(); closeModal();
}

// ── Sprints ───────────────────────────────────────────────────────────────────
function renderSprints(){
  const SC = {planning:'#3b82f6',active:'#10b981',completed:'#64748b'};
  const sprints = state.sprints;
  const html = sprints.length ? sprints.map(s=>{
    const spTasks = state.tasks.filter(t=>t.sprint_id===s.id);
    const ICONS = {backlog:'📋',todo:'📌','in-progress':'🔄',blocked:'⛔',done:'✅',failed:'❌'};
    return `<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
        <div>
          <div style="display:flex;gap:8px;align-items:center">
            <span style="font-size:14px;font-weight:600">${esc(s.name)}</span>
            <span class="badge" style="background:${SC[s.status]||'#64748b'}22;color:${SC[s.status]||'#64748b'}">${s.status}</span>
          </div>
          <div class="text-sm" style="margin-top:3px">${esc(s.goal||'')}</div>
          <div class="text-sm" style="margin-top:2px">${fmtDate(s.start_date)} → ${fmtDate(s.end_date)}</div>
        </div>
        <div style="display:flex;gap:6px">
          ${['planning','active','completed'].map(st=>`
            <button onclick="updateSprint('${s.id}','${st}')" class="btn btn-ghost"
              style="font-size:11px;padding:2px 8px;${s.status===st?'border-color:'+SC[st]+';color:'+SC[st]:''}">${st}</button>`).join('')}
        </div>
      </div>
      ${spTasks.length?spTasks.map(t=>`
        <div style="display:flex;gap:6px;padding:4px 0;border-top:1px solid var(--bdr)">
          <span>${ICONS[t.status]||'•'}</span>
          <span style="font-size:13px">${esc(t.title)}</span>
          <span class="text-sm" style="margin-left:auto">${t.status}</span>
        </div>`).join(''):'<div class="text-sm" style="margin-top:6px">No tasks in this sprint</div>'}
    </div>`;
  }).join('') : '<div class="empty">No sprints yet.</div>';
  document.getElementById('content').innerHTML = `
    <div style="display:flex;justify-content:space-between;margin-bottom:12px">
      <button class="btn btn-primary" onclick="openModal('new-sprint')">+ New Sprint</button>
    </div>${html}`;
}

async function updateSprint(id, status){
  await PUT('/sprints/'+id, {status});
  await reloadProject();
}

// ── Logs ──────────────────────────────────────────────────────────────────────
function renderLogs(){
  const icons  = {session:'🚀',attempt:'⚡',success:'✅',failure:'❌',solution:'💡',checkpoint:'🏁',note:'📝'};
  const colors = {session:'#3b82f6',attempt:'#f59e0b',success:'#10b981',
                   failure:'#ef4444',solution:'#8b5cf6',checkpoint:'#06b6d4',note:'#64748b'};
  let filter = 'all';
  function render(){
    const logs = filter==='all' ? state.logs
                                : state.logs.filter(l=>l.type===filter);
    const sorted = [...logs].sort((a,b)=>b.timestamp.localeCompare(a.timestamp));
    const typeFilters = ['all','session','attempt','success','failure','solution','checkpoint','note'];
    document.getElementById('content').innerHTML = `
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">
        <button class="btn btn-primary" onclick="openModal('new-log')">+ Log Entry</button>
        ${typeFilters.map(t=>`<button class="btn btn-ghost" style="font-size:11px;padding:3px 8px;${filter===t?'border-color:var(--blue);color:var(--blue)':''}" onclick="setLogFilter('${t}')">${t==='all'?'All':(icons[t]||'')+' '+t}</button>`).join('')}
      </div>
      <div style="position:relative;padding-left:18px">
        <div style="position:absolute;left:7px;top:0;bottom:0;width:2px;background:var(--sb)"></div>
        ${sorted.map(l=>`
          <div class="log-line">
            <div class="log-dot" style="border-color:${colors[l.type]||'#64748b'}">${icons[l.type]||'📝'}</div>
            <div class="card" style="margin-bottom:0">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span class="badge" style="background:${colors[l.type]||'#64748b'}22;color:${colors[l.type]||'#64748b'}">${icons[l.type]||''} ${l.type}</span>
                <span class="text-sm">${l.timestamp?.slice(0,16)||''}</span>
              </div>
              ${l.title?`<div style="font-size:13px;font-weight:500;margin-bottom:4px">${esc(l.title)}</div>`:''}
              ${l.content?`<div style="font-size:13px;color:var(--dim);white-space:pre-wrap">${esc(l.content)}</div>`:''}
              ${l.outcome?`<div style="margin-top:6px;padding:6px 10px;background:var(--bg);border-radius:4px;font-size:12px;border-left:3px solid var(--blue)"><span style="color:var(--muted)">Outcome: </span>${esc(l.outcome)}</div>`:''}
            </div>
          </div>`).join('')}
        ${sorted.length===0?'<div class="empty">No log entries yet.</div>':''}
      </div>`;
    // Rebind filter buttons
    document.querySelectorAll('[onclick^="setLogFilter"]').forEach(b=>{
      b.addEventListener('click', e=>{ filter=b.getAttribute('onclick').match(/'(\w+)'/)[1]; render(); });
    });
  }
  window.setLogFilter = t => { filter=t; render(); };
  render();
}

// ── Snippets ──────────────────────────────────────────────────────────────────
function renderSnippets(){
  const SC = {working:'#10b981',partial:'#f59e0b',broken:'#ef4444',deprecated:'#64748b'};
  const html = state.snippets.length ? state.snippets.map(s=>`
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
        <div>
          <div style="font-size:14px;font-weight:500">${esc(s.title)}</div>
          <div style="display:flex;gap:6px;margin-top:4px">
            <span class="badge" style="background:${SC[s.status]||'#64748b'}22;color:${SC[s.status]||'#64748b'}">${s.status}</span>
            <span class="text-sm">${esc(s.language)}</span>
          </div>
        </div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;justify-content:flex-end">
          ${Object.entries(SC).map(([st,c])=>`
            <button onclick="updateSnippet('${s.id}','${st}')" class="btn btn-ghost"
              style="font-size:11px;padding:2px 6px;${s.status===st?'border-color:'+c+';color:'+c:''}">${st}</button>`).join('')}
        </div>
      </div>
      ${s.code?`<pre><code>${esc(s.code)}</code></pre>`:''}
      ${(s.tags||[]).length?`<div style="margin-top:6px">${s.tags.map(t=>`<span class="chip">#${esc(t)}</span> `).join('')}</div>`:''}
    </div>`).join('') : '<div class="empty">No snippets yet.</div>';
  document.getElementById('content').innerHTML = `
    <div style="margin-bottom:12px">
      <button class="btn btn-primary" onclick="openModal('new-snippet')">+ Add Snippet</button>
    </div>${html}`;
}

async function updateSnippet(id, status){
  await PUT('/snippets/'+id, {status});
  await reloadProject();
}

// ── Docs ──────────────────────────────────────────────────────────────────────
function renderDocs(){
  const docs = state.project?.docs || '';
  document.getElementById('content').innerHTML = `
    <div style="height:calc(100vh - 160px);display:flex;flex-direction:column">
      <div style="display:flex;justify-content:space-between;margin-bottom:10px">
        <div class="text-sm">Project documentation, architecture notes, decisions</div>
        <button class="btn btn-green" onclick="saveDocs()">💾 Save</button>
      </div>
      <textarea id="docs-ta" style="flex:1;font-family:monospace;font-size:13px;line-height:1.6"
        placeholder="Write docs, architecture notes, known issues, tried approaches…">${esc(docs)}</textarea>
    </div>`;
}

async function saveDocs(){
  const content = document.getElementById('docs-ta').value;
  await PUT('/projects/'+state.project.id, {docs:content});
  state.project.docs = content;
  alert('Docs saved.');
}

// ── Modals ────────────────────────────────────────────────────────────────────
function openModal(type){ showModal(modalForms[type]()); }
function closeModal(){ document.getElementById('modal-area').innerHTML=''; }
function showModal(html){
  document.getElementById('modal-area').innerHTML=`
    <div class="modal-bg" onclick="closeModal()">
      <div class="modal" onclick="event.stopPropagation()">${html}</div>
    </div>`;
}

const modalForms = {
  'new-project': ()=>`
    <h3>New Project</h3>
    <div class="form-group"><label>Name</label><input id="np-name" placeholder="Project name" autofocus></div>
    <div class="form-group"><label>Description</label><textarea id="np-desc" rows="2"></textarea></div>
    <div class="form-group"><label>Status</label><select id="np-status"><option value="active">active</option><option value="paused">paused</option></select></div>
    <div class="form-group"><label>Tags (comma separated)</label><input id="np-tags" placeholder="python, ai, api"></div>
    <div class="row gap-top"><button class="btn btn-primary" onclick="createProject()">Create</button><button class="btn btn-ghost" onclick="closeModal()">Cancel</button></div>`,

  'new-task': ()=>`
    <h3>New Task</h3>
    <div class="form-group"><label>Title</label><input id="nt-title" autofocus></div>
    <div class="form-group"><label>Description</label><textarea id="nt-desc" rows="3"></textarea></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="form-group"><label>Status</label><select id="nt-status"><option value="backlog">backlog</option><option value="todo">todo</option><option value="in-progress">in-progress</option><option value="blocked">blocked</option></select></div>
      <div class="form-group"><label>Priority</label><select id="nt-prio"><option value="medium">medium</option><option value="high">high</option><option value="critical">critical</option><option value="low">low</option></select></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="form-group"><label>Start Date</label><input type="date" id="nt-start"></div>
      <div class="form-group"><label>End Date</label><input type="date" id="nt-end"></div>
    </div>
    ${state.sprints.length?`<div class="form-group"><label>Sprint</label><select id="nt-sprint"><option value="">None</option>${state.sprints.map(s=>`<option value="${s.id}">${esc(s.name)}</option>`).join('')}</select></div>`:''}
    <div class="form-group"><label>Checklist (one per line)</label><textarea id="nt-cl" rows="3" placeholder="Research&#10;Implement&#10;Test"></textarea></div>
    <div class="form-group"><label>Tags</label><input id="nt-tags" placeholder="api, bug, feature"></div>
    <div class="row gap-top"><button class="btn btn-primary" onclick="createTask()">Add Task</button><button class="btn btn-ghost" onclick="closeModal()">Cancel</button></div>`,

  'new-log': ()=>`
    <h3>Add Log Entry</h3>
    <div class="form-group"><label>Type</label>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px" id="log-type-btns">
        ${['session','attempt','success','failure','solution','checkpoint','note'].map((t,i)=>`
          <button class="btn btn-ghost log-type-btn ${i===6?'active':''}" data-type="${t}" onclick="selectLogType(this)"
            style="font-size:12px;padding:3px 10px">${t}</button>`).join('')}
      </div>
      <input type="hidden" id="nl-type" value="note">
    </div>
    <div class="form-group"><label>Title / Summary</label><input id="nl-title" autofocus></div>
    <div class="form-group"><label>Details</label><textarea id="nl-content" rows="3" placeholder="What happened? What was tried?"></textarea></div>
    <div class="form-group"><label>Outcome / Result</label><textarea id="nl-outcome" rows="2" placeholder="Did it work? Why/why not?"></textarea></div>
    ${state.tasks.length?`<div class="form-group"><label>Related Task</label><select id="nl-task"><option value="">None</option>${state.tasks.map(t=>`<option value="${t.id}">${esc(t.title)}</option>`).join('')}</select></div>`:''}
    <div class="row gap-top"><button class="btn btn-primary" onclick="createLog()">Add Entry</button><button class="btn btn-ghost" onclick="closeModal()">Cancel</button></div>`,

  'new-snippet': ()=>`
    <h3>Add Code Snippet</h3>
    <div class="form-group"><label>Title</label><input id="ns-title" placeholder="e.g. Working MCP auth setup" autofocus></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="form-group"><label>Language</label><select id="ns-lang"><option>python</option><option>typescript</option><option>javascript</option><option>bash</option><option>json</option><option>other</option></select></div>
      <div class="form-group"><label>Status</label><select id="ns-status"><option value="partial">partial</option><option value="working">working</option><option value="broken">broken</option></select></div>
    </div>
    <div class="form-group"><label>Code</label><textarea id="ns-code" rows="6" style="font-family:monospace"></textarea></div>
    <div class="form-group"><label>Tags</label><input id="ns-tags" placeholder="mcp, auth, api"></div>
    <div class="row gap-top"><button class="btn btn-primary" onclick="createSnippet()">Save Snippet</button><button class="btn btn-ghost" onclick="closeModal()">Cancel</button></div>`,

  'new-sprint': ()=>`
    <h3>New Sprint</h3>
    <div class="form-group"><label>Name</label><input id="sp-name" value="Sprint 1" autofocus></div>
    <div class="form-group"><label>Goal</label><textarea id="sp-goal" rows="2"></textarea></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="form-group"><label>Start</label><input type="date" id="sp-start"></div>
      <div class="form-group"><label>End</label><input type="date" id="sp-end"></div>
    </div>
    <div class="form-group"><label>Status</label><select id="sp-status"><option value="planning">planning</option><option value="active">active</option><option value="completed">completed</option></select></div>
    <div class="row gap-top"><button class="btn btn-primary" onclick="createSprint()">Create Sprint</button><button class="btn btn-ghost" onclick="closeModal()">Cancel</button></div>`,
};

function selectLogType(btn){
  document.querySelectorAll('.log-type-btn').forEach(b=>b.style.borderColor='');
  btn.style.borderColor = 'var(--blue)';
  btn.style.color = 'var(--blue)';
  document.getElementById('nl-type').value = btn.dataset.type;
}

// ── Create helpers ────────────────────────────────────────────────────────────
async function createProject(){
  const name = document.getElementById('np-name').value.trim();
  if(!name) return;
  const p = await POST('/projects', {
    name, description: document.getElementById('np-desc').value,
    status: document.getElementById('np-status').value,
    tags: document.getElementById('np-tags').value.split(',').map(t=>t.trim()).filter(Boolean),
  });
  state.projects.unshift(p);
  renderSidebar();
  await selectProject(p.id);
  closeModal();
}

async function createTask(){
  const title = document.getElementById('nt-title').value.trim();
  if(!title) return;
  const cl = (document.getElementById('nt-cl').value||'').split('\n').filter(Boolean);
  await POST('/projects/'+state.project.id+'/tasks', {
    title, description: document.getElementById('nt-desc').value,
    status: document.getElementById('nt-status').value,
    priority: document.getElementById('nt-prio').value,
    start_date: document.getElementById('nt-start')?.value||null,
    end_date:   document.getElementById('nt-end')?.value||null,
    sprint_id:  document.getElementById('nt-sprint')?.value||null,
    tags: document.getElementById('nt-tags').value.split(',').map(t=>t.trim()).filter(Boolean),
    checklist: cl,
  });
  await reloadProject(); closeModal();
}

async function createLog(){
  const type = document.getElementById('nl-type').value;
  await POST('/projects/'+state.project.id+'/logs', {
    type, title: document.getElementById('nl-title').value,
    content: document.getElementById('nl-content').value,
    outcome: document.getElementById('nl-outcome').value,
    task_id: document.getElementById('nl-task')?.value||null,
  });
  await reloadProject(); closeModal();
}

async function createSnippet(){
  const title = document.getElementById('ns-title').value.trim();
  if(!title) return;
  await POST('/projects/'+state.project.id+'/snippets', {
    title, language: document.getElementById('ns-lang').value,
    code: document.getElementById('ns-code').value,
    status: document.getElementById('ns-status').value,
    tags: document.getElementById('ns-tags').value.split(',').map(t=>t.trim()).filter(Boolean),
  });
  await reloadProject(); closeModal();
}

async function createSprint(){
  const name = document.getElementById('sp-name').value.trim();
  if(!name) return;
  await POST('/projects/'+state.project.id+'/sprints', {
    name, goal: document.getElementById('sp-goal').value,
    start_date: document.getElementById('sp-start').value||null,
    end_date:   document.getElementById('sp-end').value||null,
    status: document.getElementById('sp-status').value,
  });
  await reloadProject(); closeModal();
}

// ── Agent Brief export ────────────────────────────────────────────────────────
async function exportBrief(){
  const brief = await GET('/projects/'+state.project.id+'/brief');
  showModal(`
    <h3>🤖 Agent Brief</h3>
    <p style="font-size:13px;color:var(--dim);margin-bottom:10px">Copy and paste this as agent context — it contains all project state in compact, retrievable format.</p>
    <textarea readonly style="font-family:monospace;font-size:11px;min-height:360px" onclick="this.select()">${esc(brief.text||'')}</textarea>
    <div class="row gap-top">
      <button class="btn btn-primary" onclick="navigator.clipboard.writeText(document.querySelector('#modal-area textarea').value)">Copy to Clipboard</button>
      <button class="btn btn-ghost" onclick="closeModal()">Close</button>
    </div>`);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
async function reloadProject(){
  if(!state.project) return;
  [state.tasks, state.logs, state.snippets, state.sprints] = await Promise.all([
    GET('/projects/'+state.project.id+'/tasks'),
    GET('/projects/'+state.project.id+'/logs'),
    GET('/projects/'+state.project.id+'/snippets'),
    GET('/projects/'+state.project.id+'/sprints'),
  ]);
  const updated = await GET('/projects');
  state.projects = updated;
  renderSidebar();
  renderTab(state.tab);
}

function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function fmtDate(s){ if(!s) return '—'; try{ return new Date(s).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}); } catch{ return s; } }

init();
</script>
</body>
</html>"""


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default logs
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path in ("", "/"):
            self._html(_HTML.encode())
        elif path == "/api/projects":
            rows = self._query("SELECT * FROM projects ORDER BY updated DESC")
            self._json([self._proj(r) for r in rows])
        elif path.startswith("/api/projects/") and path.endswith("/tasks"):
            pid = path.split("/")[3]
            rows = self._query("SELECT * FROM tasks WHERE project_id=? ORDER BY created", (pid,))
            self._json([self._task(r) for r in rows])
        elif path.startswith("/api/projects/") and path.endswith("/logs"):
            pid = path.split("/")[3]
            rows = self._query("SELECT * FROM logs WHERE project_id=? ORDER BY timestamp DESC LIMIT 50", (pid,))
            self._json([dict(r) for r in rows])
        elif path.startswith("/api/projects/") and path.endswith("/snippets"):
            pid = path.split("/")[3]
            rows = self._query("SELECT * FROM snippets WHERE project_id=? ORDER BY created DESC", (pid,))
            self._json([self._snip(r) for r in rows])
        elif path.startswith("/api/projects/") and path.endswith("/sprints"):
            pid = path.split("/")[3]
            rows = self._query("SELECT * FROM sprints WHERE project_id=? ORDER BY start_date, created", (pid,))
            self._json([dict(r) for r in rows])
        elif path.startswith("/api/projects/") and path.endswith("/brief"):
            pid = path.split("/")[3]
            from tools.pm_tool import _brief
            self._json({"text": _brief(pid)})
        else:
            self._404()

    def do_POST(self):
        body = self._body()
        path = urlparse(self.path).path.rstrip("/")
        uid = lambda: str(uuid.uuid4())[:8]
        ts  = lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        if path == "/api/projects":
            p = {"id": uid(), "name": body.get("name","New Project"),
                 "description": body.get("description",""), "status": body.get("status","active"),
                 "tags": json.dumps(body.get("tags",[])), "docs": "",
                 "created": ts(), "updated": ts()}
            self._exec("INSERT INTO projects VALUES(:id,:name,:description,:status,:tags,:docs,:created,:updated)", p)
            p["tags"] = body.get("tags",[])
            self._json(p)

        elif "/tasks" in path and path.count("/") == 4:
            pid = path.split("/")[3]
            cl  = [{"id": uid(), "text": c, "done": False} for c in body.get("checklist",[])]
            t   = {"id": uid(), "project_id": pid,
                   "title": body.get("title",""), "description": body.get("description",""),
                   "status": body.get("status","backlog"), "priority": body.get("priority","medium"),
                   "sprint_id": body.get("sprint_id"), "start_date": body.get("start_date"),
                   "end_date": body.get("end_date"), "tags": json.dumps(body.get("tags",[])),
                   "checklist": json.dumps(cl), "created": ts(), "updated": ts()}
            self._exec("INSERT INTO tasks VALUES(:id,:project_id,:title,:description,:status,:priority,:sprint_id,:start_date,:end_date,:tags,:checklist,:created,:updated)", t)
            t["tags"] = body.get("tags",[]); t["checklist"] = cl
            self._json(t)

        elif "/logs" in path and path.count("/") == 4:
            pid = path.split("/")[3]
            l   = {"id": uid(), "project_id": pid, "task_id": body.get("task_id"),
                   "type": body.get("type","note"), "title": body.get("title",""),
                   "content": body.get("content",""), "outcome": body.get("outcome",""), "timestamp": ts()}
            self._exec("INSERT INTO logs VALUES(:id,:project_id,:task_id,:type,:title,:content,:outcome,:timestamp)", l)
            self._json(l)

        elif "/snippets" in path and path.count("/") == 4:
            pid = path.split("/")[3]
            s   = {"id": uid(), "project_id": pid, "title": body.get("title",""),
                   "language": body.get("language","python"), "code": body.get("code",""),
                   "status": body.get("status","partial"), "tags": json.dumps(body.get("tags",[])),
                   "created": ts()}
            self._exec("INSERT INTO snippets VALUES(:id,:project_id,:title,:language,:code,:status,:tags,:created)", s)
            s["tags"] = body.get("tags",[])
            self._json(s)

        elif "/sprints" in path and path.count("/") == 4:
            pid = path.split("/")[3]
            s   = {"id": uid(), "project_id": pid, "name": body.get("name","Sprint"),
                   "goal": body.get("goal",""), "start_date": body.get("start_date"),
                   "end_date": body.get("end_date"), "status": body.get("status","planning"), "created": ts()}
            self._exec("INSERT INTO sprints VALUES(:id,:project_id,:name,:goal,:start_date,:end_date,:status,:created)", s)
            self._json(s)
        else:
            self._404()

    def do_PUT(self):
        body   = self._body()
        path   = urlparse(self.path).path.rstrip("/")
        parts  = path.split("/")

        if len(parts) == 4 and parts[2] == "projects":
            pid = parts[3]
            allowed = {"name","description","status","tags","docs"}
            updates = {k: (json.dumps(v) if k == "tags" and isinstance(v, list) else v)
                       for k, v in body.items() if k in allowed}
            updates["updated"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            clause = ", ".join(f"{k}=?" for k in updates)
            self._exec(f"UPDATE projects SET {clause} WHERE id=?", (*updates.values(), pid))
            self._json({"ok": True})

        elif len(parts) == 4 and parts[2] == "tasks":
            tid = parts[3]
            allowed = {"title","description","status","priority","sprint_id","start_date","end_date","tags","checklist"}
            updates = {}
            for k, v in body.items():
                if k in allowed:
                    updates[k] = json.dumps(v) if k in ("tags","checklist") and isinstance(v,list) else v
            updates["updated"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            clause = ", ".join(f"{k}=?" for k in updates)
            self._exec(f"UPDATE tasks SET {clause} WHERE id=?", (*updates.values(), tid))
            self._json({"ok": True})

        elif len(parts) == 4 and parts[2] == "sprints":
            sid = parts[3]
            allowed = {"name","goal","start_date","end_date","status"}
            updates = {k: v for k, v in body.items() if k in allowed}
            if updates:
                clause = ", ".join(f"{k}=?" for k in updates)
                self._exec(f"UPDATE sprints SET {clause} WHERE id=?", (*updates.values(), sid))
            self._json({"ok": True})

        elif len(parts) == 4 and parts[2] == "snippets":
            sid = parts[3]
            allowed = {"title","language","code","status","tags"}
            updates = {k: (json.dumps(v) if k == "tags" and isinstance(v,list) else v)
                       for k, v in body.items() if k in allowed}
            if updates:
                clause = ", ".join(f"{k}=?" for k in updates)
                self._exec(f"UPDATE snippets SET {clause} WHERE id=?", (*updates.values(), sid))
            self._json({"ok": True})
        else:
            self._404()

    def do_DELETE(self):
        parts = urlparse(self.path).path.split("/")
        if len(parts) == 4 and parts[2] == "tasks":
            self._exec("DELETE FROM tasks WHERE id=?", (parts[3],))
            self._json({"ok": True})
        else:
            self._404()

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _query(self, sql, params=()):
        with _conn() as c:
            return c.execute(sql, params).fetchall()

    def _exec(self, sql, params=()):
        with _conn() as c:
            c.execute(sql, params)

    def _proj(self, r):
        d = dict(r)
        try:    d["tags"] = json.loads(d.get("tags") or "[]")
        except: d["tags"] = []
        return d

    def _task(self, r):
        d = dict(r)
        for k in ("tags","checklist"):
            try:    d[k] = json.loads(d.get(k) or "[]")
            except: d[k] = []
        return d

    def _snip(self, r):
        d = dict(r)
        try:    d["tags"] = json.loads(d.get("tags") or "[]")
        except: d["tags"] = []
        return d

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length) if length else b"{}"
        try:   return json.loads(raw)
        except: return {}

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _404(self):
        self._json({"error": "not found"}, 404)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _DB_PATH.exists():
        print(f"⚠️  DB not found at {_DB_PATH}")
        print("   Start your agent at least once to initialise it, then rerun pm_web.py")
        sys.exit(1)

    server = HTTPServer(("localhost", PORT), Handler)
    url    = f"http://localhost:{PORT}"
    print(f"✅  PM Suite running → {url}")
    print(f"   DB: {_DB_PATH}")
    print(f"   Press Ctrl+C to stop")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n   Stopped.")