/**
 * wa_bridge.js — Baileys WhatsApp bridge for EVE
 *
 * HOW IT WORKS
 * ─────────────────────────────────────────────────────────────
 *  • Connects to WhatsApp via Baileys (same protocol as WhatsApp Web).
 *  • First run: prints a QR code in the terminal — scan it once with
 *    your phone (WhatsApp > Linked Devices > Link a Device).
 *  • Session is saved to ./wa_session/ — never scan again.
 *  • Exposes a tiny HTTP API on localhost:5005 that main.py talks to:
 *      POST /send   { "to": "393XXXXXXXXX", "text": "..." }
 *      POST /recv   → long-poll, returns { "from": "...", "text": "..." }
 *      GET  /status → { "ready": true/false, "owner": "39..." }
 *
 * REQUIREMENTS (auto-installed by main.py on first run)
 * ─────────────────────────────────────────────────────────────
 *  Node.js ≥ 18  (install from https://nodejs.org — LTS version)
 *  npm packages: @whiskeysockets/baileys  qrcode-terminal  pino
 *
 * USAGE
 * ─────────────────────────────────────────────────────────────
 *  node wa_bridge.js          (main.py starts this automatically)
 */

'use strict';

// Force UTF-8 output on Windows so the QR block-characters render correctly
if (process.platform === 'win32') {
  try { process.stdout.setEncoding('utf8'); } catch (_) {}
  try {
    const { execSync } = require('child_process');
    execSync('chcp 65001', { stdio: 'ignore' });
  } catch (_) {}
}

const http    = require('http');
const fs      = require('fs');
const path    = require('path');

// ── deps (installed by npm install in main.py) ─────────────────────────────
let makeWASocket, useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion;
let Boom, QR;

try {
  const baileys   = require('@whiskeysockets/baileys');
  makeWASocket            = baileys.default || baileys.makeWASocket;
  useMultiFileAuthState   = baileys.useMultiFileAuthState;
  DisconnectReason        = baileys.DisconnectReason;
  fetchLatestBaileysVersion = baileys.fetchLatestBaileysVersion;
  Boom                    = require('@hapi/boom');
  QR                      = require('qrcode-terminal');
} catch (e) {
  console.error('[wa_bridge] Missing deps. Run: npm install');
  process.exit(1);
}

// ── config ─────────────────────────────────────────────────────────────────
const PORT         = 5005;
const SESSION_DIR  = path.join(__dirname, 'wa_session');
const POLL_TIMEOUT = 30_000;   // ms — long-poll timeout
const MAX_QUEUE    = 50;

// ── state ──────────────────────────────────────────────────────────────────
let sock        = null;
let ready       = false;
let ownerJid    = '';           // "393XXXXXXXXX@s.whatsapp.net"
const inbound   = [];           // queue of { from, text }
const waiters   = [];           // pending /recv long-poll resolvers

// ── helpers ────────────────────────────────────────────────────────────────
function bareNumber(jid) {
  // "393XXXXXXXXX@s.whatsapp.net"  →  "393XXXXXXXXX"
  return (jid || '').replace(/@.+$/, '').replace(/:\d+$/, '');
}

function toJid(number) {
  // "393XXXXXXXXX"  →  "393XXXXXXXXX@s.whatsapp.net"
  const clean = number.replace(/\D/g, '');
  return clean.includes('@') ? clean : `${clean}@s.whatsapp.net`;
}

function enqueue(msg) {
  if (inbound.length >= MAX_QUEUE) inbound.shift();
  inbound.push(msg);
  if (waiters.length) waiters.shift()(msg);
}

// ── Baileys connection ──────────────────────────────────────────────────────
async function connect() {
  if (!fs.existsSync(SESSION_DIR)) fs.mkdirSync(SESSION_DIR, { recursive: true });

  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);

  let version;
  try {
    const result = await fetchLatestBaileysVersion();
    version = result.version;
  } catch {
    version = [2, 3000, 1017531287];  // safe fallback
  }

  // Silence Baileys' pino logger — we only want QR / status lines
  const pino = require('pino');
  const logger = pino({ level: 'silent' });

  sock = makeWASocket({
    version,
    auth:   state,
    logger,
    printQRInTerminal: false,    // we print it ourselves for cleaner output
    browser: ['EVE', 'Desktop', '3.0'],
    connectTimeoutMs:  60_000,
    retryRequestDelayMs: 250,
  });

  // ── QR code ──────────────────────────────────────────────────────────────
  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      console.log('\n\x1b[96m[whatsapp] Scan this QR code with your phone:\x1b[0m');
      console.log('\x1b[2m(WhatsApp → ⋮ → Linked Devices → Link a Device)\x1b[0m\n');
      QR.generate(qr, { small: true });
      console.log('\x1b[2m(QR refreshes every ~20s if not scanned)\x1b[0m\n');
    }

    if (connection === 'open') {
      ready    = true;
      ownerJid = sock.user?.id || '';
      const num = bareNumber(ownerJid);
      console.log(`\x1b[92m[whatsapp] ✓ Connected as +${num}\x1b[0m`);
      console.log(`\x1b[2m[whatsapp] Session saved — won't need QR again.\x1b[0m`);
    }

    if (connection === 'close') {
      ready = false;
      const code   = lastDisconnect?.error?.output?.statusCode;
      const reason = Object.keys(DisconnectReason).find(k => DisconnectReason[k] === code);

      if (code === DisconnectReason.loggedOut) {
        console.log('\x1b[91m[whatsapp] Logged out. Delete wa_session/ and restart to re-scan.\x1b[0m');
        // wipe session so next restart shows fresh QR
        fs.rmSync(SESSION_DIR, { recursive: true, force: true });
        process.exit(0);
      } else {
        console.log(`\x1b[33m[whatsapp] Disconnected (${reason || code}) — reconnecting…\x1b[0m`);
        setTimeout(connect, 3000);
      }
    }
  });

  // ── save credentials whenever they update ────────────────────────────────
  sock.ev.on('creds.update', saveCreds);

  // ── receive messages ──────────────────────────────────────────────────────
  sock.ev.on('messages.upsert', ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const msg of messages) {
      if (msg.key.fromMe)           continue;  // ignore own messages
      if (!msg.message)             continue;  // ignore empty/revoked

      const from = msg.key.remoteJid || '';
      if (!from || from.endsWith('@g.us')) continue;  // ignore group messages

      // extract text from any message type
      const text = (
        msg.message?.conversation ||
        msg.message?.extendedTextMessage?.text ||
        msg.message?.imageMessage?.caption ||
        msg.message?.videoMessage?.caption ||
        ''
      ).trim();

      if (!text) continue;

      // remember owner so /send knows who to reply to
      if (!ownerJid) ownerJid = from;

      console.log(`\x1b[2m[whatsapp] ← ${bareNumber(from)}: ${text.slice(0, 60)}\x1b[0m`);
      enqueue({ from: bareNumber(from), text });
    }
  });
}

// ── HTTP API ───────────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];

  // ── GET /status ──────────────────────────────────────────────────────────
  if (req.method === 'GET' && url === '/status') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    const ownerOut = ownerJid || (sock && sock.user && sock.user.id ? sock.user.id : '');
    res.end(JSON.stringify({ ready, owner: bareNumber(ownerOut) }));
    return;
  }

  // ── POST /send  { "to": "393...", "text": "..." } ────────────────────────
  if (req.method === 'POST' && url === '/send') {
    let body = '';
    req.on('data', d => body += d);
    req.on('end', async () => {
      try {
        const { to, text } = JSON.parse(body);
        if (!ready || !sock) {
          res.writeHead(503, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'not connected' }));
          return;
        }
        // Resolve the exact JID registered on WhatsApp for this number
        let jid = toJid(to);
        try {
          const [result] = await sock.onWhatsApp(jid);
          if (result && result.exists && result.jid) {
            jid = result.jid;
            console.log(`\x1b[2m[whatsapp] resolved JID: ${jid}\x1b[0m`);
          } else {
            console.warn(`\x1b[33m[whatsapp] number ${to} not on WhatsApp or JID lookup failed\x1b[0m`);
          }
        } catch (lookupErr) {
          console.warn(`\x1b[33m[whatsapp] JID lookup error (proceeding anyway): ${lookupErr.message}\x1b[0m`);
        }
        // chunk long messages (WhatsApp limit ~4096 chars)
        const chunks = [];
        for (let i = 0; i < text.length; i += 3900) chunks.push(text.slice(i, i + 3900));
        for (const chunk of chunks) {
          await sock.sendMessage(jid, { text: chunk });
        }
        console.log(`\x1b[2m[whatsapp] → ${to} (${jid}): ${text.slice(0, 60)}\x1b[0m`);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, jid }));
      } catch (e) {
        console.error('[wa_bridge] /send error:', e.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // ── POST /recv  — long-poll: waits up to POLL_TIMEOUT for next message ───
  if (req.method === 'POST' && url === '/recv') {
    // If there's already a queued message, return it immediately
    if (inbound.length > 0) {
      const msg = inbound.shift();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(msg));
      return;
    }
    // Otherwise wait
    const timer = setTimeout(() => {
      const idx = waiters.indexOf(resolve);
      if (idx !== -1) waiters.splice(idx, 1);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ text: null }));   // timeout — no message
    }, POLL_TIMEOUT);

    const resolve = (msg) => {
      clearTimeout(timer);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(msg));
    };
    waiters.push(resolve);
    return;
  }

  res.writeHead(404);
  res.end('not found');
});

function startServer(attempt) {
  server.listen(PORT, '127.0.0.1', () => {
    console.log(`\x1b[2m[wa_bridge] HTTP API on 127.0.0.1:${PORT}\x1b[0m`);
  });
}

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.warn(`\x1b[33m[wa_bridge] Port ${PORT} busy — retrying in 2s…\x1b[0m`);
    setTimeout(() => {
      server.close();
      startServer();
    }, 2000);
  } else {
    console.error('[wa_bridge] Server error:', err.message);
    process.exit(1);
  }
});

startServer();

// ── start ──────────────────────────────────────────────────────────────────
connect().catch(e => {
  console.error('[wa_bridge] Fatal:', e);
  process.exit(1);
});

// graceful shutdown
process.on('SIGTERM', () => { server.close(); process.exit(0); });
process.on('SIGINT',  () => { server.close(); process.exit(0); });