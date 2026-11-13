import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const readSource = (path) => readFile(new URL(path, import.meta.url), 'utf8');

test('technician experience is a distinct access-code instance', async () => {
  const [page, app, vite, launcher, backend, customerPage, customerLauncher] = await Promise.all([
    readSource('../src/pages/TechnicianSecretPage.jsx'),
    readSource('../src/App.jsx'),
    readSource('../vite.config.js'),
    readSource('../../tools/Start-ServyTechnicianDemo.ps1'),
    readSource('../../tools/servy_technician_demo.py'),
    readSource('../src/pages/RealDemoPage.jsx'),
    readSource('../../tools/Start-ServyRealDemo.ps1'),
  ]);

  assert.match(app, /VITE_SERVY_TECH_DEMO === '1'\) return <TechnicianSecretPage \/>/);
  assert.match(vite, /technicianDemo \? 5176/);
  assert.match(vite, /127\.0\.0\.1:8003/);
  assert.match(page, /Technician access code/);
  assert.match(page, /Authorization:/);
  assert.ok(page.includes("Authorization: `Technician ${token}`"));
  assert.equal(page.includes("\\`"), false);
  assert.equal(page.includes("\\${"), false);
  assert.match(page, /tech-api\/tickets/);
  assert.match(page, /technicianNotes/);
  assert.match(page, /downloadAttachment/);
  assert.match(page, /window\.setInterval\(refreshQueue, 8000\)/);
  assert.match(page, /queue refreshes automatically/i);
  assert.doesNotMatch(customerPage, /http:\/\/127\.0\.0\.1:5176\//);
  assert.doesNotMatch(customerPage, /technician_access\.txt/);
  assert.doesNotMatch(customerPage, /RAGExecutionLog|Customer RAG execution log/);
  assert.match(customerPage, /Ticket created/);
  assert.match(launcher, /technician_access\.txt/);
  assert.match(launcher, /VITE_SERVY_TECH_DEMO/);
  assert.match(launcher, /No customer RAG ticket workspace exists yet/);
  assert.match(launcher, /GetFolderPath\(\[System\.Environment\+SpecialFolder\]::LocalApplicationData\)/);
  assert.match(customerLauncher, /GetFolderPath\(\[System\.Environment\+SpecialFolder\]::LocalApplicationData\)/);
  assert.match(launcher, /SERVY_LOCAL_TICKET_DB_PATH/);
  assert.match(customerLauncher, /SERVY_LOCAL_TICKET_DB_PATH/);
  assert.match(customerLauncher, /Get-Content -LiteralPath \$backendOutLog -Tail 25 -Wait/);
  assert.match(customerLauncher, /Live customer RAG events/);
  assert.match(launcher, /Get-Content -LiteralPath \$backendOut -Tail 20 -Wait/);
  assert.match(launcher, /Live technician events/);
  assert.match(backend, /TICKET_ARRIVED new_tickets=/);
  assert.match(backend, /_quiet_queue_poll = True/);
  assert.match(backend, /if not path\.is_file\(\):\s+return \[\]/);
  assert.match(backend, /secrets\.compare_digest/);
  assert.match(backend, /127\.0\.0\.1/);
  assert.match(backend, /created_by = \?/);
  assert.match(backend, /servy-real-demo/);
  assert.match(backend, /never connects to MySQL/i);
});

test('technician interface supports status and notes but not ownership edits', async () => {
  const [page, backend] = await Promise.all([
    readSource('../src/pages/TechnicianSecretPage.jsx'),
    readSource('../../tools/servy_technician_demo.py'),
  ]);
  assert.match(page, /Save technician update/);
  assert.match(page, /Customer report and RAG conversation/);
  assert.match(page, /Issue documentation/);
  assert.match(backend, /"status", "technicianNotes"/);
  assert.match(backend, /Only ticket status and technician notes can be changed/);
  assert.doesNotMatch(backend, /UPDATE ServiceCall/i);
});
