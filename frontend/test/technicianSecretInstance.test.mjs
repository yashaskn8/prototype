import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const readSource = (path) => readFile(new URL(path, import.meta.url), 'utf8');

test('technician experience is a distinct access-code instance', async () => {
  const [page, app, vite, launcher, backend, customerPage] = await Promise.all([
    readSource('../src/pages/TechnicianSecretPage.jsx'),
    readSource('../src/App.jsx'),
    readSource('../vite.config.js'),
    readSource('../../tools/Start-ServyTechnicianDemo.ps1'),
    readSource('../../tools/servy_technician_demo.py'),
    readSource('../src/pages/RealDemoPage.jsx'),
  ]);

  assert.match(app, /VITE_SERVY_TECH_DEMO === '1'\) return <TechnicianSecretPage \/>/);
  assert.match(vite, /technicianDemo \? 5176/);
  assert.match(vite, /127\.0\.0\.1:8003/);
  assert.match(page, /Technician access code/);
  assert.match(page, /Authorization:/);
  assert.match(page, /Authorization: `Technician \\$\{token\}`/);
  assert.doesNotMatch(page, /\\\\`|\\\\\\$\\{/);
  assert.match(page, /tech-api\/tickets/);
  assert.match(page, /technicianNotes/);
  assert.match(page, /downloadAttachment/);
  assert.match(customerPage, /http:\/\/127\.0\.0\.1:5176\//);
  assert.match(customerPage, /technician_access\.txt/);
  assert.match(launcher, /technician_access\.txt/);
  assert.match(launcher, /VITE_SERVY_TECH_DEMO/);
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
