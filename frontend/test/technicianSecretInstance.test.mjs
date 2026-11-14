import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const readSource = (path) => readFile(new URL(path, import.meta.url), 'utf8');

test('technician experience is a distinct access-code instance', async () => {
  const [page, client, app, vite, launcher, gateway, settings, customerPage, customerLauncher] = await Promise.all([
    readSource('../src/pages/TechnicianSecretPage.jsx'),
    readSource('../src/pages/technicianGatewayClient.js'),
    readSource('../src/App.jsx'),
    readSource('../vite.config.js'),
    readSource('../../tools/Start-ServyTechnicianDemo.ps1'),
    readSource('../../core/demo_gateway.py'),
    readSource('../../servy_rag/settings.py'),
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
  assert.match(client, /INBOX_PATH = '\/tech-api\/technician\/inbox\/'/);
  assert.match(page, /technicianCall\(INBOX_PATH/);
  assert.match(page, /ticketPath\(selected\.id, 'update\/'\)/);
  assert.match(page, /downloadAttachment/);
  assert.match(page, /window\.setInterval\(refreshQueue, 8000\)/);
  assert.match(page, /queue refreshes automatically/i);
  assert.doesNotMatch(customerPage, /http:\/\/127\.0\.0\.1:5176\//);
  assert.doesNotMatch(customerPage, /technician_access\.txt/);
  assert.doesNotMatch(customerPage, /RAGExecutionLog|Customer RAG execution log/);
  assert.match(customerPage, /Ticket created/);
  assert.match(launcher, /technician_access\.txt/);
  assert.match(launcher, /VITE_SERVY_TECH_DEMO/);
  // Both demos run the Django gateway against the isolated servy_dev database only,
  // and refuse to start unless the isolation check passes.
  for (const [source, gatewayName, port] of [[launcher, 'technician', 8003], [customerLauncher, 'customer', 8002]]) {
    assert.match(source, /\.env\.mysql\.servy_dev/);
    assert.ok(source.includes(`'SERVY_DEMO_GATEWAY', '${gatewayName}'`) || source.includes(`SERVY_DEMO_GATEWAY = '${gatewayName}'`));
    assert.match(source, /'manage\.py' 'check_dev_database'/);
    assert.ok(source.includes(`'runserver','127.0.0.1:${port}'`) || source.includes(`'runserver', '127.0.0.1:${port}'`));
    assert.doesNotMatch(source, /servy_(real|technician)_demo\.py/);
  }
  assert.match(settings, /MYSQL_DATABASE[^\n]*!= "servy_dev"|"servy_dev"[^\n]*MYSQL_DATABASE/);
  assert.match(customerLauncher, /Get-Content -LiteralPath \$backendErrLog -Tail 25 -Wait/);
  assert.match(customerLauncher, /Live customer RAG events/);
  assert.match(launcher, /Get-Content -LiteralPath \$backendErr -Tail 20 -Wait/);
  assert.match(launcher, /Live technician events/);
  assert.match(gateway, /secrets\.compare_digest/);
  assert.match(gateway, /"127\.0\.0\.1", "::1"/);
  assert.match(gateway, /verify_isolated_database\(/);
  assert.match(gateway, /never logged/);
});

test('technician interface supports status and notes but not ownership edits', async () => {
  const [page, inbox] = await Promise.all([
    readSource('../src/pages/TechnicianSecretPage.jsx'),
    readSource('../../core/technician_inbox.py'),
  ]);
  assert.match(page, /Save technician update/);
  assert.match(page, /Service call description/);
  assert.match(page, /Customer conversation/);
  assert.match(page, /Issue documentation/);
  assert.match(page, /JSON\.stringify\(\{ status, note: notes\.trim\(\) \}\)/);
  const update = inbox.slice(inbox.indexOf('class TechnicianTicketUpdateView'));
  assert.ok(update.length > 0, 'technician update view is missing');
  // Only status and an append-only note are accepted; ownership fields are not.
  assert.match(update, /request\.data\.get\("status"\)/);
  assert.match(update, /request\.data\.get\("note"\)/);
  assert.doesNotMatch(update, /request\.data\.get\("(technician|customer|asset|site|tenant|contact)/);
  assert.match(update, /assigned to another technician/);
  assert.match(update, /CallUpdate\.objects\.create\(/);
  assert.doesNotMatch(update, /technician_notes\s*=/);
});
