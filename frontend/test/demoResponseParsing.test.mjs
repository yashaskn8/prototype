import test from 'node:test';
import assert from 'node:assert/strict';
import { parseDemoResponse } from '../src/pages/demoGatewayClient.js';

function response(status, body, type) {
  return new Response(body, { status, headers: type ? { 'content-type': type } : {} });
}

test('JSON success returns the data', async () => {
  assert.deepEqual(await parseDemoResponse(response(200, '{"assets":[1]}', 'application/json')), { assets: [1] });
});

test('an HTML 500 page is never parsed as JSON', async () => {
  await assert.rejects(parseDemoResponse(response(500, '<!DOCTYPE html><html>err</html>', 'text/html')),
    err => err.status === 500 && !/Unexpected token/.test(err.message) && /could not complete/.test(err.message));
});

test('an invalid access code gives a clear 401 message', async () => {
  await assert.rejects(parseDemoResponse(response(401, '{"detail":"Invalid access code."}', 'application/json')),
    err => err.status === 401 && err.message === 'Invalid access code.');
  await assert.rejects(parseDemoResponse(response(401, '', 'text/plain')),
    err => /access code was not accepted/.test(err.message));
});

test('a 200 with a non-JSON body and a 204 are handled', async () => {
  await assert.rejects(parseDemoResponse(response(200, '<html></html>', 'text/html')), /unexpected response/);
  assert.deepEqual(await parseDemoResponse(new Response(null, { status: 204 })), {});
});
