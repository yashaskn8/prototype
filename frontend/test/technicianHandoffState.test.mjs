import test from 'node:test';
import assert from 'node:assert/strict';
import { customerHandoffMessage, handoffBucket, handoffLabel } from '../src/pages/technicianHandoffState.js';

test('queue and assignment are distinct from technician acknowledgement', () => {
  assert.equal(handoffBucket('queued'), 'incoming');
  assert.equal(handoffBucket('assigned'), 'incoming');
  assert.equal(handoffLabel('assigned'), 'Assigned · awaiting acknowledgement');
  assert.match(customerHandoffMessage('queued'), /visible in the authorised technician inbox/);
  assert.match(customerHandoffMessage('assigned'), /awaiting acknowledgement/);
});

test('only persisted acknowledgement state is presented as acknowledged', () => {
  assert.equal(handoffBucket('acknowledged'), 'acknowledged');
  assert.equal(handoffLabel('acknowledged'), 'Acknowledged');
  assert.match(customerHandoffMessage('acknowledged'), /technician has acknowledged/);
  assert.notEqual(handoffBucket('assigned'), handoffBucket('acknowledged'));
});

test('closure review and resolved remain distinct', () => {
  assert.equal(handoffBucket('closure_review'), 'in_progress');
  assert.match(customerHandoffMessage('closure_review'), /not yet marked resolved/);
  assert.equal(handoffBucket('resolved'), 'resolved');
  assert.equal(customerHandoffMessage('resolved'), 'This service request is marked resolved.');
});
