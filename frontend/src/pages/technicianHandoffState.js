export function handoffBucket(state) {
  if (state === 'resolved') return 'resolved';
  if (state === 'in_progress' || state === 'closure_review') return 'in_progress';
  if (state === 'acknowledged') return 'acknowledged';
  return 'incoming';
}

export function handoffLabel(state) {
  return ({
    queued: 'Queued',
    awaiting_assignment: 'Awaiting assignment',
    assigned: 'Assigned · awaiting acknowledgement',
    acknowledged: 'Acknowledged',
    in_progress: 'In progress',
    closure_review: 'Closure review',
    resolved: 'Resolved',
  })[state] || String(state || 'Open').replaceAll('_', ' ');
}

export function customerHandoffMessage(state) {
  return ({
    queued: 'Your request is visible in the authorised technician inbox. The status will update when a technician acknowledges it.',
    assigned: 'Your request is assigned to a technician and is awaiting acknowledgement.',
    acknowledged: 'A technician has acknowledged your service request.',
    in_progress: 'A technician is working on your service request.',
    resolved: 'This service request is marked resolved.',
    closure_review: 'The technician marked this request for closure review. It is not yet marked resolved.',
    awaiting_assignment: 'Your service request was created and is awaiting an available technician.',
  })[state] || 'Your service request was created. Its current status is available in Service Tickets.';
}
