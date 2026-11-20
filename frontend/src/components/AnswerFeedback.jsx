import React, { useState } from 'react';

export const FEEDBACK_REASONS = [
  ['incorrect', 'Incorrect information'],
  ['missing', 'Missing information'],
  ['unclear', 'Unclear explanation'],
  ['wrong_source', 'Wrong supporting source'],
];

// Thumbs up / down for one saved answer. `save` sends the PUT and resolves to the
// stored feedback; the server binds it to this customer's own answer.
export function AnswerFeedback({ initial = null, save }) {
  const [feedback, setFeedback] = useState(initial);
  const [draft, setDraft] = useState(initial?.rating === 'not_helpful'
    ? { reason: initial.reason || '', comment: initial.comment || '' } : null);
  const [state, setState] = useState('idle');

  async function submit(rating, extra = {}) {
    setState('saving');
    try {
      const saved = await save({ rating, ...extra });
      setFeedback(saved);
      setDraft(null);
      setState('saved');
    } catch {
      setState('error');
    }
  }

  const rating = feedback?.rating;
  return (
    <div className="answer-feedback" aria-label="Rate this answer">
      <span className="answer-feedback-label">Was this answer helpful?</span>
      <button type="button" aria-label="Helpful" aria-pressed={rating === 'helpful'} disabled={state === 'saving'}
        onClick={() => submit('helpful')}>👍</button>
      <button type="button" aria-label="Not helpful" aria-pressed={rating === 'not_helpful' || Boolean(draft)}
        disabled={state === 'saving'} onClick={() => setDraft({ reason: '', comment: '' })}>👎</button>
      {draft && (
        <form className="answer-feedback-details" onSubmit={event => {
          event.preventDefault();
          submit('not_helpful', { reason: draft.reason, comment: draft.comment.trim() });
        }}>
          <label>
            What was wrong? (optional)
            <select value={draft.reason} onChange={event => setDraft({ ...draft, reason: event.target.value })}>
              <option value="">Choose a reason</option>
              {FEEDBACK_REASONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <label>
            Comment (optional)
            <textarea maxLength={500} rows={2} value={draft.comment}
              onChange={event => setDraft({ ...draft, comment: event.target.value })} />
          </label>
          <button type="submit" disabled={state === 'saving'}>Send feedback</button>
          <button type="button" onClick={() => setDraft(null)}>Cancel</button>
        </form>
      )}
      <span role="status" aria-live="polite" className="answer-feedback-status">
        {state === 'saving' && 'Saving…'}
        {state === 'saved' && 'Thank you. Your feedback was saved.'}
        {state === 'error' && 'Your feedback could not be saved. Please try again.'}
      </span>
    </div>
  );
}
