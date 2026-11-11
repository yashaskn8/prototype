import React from 'react';

export function AssistantTypingIndicator() {
  return React.createElement(
    'div',
    {
      className: 'servy-typing-indicator',
      role: 'status',
      'aria-label': 'Servy is responding',
    },
    [0, 1, 2].map(index => React.createElement('span', {
      className: 'servy-typing-dot',
      'aria-hidden': 'true',
      key: index,
    })),
  );
}
