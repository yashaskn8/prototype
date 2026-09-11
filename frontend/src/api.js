// API client with automated CSRF handling and credential management

let cachedCsrfToken = null;

export function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

export async function bootstrapCsrf() {
  try {
    const res = await fetch('/api/auth/csrf/', {
      method: 'GET',
      credentials: 'include',
    });
    if (res.ok) {
      const data = await res.json();
      if (data.csrfToken) {
        cachedCsrfToken = data.csrfToken;
      }
    }
  } catch (err) {
    console.warn('Could not bootstrap CSRF token:', err);
  }
  return getCsrfToken();
}

export function getCsrfToken() {
  return cachedCsrfToken || getCookie('csrftoken') || '';
}

export async function request(endpoint, options = {}) {
  const url = endpoint.startsWith('/') ? endpoint : `/api/${endpoint}`;
  const method = (options.method || 'GET').toUpperCase();

  const headers = {
    Accept: 'application/json',
    ...(options.headers || {}),
  };

  // Attach CSRF token for mutating methods
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
    let token = getCsrfToken();
    if (!token) {
      token = await bootstrapCsrf();
    }
    if (token) {
      headers['X-CSRFToken'] = token;
    }
  }

  const isFormData = options.body instanceof FormData;
  if (!isFormData && options.body && typeof options.body === 'object' && !(options.body instanceof Blob)) {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }

  const response = await fetch(url, {
    ...options,
    method,
    headers,
    credentials: 'include',
  });

  // Automatically update cached CSRF if returned in response
  const newCsrf = getCookie('csrftoken');
  if (newCsrf) cachedCsrfToken = newCsrf;

  return response;
}

export const api = {
  async get(endpoint) {
    const res = await request(endpoint, { method: 'GET' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw { status: res.status, ...data };
    return data;
  },

  async post(endpoint, data) {
    const res = await request(endpoint, { method: 'POST', body: data });
    const resData = await res.json().catch(() => ({}));
    if (!res.ok) throw { status: res.status, ...resData };
    return resData;
  },

  async delete(endpoint) {
    const res = await request(endpoint, { method: 'DELETE' });
    const resData = await res.json().catch(() => ({}));
    if (!res.ok) throw { status: res.status, ...resData };
    return resData;
  },

  async upload(endpoint, formData) {
    const res = await request(endpoint, { method: 'POST', body: formData });
    const resData = await res.json().catch(() => ({}));
    if (!res.ok) throw { status: res.status, ...resData };
    return resData;
  },
};
