document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-auto-submit]').forEach((el) => {
    el.addEventListener('change', () => el.form.submit());
  });
});
