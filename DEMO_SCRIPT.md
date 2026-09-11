# 5-minute manager demo

1. Run `setup_windows.bat` once, then `run_windows.bat`.
2. Login as `admin / ServyAdmin!2026` and show the dashboard/modules.
3. Open **Knowledge Base**. Explain that the demo has 118 Servy-like items but only approved documents are RAG-enabled; malicious/test content can be quarantined or disabled.
4. Logout and login as `freshdairy / FreshDairy!2026`.
5. Open **Customer AI Support**, choose `Milk Analyzer Lab-01`, and ask: `The reading is unstable after cleaning. What should I check?`
6. Show the grounded answer and exact server-side references. Explain that customer mode cannot retrieve confidential documents or another customer's material.
7. Click **No — create service call**. The same interaction becomes a Call Register entry with the customer's question and AI guidance already attempted, SLA context and technician assignment.
8. Logout and login as `engineer / ServyTech!2026`. Open **Call Register** → the new call → use **Call Copilot** to retrieve KB material plus sanitized previous resolutions.
9. Login as `admin` again to demonstrate **Download Excel / Import Excel**, Knowledge Base uploads and Operations modules.
10. For tenant-isolation demonstration, logout and login separately as `aero_admin / AeroAdmin!2026`. Its private tenant data is isolated from the Starlly demo tenant.
