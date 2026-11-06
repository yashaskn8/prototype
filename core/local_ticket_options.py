"""Shared validated choices for the isolated local technician ticket workspace."""

TICKET_TYPES = {
    "Breakdown Service", "Installation / Demo", "Service",
    "Preventive Maintenance (PM)", "AMC",
}
SEVERITIES = {"", "Low", "Medium", "High", "Critical"}
MACHINE_STATUSES = {"", "Running", "Intermittent", "Stopped", "Unknown"}
COMPLAINT_TYPES = {
    "Cooling / performance", "Electrical", "Noise / vibration", "Leakage",
    "Installation / commissioning", "Preventive maintenance", "Other",
}
PRIORITIES = [
    ("low", "Low"), ("normal", "Normal"), ("high", "High"), ("critical", "Critical"),
]
