"""Structured ServiceCall description built from a complete customer conversation.

The description is built from every persisted message of the conversation, in
chronological order, without truncation. It reports what the customer said and
what the assistant showed from approved documents; it never states a diagnosis.
Statements are labelled as customer-reported, documented, or unconfirmed.
"""
import re

SECTION_TITLES = (
    "1. Customer-reported problem",
    "2. Asset",
    "3. Symptoms and observations reported by the customer",
    "4. Questions asked during the conversation",
    "5. Customer responses",
    "6. Troubleshooting discussed (not confirmed as performed)",
    "7. Approved-manual findings and citations",
    "8. Unresolved",
    "9. For technician investigation",
)

_UNANSWERED_TYPES = {"INSUFFICIENT_EVIDENCE", "CLARIFICATION_REQUIRED", "ESCALATION_REQUIRED"}


def _clean(text):
    return " ".join(str(text or "").split())


def _contract(message):
    contract = message.structured_contract if isinstance(message.structured_contract, dict) else {}
    return contract


def _questions_from(message):
    """Clarifying questions the assistant asked in this message."""
    contract = _contract(message)
    questions = []
    for key in ("clarification_question", "follow_up_question"):
        value = _clean(contract.get(key))
        if value:
            questions.append(value)
    presentation = contract.get("presentation") if isinstance(contract.get("presentation"), dict) else {}
    value = _clean(presentation.get("follow_up_question"))
    if value:
        questions.append(value)
    # The composed conversational reply marks its clarifying question explicitly.
    response = presentation.get("response") if isinstance(presentation.get("response"), dict) else {}
    for segment in response.get("segments") or []:
        if isinstance(segment, dict) and segment.get("kind") == "question" and _clean(segment.get("text")):
            questions.append(_clean(segment["text"]))
    if not questions and response.get("segments"):
        return []
    if not questions:
        questions = [s.strip() for s in re.split(r"(?<=[?])\s+", _clean(message.message_text)) if s.strip().endswith("?")]
    return list(dict.fromkeys(questions))


def _actions_from(message):
    contract = _contract(message)
    actions = []
    for item in contract.get("customer_safe_actions") or []:
        text = _clean(item.get("action") if isinstance(item, dict) else item)
        if text:
            actions.append(text)
    return actions


def _citations_from(message):
    """(title, heading) pairs the assistant cited, from the saved contract and references."""
    contract = _contract(message)
    pairs = []
    sources = list(contract.get("citations") or [])
    presentation = contract.get("presentation") if isinstance(contract.get("presentation"), dict) else {}
    sources += list(presentation.get("citations") or [])
    sources += list(message.source_refs or []) if isinstance(message.source_refs, list) else []
    for ref in sources:
        if not isinstance(ref, dict) or ref.get("source_kind") == "past_resolution":
            continue
        title = _clean(ref.get("title") or ref.get("reference"))
        heading = _clean(ref.get("heading"))
        if title:
            pairs.append((title, heading))
    return list(dict.fromkeys(pairs))


def build_conversation_ticket_description(conversation):
    """Return {"complaint": str, "description": str, "sections": {title: [lines]}, "message_count": int}.

    Reads the complete conversation (all messages, chronological). Raises
    ValueError when the conversation has no customer-reported text.
    """
    messages = list(conversation.messages.order_by("created_at", "id"))
    customer_messages = [m for m in messages if m.sender_role == "customer" and _clean(m.message_text)]
    if not customer_messages:
        raise ValueError("A customer-reported issue is required before creating a service ticket.")
    assistant_messages = [m for m in messages if m.sender_role == "assistant"]

    sections = {title: [] for title in SECTION_TITLES}
    complaint = _clean(customer_messages[0].message_text)
    sections[SECTION_TITLES[0]].append(complaint)

    asset = conversation.asset
    if asset is not None:
        parts = [f"Asset ID {asset.id}", asset.name]
        if asset.asset_code:
            parts.append(f"code {asset.asset_code}")
        if asset.model_number:
            parts.append(f"model {asset.model_number}")
        sections[SECTION_TITLES[1]].append(", ".join(p for p in parts if p))
    else:
        sections[SECTION_TITLES[1]].append("No asset is associated with this conversation.")
    if conversation.site_id:
        sections[SECTION_TITLES[1]].append(f"Site: {conversation.site.name}")

    for index, message in enumerate(customer_messages, start=1):
        sections[SECTION_TITLES[2]].append(f"[{index}] {_clean(message.message_text)} (customer-reported, not independently verified)")

    asked, answered_after = [], []
    for position, message in enumerate(messages):
        if message.sender_role != "assistant":
            continue
        for question in _questions_from(message):
            asked.append(question)
            reply = next((m for m in messages[position + 1:] if m.sender_role == "customer"), None)
            answered_after.append((question, reply))
    sections[SECTION_TITLES[3]] = [f"Q: {q}" for q in dict.fromkeys(asked)] or ["No clarifying questions were asked."]
    responses = [f"Re: {q} -> {_clean(r.message_text)}" for q, r in answered_after if r is not None]
    later = [_clean(m.message_text) for m in customer_messages[1:]]
    sections[SECTION_TITLES[4]] = responses or ([f"{t}" for t in later] or ["No follow-up responses from the customer."])

    actions = []
    for message in assistant_messages:
        actions += _actions_from(message)
    sections[SECTION_TITLES[5]] = (
        [f"Suggested: {a} (completion not confirmed)" for a in dict.fromkeys(actions)]
        or ["No customer troubleshooting steps were suggested."])

    findings = []
    for number, message in enumerate(assistant_messages, start=1):
        answer = _clean(message.message_text)
        cites = _citations_from(message)
        if answer:
            findings.append(f"Assistant answer {number}: {answer}")
        for title, heading in cites:
            findings.append(f"   Source: {title}" + (f" - {heading}" if heading else ""))
    sections[SECTION_TITLES[6]] = findings or ["No approved-manual evidence was shown."]

    unresolved = ["The customer requested a service call; the issue is not confirmed as resolved."]
    last_contract = _contract(assistant_messages[-1]) if assistant_messages else {}
    if last_contract.get("response_type") in _UNANSWERED_TYPES:
        unresolved.append(f"Last assistant response: {last_contract.get('response_type')} (no source-backed answer).")
    open_questions = [q for q, r in answered_after if r is None]
    unresolved += [f"Unanswered question: {q}" for q in dict.fromkeys(open_questions)]
    sections[SECTION_TITLES[7]] = unresolved

    investigate = ["No diagnosis has been confirmed. Verify the customer-reported symptoms on site."]
    for message in assistant_messages:
        for item in _contract(message).get("technician_only_actions") or []:
            text = _clean(item.get("action") if isinstance(item, dict) else item)
            if text:
                investigate.append(f"Technician-only: {text}")
    sections[SECTION_TITLES[8]] = list(dict.fromkeys(investigate))

    lines = [f"Customer conversation {conversation.conversation_uuid} ({len(messages)} messages)"]
    for title in SECTION_TITLES:
        lines.append("")
        lines.append(title)
        lines += [f"- {item}" for item in sections[title]]
    return {"complaint": complaint, "description": "\n".join(lines), "sections": sections,
            "message_count": len(messages)}


REQUIRED_CONTACT_FIELDS = ("contact_name", "phone_or_email")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?[0-9][0-9 ()-]{6,}$")


def missing_contact_fields(customer):
    """Fields a technician needs to reach the customer, taken only from the customer record."""
    missing = []
    if not _clean(getattr(customer, "contact_name", "")):
        missing.append("contact_name")
    phone = _clean(getattr(customer, "phone", ""))
    email = _clean(getattr(customer, "email", ""))
    if not ((phone and _PHONE.match(phone)) or (email and _EMAIL.match(email))):
        missing.append("phone_or_email")
    return missing
