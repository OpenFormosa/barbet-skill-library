"""Prospective source-bound checks found in the fresh continuation audit.

Calendar and legacy-action matches are conservative holds, not corpus accuracy.
Source programs and observations are never executed or repaired here.
"""
from datetime import date
import re

WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
          'september', 'october', 'november', 'december']
DATE = re.compile(r'\b(' + '|'.join(WEEKDAYS) + r')\s*,?\s+(' + '|'.join(MONTHS) +
                  r')\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s+(\d{4})\b', re.I)
LEGACY_ACTION = re.compile(r'^\s*\[[^\]\n]{1,128}\]\s*=>\s*\{')


def continuation_content_holds(source, domain, messages, *, has_tools):
    issues = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant' or not message.get('loss'):
            continue
        content = message['content']
        if (source == 'voidful/agent-sft' and domain == 'tool_use' and has_tools
                and not message.get('tool_calls') and LEGACY_ACTION.match(content)):
            issues.append({'reason': 'legacy_serialized_action_conflicts_with_declared_JSON_interface', 'turn': i})
        if source == 'nvidia/Nemotron-SFT-Agentic-v2' and domain == 'tool_use':
            for match in DATE.finditer(content):
                weekday, month, day, year = match.groups()
                try:
                    actual = date(int(year), MONTHS.index(month.lower()) + 1, int(day)).weekday()
                except ValueError:
                    issues.append({'reason': 'invalid_literal_calendar_date', 'turn': i, 'literal': match.group()})
                    continue
                if actual != WEEKDAYS.index(weekday.lower()):
                    issues.append({'reason': 'literal_weekday_date_mismatch', 'turn': i,
                                   'literal': match.group(), 'actual_weekday': WEEKDAYS[actual]})
    return issues
