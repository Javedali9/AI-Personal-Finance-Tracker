

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parent / "ledgely_data.db"
GEMINI_MODEL = "gemini-4-hydrogen"
DEFAULT_REFERENCE_DATE = date.today().isoformat()

CURRENCY_META: dict[str, dict[str, Any]] = {
    "INR": {
        "symbol": "₹",
        "label": "INR (₹)",
        "spike_min_delta": 250.0,
        "spike_high_delta": 2500.0,
        "status_threshold": 50.0,
        "default_goal": 35000.0,
    },
    "USD": {
        "symbol": "$",
        "label": "USD ($)",
        "spike_min_delta": 15.0,
        "spike_high_delta": 120.0,
        "status_threshold": 5.0,
        "default_goal": 1200.0,
    },
    "EUR": {
        "symbol": "€",
        "label": "EUR (€)",
        "spike_min_delta": 15.0,
        "spike_high_delta": 120.0,
        "status_threshold": 5.0,
        "default_goal": 1000.0,
    },
    "GBP": {
        "symbol": "£",
        "label": "GBP (£)",
        "spike_min_delta": 12.0,
        "spike_high_delta": 100.0,
        "status_threshold": 5.0,
        "default_goal": 900.0,
    },
}

EXPENSE_CATEGORIES = [
    "Dining & Coffee",
    "Groceries",
    "Transport & Fuel",
    "Subscriptions",
    "Shopping",
    "Utilities & Bills",
    "Housing & Rent",
    "Healthcare",
    "Entertainment",
    "Transfers",
]

ALL_CATEGORIES = [*EXPENSE_CATEGORIES, "Salary & Income"]

SAMPLE_SMS_EXAMPLES = [
    (
        "Food Delivery SMS",
        f"HDFC Bank: Rs.1,120.00 debited from A/c **9012 via UPI to ZOMATO*SUSHI JUNCTION on {DEFAULT_REFERENCE_DATE} (incl. Rs.135 delivery & priority fee).",
    ),
    (
        "Subscription SMS",
        f"ICICI Bank Alert: Recurring mandate of Rs.179.00 for SPOTIFY FAMILY INDIA processed on Card **4821 on {DEFAULT_REFERENCE_DATE}.",
    ),
    (
        "Grocery Receipt SMS",
        f"SBI UPI: Rs.1,945.00 debited from A/c **4821 to BIGBASKET GROCERIES on {DEFAULT_REFERENCE_DATE}.",
    ),
    (
        "Freelance Income SMS",
        f"HDFC Bank: Rs.12,500.00 credited to A/c **9012 via IMPS from DESIGN CONSULTING INVOICE #108 on {DEFAULT_REFERENCE_DATE}.",
    ),
    (
        "Multi-Message Batch (with OTP)",
        f"HDFC UPI: Rs.420.00 paid to BLUE TOKAI COFFEE ROASTERS from A/c **9012 on {DEFAULT_REFERENCE_DATE}.\n"
        f"Your OTP for transaction is 849201. Do not share this OTP with anyone.\n"
        f"Card Alert: Rs.3,499.00 spent on HDFC Card **4821 at NIKE INDIA ONLINE STORE on {DEFAULT_REFERENCE_DATE}.",
    ),
]

DEMO_CURRENCY_SCALE: dict[str, tuple[float, str]] = {
    "INR": (1.0, "Rs."),
    "USD": (0.04, "$"),
    "EUR": (0.036, "EUR "),
    "GBP": (0.032, "GBP "),
}


def _shift_month_key(month_key: str, months_back: int) -> str:
    y_s, m_s = month_key.split("-")
    year = int(y_s)
    month = int(m_s) - months_back
    while month <= 0:
        month += 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def build_demo_transactions(
    reference_date: str, currency: str = "INR", owner_scope: str = "user"
) -> list[dict[str, Any]]:
    base_mk = reference_date[:7]
    try:
        ref_day = min(28, max(1, int(reference_date[8:10])))
    except ValueError:
        ref_day = 21

    mult, prefix = DEMO_CURRENCY_SCALE.get(currency, (1.0, "Rs."))
    scope_tag = hashlib.sha1(owner_scope.encode("utf-8")).hexdigest()[:6]

    def amt(inr_val: float) -> float:
        return float(max(1, round(inr_val * mult)))

    def fmt(val: float) -> str:
        return f"{prefix}{val:,.2f}"

    def mk_date(mk: str, day_num: int) -> str:
        return f"{mk}-{min(28, max(1, day_num)):02d}"

    items: list[dict[str, Any]] = []

    groc_base = [6200, 5950, 6700, 6100, 6400]
    util_base = [2950, 3150, 2850, 3200, 3050]
    trans_base = [2150, 2400, 2250, 2550, 2200]
    dine_base = [1450, 1850, 1550, 2050, 1400]

    for offset in range(5, 0, -1):
        mk = _shift_month_key(base_mk, offset)
        idx = offset - 1
        items.append(
            {
                "id": f"demo_{scope_tag}_m{offset}_salary",
                "merchant": "Monthly Salary",
                "amount": amt(78000),
                "currency": currency,
                "type": "income",
                "category": "Salary & Income",
                "paymentMethod": "Bank Transfer",
                "date": mk_date(mk, 1),
                "rawMessage": "",
                "isAvoidable": False,
                "confidence": 1.0,
                "source": "manual",
            }
        )
        items.append(
            {
                "id": f"demo_{scope_tag}_m{offset}_rent",
                "merchant": "Apartment Rent",
                "amount": amt(22000),
                "currency": currency,
                "type": "expense",
                "category": "Housing & Rent",
                "paymentMethod": "Bank Transfer",
                "date": mk_date(mk, 3),
                "rawMessage": "",
                "isAvoidable": False,
                "confidence": 1.0,
                "source": "manual",
            }
        )
        g_val = amt(groc_base[idx])
        g_dt = mk_date(mk, 8)
        items.append(
            {
                "id": f"demo_{scope_tag}_m{offset}_groc",
                "merchant": "BigBasket Groceries",
                "amount": g_val,
                "currency": currency,
                "type": "expense",
                "category": "Groceries",
                "paymentMethod": "UPI",
                "date": g_dt,
                "rawMessage": f"SBI UPI: {fmt(g_val)} debited to BIGBASKET GROCERIES on {g_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": False,
                "confidence": 0.96,
                "source": "sms-ai",
            }
        )
        u_val = amt(util_base[idx])
        u_dt = mk_date(mk, 12)
        items.append(
            {
                "id": f"demo_{scope_tag}_m{offset}_util",
                "merchant": "City Power & Fiber",
                "amount": u_val,
                "currency": currency,
                "type": "expense",
                "category": "Utilities & Bills",
                "paymentMethod": "Auto-Debit",
                "date": u_dt,
                "rawMessage": f"HDFC AutoPay: {fmt(u_val)} paid to CITY POWER & FIBER on {u_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": False,
                "confidence": 0.96,
                "source": "sms-ai",
            }
        )
        t_val = amt(trans_base[idx])
        t_dt = mk_date(mk, 15)
        items.append(
            {
                "id": f"demo_{scope_tag}_m{offset}_trans",
                "merchant": "Shell Fuel & Metro",
                "amount": t_val,
                "currency": currency,
                "type": "expense",
                "category": "Transport & Fuel",
                "paymentMethod": "Card",
                "date": t_dt,
                "rawMessage": f"Card Alert: {fmt(t_val)} spent at SHELL FUEL STATION on {t_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": False,
                "confidence": 0.95,
                "source": "sms-ai",
            }
        )
        d_val = amt(dine_base[idx])
        d_dt = mk_date(mk, 18)
        items.append(
            {
                "id": f"demo_{scope_tag}_m{offset}_dine",
                "merchant": "Local Cafe & Bistro",
                "amount": d_val,
                "currency": currency,
                "type": "expense",
                "category": "Dining & Coffee",
                "paymentMethod": "UPI",
                "date": d_dt,
                "rawMessage": f"HDFC UPI: {fmt(d_val)} paid to LOCAL BISTRO on {d_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": True,
                "avoidableReason": "Weekend restaurant meal",
                "confidence": 0.94,
                "source": "sms-ai",
            }
        )
        if offset % 2 == 0:
            items.append(
                {
                    "id": f"demo_{scope_tag}_m{offset}_health",
                    "merchant": "Apollo Pharmacy",
                    "amount": amt(1550),
                    "currency": currency,
                    "type": "expense",
                    "category": "Healthcare",
                    "paymentMethod": "UPI",
                    "date": mk_date(mk, 21),
                    "rawMessage": "",
                    "isAvoidable": False,
                    "confidence": 1.0,
                    "source": "manual",
                }
            )
        if offset == 1:
            s1_val = amt(179)
            s1_dt = mk_date(mk, 6)
            items.append(
                {
                    "id": f"demo_{scope_tag}_m1_sub",
                    "merchant": "Spotify Family",
                    "amount": s1_val,
                    "currency": currency,
                    "type": "expense",
                    "category": "Subscriptions",
                    "paymentMethod": "Card **4821",
                    "date": s1_dt,
                    "rawMessage": f"ICICI Alert: {fmt(s1_val)} for SPOTIFY FAMILY on {s1_dt}.",
                    "sender": "BANK-SMS",
                    "isAvoidable": True,
                    "avoidableReason": "Music subscription",
                    "confidence": 0.95,
                    "source": "sms-ai",
                }
            )
            items.append(
                {
                    "id": f"demo_{scope_tag}_m1_shop",
                    "merchant": "Amazon Basics",
                    "amount": amt(1200),
                    "currency": currency,
                    "type": "expense",
                    "category": "Shopping",
                    "paymentMethod": "Card **4821",
                    "date": mk_date(mk, 22),
                    "rawMessage": "",
                    "isAvoidable": False,
                    "confidence": 1.0,
                    "source": "manual",
                }
            )

    def cur_day(d: int) -> str:
        return mk_date(base_mk, min(d, ref_day))

    f_val = amt(12500)
    f_dt = cur_day(15)
    sp_val = amt(179)
    sp_dt = cur_day(4)
    cf_val = amt(480)
    cf_dt = cur_day(6)
    cg_val = amt(6600)
    cg_dt = cur_day(8)
    st_val = amt(999)
    st_dt = cur_day(10)
    cu_val = amt(3150)
    cu_dt = cur_day(12)
    z1_val = amt(1850)
    z1_dt = cur_day(16)
    nk_val = amt(5499)
    nk_dt = cur_day(18)
    ct_val = amt(2350)
    ct_dt = cur_day(20)
    z2_val = amt(2150)
    z2_dt = cur_day(ref_day)

    items.extend(
        [
            {
                "id": f"demo_{scope_tag}_m0_salary",
                "merchant": "Monthly Salary",
                "amount": amt(82000),
                "currency": currency,
                "type": "income",
                "category": "Salary & Income",
                "paymentMethod": "Bank Transfer",
                "date": cur_day(1),
                "rawMessage": "",
                "isAvoidable": False,
                "confidence": 1.0,
                "source": "manual",
            },
            {
                "id": f"demo_{scope_tag}_m0_freelance",
                "merchant": "Design Consulting Payout",
                "amount": f_val,
                "currency": currency,
                "type": "income",
                "category": "Salary & Income",
                "paymentMethod": "IMPS",
                "date": f_dt,
                "rawMessage": f"HDFC Bank: {fmt(f_val)} credited via IMPS from DESIGN CONSULTING on {f_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": False,
                "confidence": 0.98,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_rent",
                "merchant": "Apartment Rent",
                "amount": amt(22000),
                "currency": currency,
                "type": "expense",
                "category": "Housing & Rent",
                "paymentMethod": "Bank Transfer",
                "date": cur_day(3),
                "rawMessage": "",
                "isAvoidable": False,
                "confidence": 1.0,
                "source": "manual",
            },
            {
                "id": f"demo_{scope_tag}_m0_spotify",
                "merchant": "Spotify Family India",
                "amount": sp_val,
                "currency": currency,
                "type": "expense",
                "category": "Subscriptions",
                "paymentMethod": "Card **4821",
                "date": sp_dt,
                "rawMessage": f"ICICI Bank Alert: {fmt(sp_val)} for SPOTIFY FAMILY INDIA on Card **4821 on {sp_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": True,
                "avoidableReason": "Overlapping music subscription",
                "confidence": 0.96,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_coffee",
                "merchant": "Blue Tokai Coffee",
                "amount": cf_val,
                "currency": currency,
                "type": "expense",
                "category": "Dining & Coffee",
                "paymentMethod": "UPI",
                "date": cf_dt,
                "rawMessage": f"HDFC UPI: {fmt(cf_val)} paid to BLUE TOKAI COFFEE ROASTERS on {cf_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": True,
                "avoidableReason": "Specialty cafe drinks",
                "confidence": 0.95,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_groceries",
                "merchant": "BigBasket Groceries",
                "amount": cg_val,
                "currency": currency,
                "type": "expense",
                "category": "Groceries",
                "paymentMethod": "UPI",
                "date": cg_dt,
                "rawMessage": f"SBI UPI: {fmt(cg_val)} debited to BIGBASKET GROCERIES on {cg_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": False,
                "confidence": 0.97,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_streaming",
                "merchant": "Streaming Bundle 4K",
                "amount": st_val,
                "currency": currency,
                "type": "expense",
                "category": "Subscriptions",
                "paymentMethod": "Card **4821",
                "date": st_dt,
                "rawMessage": f"Card Alert: {fmt(st_val)} recurring charge for STREAMING BUNDLE on {st_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": True,
                "avoidableReason": "Unused 4K streaming add-on",
                "confidence": 0.95,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_utilities",
                "merchant": "City Power & Fiber",
                "amount": cu_val,
                "currency": currency,
                "type": "expense",
                "category": "Utilities & Bills",
                "paymentMethod": "Auto-Debit",
                "date": cu_dt,
                "rawMessage": f"HDFC AutoPay: {fmt(cu_val)} paid to CITY POWER & FIBER on {cu_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": False,
                "confidence": 0.97,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_zomato1",
                "merchant": "Zomato Sushi Junction",
                "amount": z1_val,
                "currency": currency,
                "type": "expense",
                "category": "Dining & Coffee",
                "paymentMethod": "UPI",
                "date": z1_dt,
                "rawMessage": f"HDFC Bank: {fmt(z1_val)} debited via UPI to ZOMATO*SUSHI JUNCTION on {z1_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": True,
                "avoidableReason": "Food delivery & priority fees",
                "confidence": 0.96,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_nike",
                "merchant": "Nike India Online",
                "amount": nk_val,
                "currency": currency,
                "type": "expense",
                "category": "Shopping",
                "paymentMethod": "Card **4821",
                "date": nk_dt,
                "rawMessage": f"Card Alert: {fmt(nk_val)} spent on HDFC Card **4821 at NIKE INDIA ONLINE STORE on {nk_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": True,
                "avoidableReason": "Impulse sneaker order",
                "confidence": 0.97,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_transport",
                "merchant": "Shell Fuel & Metro",
                "amount": ct_val,
                "currency": currency,
                "type": "expense",
                "category": "Transport & Fuel",
                "paymentMethod": "Card",
                "date": ct_dt,
                "rawMessage": f"Card Alert: {fmt(ct_val)} spent at SHELL FUEL STATION on {ct_dt}.",
                "sender": "BANK-SMS",
                "isAvoidable": False,
                "confidence": 0.95,
                "source": "sms-ai",
            },
            {
                "id": f"demo_{scope_tag}_m0_zomato2",
                "merchant": "Swiggy Gourmet Feast",
                "amount": z2_val,
                "currency": currency,
                "type": "expense",
                "category": "Dining & Coffee",
                "paymentMethod": "UPI",
                "date": z2_dt,
                "rawMessage": f"HDFC UPI: {fmt(z2_val)} paid to SWIGGY GOURMET on {z2_dt} (incl. surge fee).",
                "sender": "BANK-SMS",
                "isAvoidable": True,
                "avoidableReason": "Weekend delivery surge",
                "confidence": 0.95,
                "source": "sms-ai",
            },
        ]
    )
    return items


# ---------------------------------------------------------------------------
# SQLite Database Layer (Parameterized Queries)
# ---------------------------------------------------------------------------


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=32,
    )
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) ->
