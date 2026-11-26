#!/usr/bin/env python3
"""
Ledgely — Single-File Python Streamlit Edition
==============================================
An AI-powered personal finance, SMS expense parser, and monthly/annual tracker
built with Streamlit, Pandas, SQLite, and the Google GenAI Python SDK.

Quick Start:
  1. Install dependencies:
       pip install streamlit google-genai pandas
  2. Set your Gemini API key in your environment:
       export GEMINI_API_KEY="your_api_key_here"   # macOS/Linux
       set GEMINI_API_KEY=your_api_key_here        # Windows CMD
  3. Run the dashboard:
       streamlit run ledgely_streamlit_app.py
"""

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


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    _, candidate_hex = hash_password(password, salt_hex)
    return hmac.compare_digest(candidate_hex, expected_hash_hex)


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                owner_username TEXT NOT NULL DEFAULT 'default',
                merchant TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                tx_type TEXT NOT NULL CHECK(tx_type IN ('expense', 'income')),
                category TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                tx_date TEXT NOT NULL,
                raw_message TEXT NOT NULL DEFAULT '',
                sender TEXT NOT NULL DEFAULT '',
                is_avoidable INTEGER NOT NULL DEFAULT 0,
                avoidable_reason TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL
            )
            """
        )
        tx_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(transactions)").fetchall()
        }
        if "owner_username" not in tx_cols:
            conn.execute(
                "ALTER TABLE transactions ADD COLUMN owner_username TEXT NOT NULL DEFAULT 'default'"
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                owner_username TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (owner_username, key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_ai_reports (
                owner_username TEXT NOT NULL,
                month_key TEXT NOT NULL,
                currency TEXT NOT NULL,
                report_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (owner_username, month_key)
            )
            """
        )


def has_any_users() -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
    return bool(row and int(row["cnt"]) > 0)


def create_user(username: str, display_name: str, password: str) -> tuple[bool, str]:
    clean_user = username.strip().lower()
    clean_name = display_name.strip() or clean_user
    if not re.match(r"^[a-zA-Z0-9_.-]{3,32}$", clean_user):
        return False, "Username must be 3–32 characters (letters, numbers, _, ., -)."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    first_account = not has_any_users()
    salt_hex, hash_hex = hash_password(password)
    try:
        with get_conn() as conn:
            existing = conn.execute(
                "SELECT username FROM users WHERE username = ?", (clean_user,)
            ).fetchone()
            if existing:
                return False, "That username is already taken."

            conn.execute(
                """
                INSERT INTO users (username, display_name, password_salt, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_user, clean_name[:60], salt_hex, hash_hex, datetime.utcnow().isoformat()),
            )
            conn.execute(
                "INSERT OR IGNORE INTO user_settings (owner_username, key, value) VALUES (?, ?, ?)",
                (clean_user, "currency", "INR"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO user_settings (owner_username, key, value) VALUES (?, ?, ?)",
                (clean_user, "savings_goal", "35000"),
            )
            if first_account:
                conn.execute(
                    "UPDATE transactions SET owner_username = ? WHERE owner_username = 'default'",
                    (clean_user,),
                )
        return True, "Account created. You are now signed in."
    except sqlite3.Error as exc:
        return False, f"Database error creating user: {exc}"


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    clean_user = username.strip().lower()
    if not clean_user or not password:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT username, display_name, password_salt, password_hash FROM users WHERE username = ?",
            (clean_user,),
        ).fetchone()
    if not row:
        return None
    if verify_password(password, str(row["password_salt"]), str(row["password_hash"])):
        return {
            "username": str(row["username"]),
            "display_name": str(row["display_name"]),
        }
    return None


def get_setting(username: str, key: str, default: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM user_settings WHERE owner_username = ? AND key = ?",
            (username, key),
        ).fetchone()
    return str(row["value"]) if row else default


def set_setting(username: str, key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_settings (owner_username, key, value) VALUES (?, ?, ?)",
            (username, key, value),
        )


def fetch_transactions(username: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM transactions
            WHERE owner_username = ?
            ORDER BY tx_date DESC, created_at DESC
            """,
            (username,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "merchant": r["merchant"],
            "amount": float(r["amount"]),
            "currency": r["currency"],
            "type": r["tx_type"],
            "category": r["category"],
            "paymentMethod": r["payment_method"],
            "date": r["tx_date"],
            "rawMessage": r["raw_message"],
            "sender": r["sender"],
            "isAvoidable": bool(r["is_avoidable"]),
            "avoidableReason": r["avoidable_reason"],
            "confidence": float(r["confidence"]),
            "source": r["source"],
        }
        for r in rows
    ]


def insert_transactions(username: str, items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    now_iso = datetime.utcnow().isoformat()
    rows = []
    for item in items:
        rows.append(
            (
                str(item.get("id") or f"tx_{uuid.uuid4().hex[:10]}"),
                username,
                str(item.get("merchant") or "Unknown")[:120],
                round(abs(float(item.get("amount") or 0.0)), 2),
                str(item.get("currency") or "INR")[:8],
                "income" if item.get("type") == "income" else "expense",
                str(item.get("category") or "Shopping")[:60],
                str(item.get("paymentMethod") or "Manual Entry")[:60],
                str(item.get("date") or DEFAULT_REFERENCE_DATE)[:10],
                str(item.get("rawMessage") or "")[:1000],
                str(item.get("sender") or "")[:60],
                1 if item.get("isAvoidable") and item.get("type") != "income" else 0,
                str(item.get("avoidableReason") or "")[:200],
                float(item.get("confidence", 1.0)),
                str(item.get("source") or "manual")[:30],
                now_iso,
            )
        )
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO transactions (
                id, owner_username, merchant, amount, currency, tx_type, category, payment_method,
                tx_date, raw_message, sender, is_avoidable, avoidable_reason,
                confidence, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def update_transaction_fields(
    username: str, tx_id: str, category: str, is_avoidable: bool, avoidable_reason: str
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE transactions
            SET category = ?, is_avoidable = ?, avoidable_reason = ?
            WHERE id = ? AND owner_username = ?
            """,
            (category, 1 if is_avoidable else 0, avoidable_reason[:200], tx_id, username),
        )


def delete_transaction(username: str, tx_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM transactions WHERE id = ? AND owner_username = ?",
            (tx_id, username),
        )


def clear_all_transactions(username: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM transactions WHERE owner_username = ?",
            (username,),
        )
        conn.execute(
            "DELETE FROM user_ai_reports WHERE owner_username = ?",
            (username,),
        )


def load_saved_ai_report(username: str, month_key: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT report_json FROM user_ai_reports WHERE owner_username = ? AND month_key = ?",
            (username, month_key),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["report_json"])
    except json.JSONDecodeError:
        return None


def save_ai_report(
    username: str, month_key: str, currency: str, report: dict[str, Any]
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_ai_reports (owner_username, month_key, currency, report_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, month_key, currency, json.dumps(report), datetime.utcnow().isoformat()),
        )


# ---------------------------------------------------------------------------
# Formatting & Financial Analytics
# ---------------------------------------------------------------------------


def format_currency(amount: float, currency: str = "INR") -> str:
    symbol = CURRENCY_META.get(currency, CURRENCY_META["INR"])["symbol"]
    sign = "-" if amount < 0 else ""
    return f"{sign}{symbol}{abs(amount):,.2f}"


def get_previous_month_key(month_key: str) -> str:
    year_s, month_s = month_key.split("-")
    year = int(year_s)
    month = int(month_s) - 1
    if month == 0:
        month = 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def get_month_label(month_key: str) -> str:
    try:
        dt = datetime.strptime(f"{month_key}-01", "%Y-%m-%d")
        return dt.strftime("%b %Y")
    except ValueError:
        return month_key


def calculate_monthly_metrics(
    transactions: list[dict[str, Any]],
    month_key: str,
    currency: str = "INR",
    reference_day: int = 21,
) -> dict[str, Any]:
    meta = CURRENCY_META.get(currency, CURRENCY_META["INR"])
    prev_key = get_previous_month_key(month_key)

    curr_tx = [t for t in transactions if t["date"].startswith(month_key)]
    prev_tx = [t for t in transactions if t["date"].startswith(prev_key)]

    curr_income = sum(t["amount"] for t in curr_tx if t["type"] == "income")
    prev_income = sum(t["amount"] for t in prev_tx if t["type"] == "income")

    curr_exp_tx = [t for t in curr_tx if t["type"] == "expense"]
    prev_exp_tx = [t for t in prev_tx if t["type"] == "expense"]

    curr_expenses = sum(t["amount"] for t in curr_exp_tx)
    prev_expenses = sum(t["amount"] for t in prev_exp_tx)

    expense_delta = curr_expenses - prev_expenses
    expense_delta_pct = (
        (expense_delta / prev_expenses) * 100.0 if prev_expenses > 0 else None
    )

    net_savings = curr_income - curr_expenses
    prev_net_savings = prev_income - prev_expenses
    savings_rate = (net_savings / curr_income) * 100.0 if curr_income > 0 else 0.0
    prev_savings_rate = (
        (prev_net_savings / prev_income) * 100.0 if prev_income > 0 else 0.0
    )

    avoidable_tx = sorted(
        [t for t in curr_exp_tx if t["isAvoidable"]],
        key=lambda x: x["amount"],
        reverse=True,
    )
    recoverable_total = sum(t["amount"] for t in avoidable_tx)

    category_comparisons: list[dict[str, Any]] = []
    for cat in EXPENSE_CATEGORIES:
        c_list = [t for t in curr_exp_tx if t["category"] == cat]
        p_list = [t for t in prev_exp_tx if t["category"] == cat]
        c_amt = sum(t["amount"] for t in c_list)
        p_amt = sum(t["amount"] for t in p_list)
        if c_amt == 0 and p_amt == 0:
            continue
        delta = c_amt - p_amt
        delta_pct = (delta / p_amt) * 100.0 if p_amt > 0 else None
        avoid_amt = sum(t["amount"] for t in c_list if t["isAvoidable"])
        if p_amt == 0 and c_amt > 0:
            status = "new"
        elif delta > meta["status_threshold"]:
            status = "increased"
        elif delta < -meta["status_threshold"]:
            status = "decreased"
        else:
            status = "unchanged"
        category_comparisons.append(
            {
                "category": cat,
                "currentAmount": c_amt,
                "previousAmount": p_amt,
                "deltaAmount": delta,
                "deltaPercent": delta_pct,
                "transactionCount": len(c_list),
                "avoidableAmount": avoid_amt,
                "status": status,
            }
        )

    category_comparisons.sort(
        key=lambda x: max(x["currentAmount"], x["previousAmount"]), reverse=True
    )

    spike_alerts: list[dict[str, Any]] = []
    for comp in category_comparisons:
        if (
            comp["status"] in ("increased", "new")
            and comp["deltaAmount"] >= meta["spike_min_delta"]
        ):
            merchants = list(
                dict.fromkeys(
                    t["merchant"]
                    for t in sorted(
                        [x for x in curr_exp_tx if x["category"] == comp["category"]],
                        key=lambda z: z["amount"],
                        reverse=True,
                    )
                )
            )[:3]
            severity = (
                "high"
                if (comp["deltaPercent"] is not None and comp["deltaPercent"] >= 25)
                or comp["deltaAmount"] >= meta["spike_high_delta"]
                else "moderate"
            )
            spike_alerts.append(
                {
                    "id": f"spike-{comp['category']}",
                    "category": comp["category"],
                    "currentAmount": comp["currentAmount"],
                    "previousAmount": comp["previousAmount"],
                    "deltaAmount": comp["deltaAmount"],
                    "deltaPercent": comp["deltaPercent"],
                    "severity": severity,
                    "topMerchants": merchants,
                }
            )
    spike_alerts.sort(key=lambda x: x["deltaAmount"], reverse=True)

    rent_amt = sum(t["amount"] for t in curr_exp_tx if t["category"] == "Housing & Rent")
    variable_amt = max(0.0, curr_expenses - rent_amt)
    daily_burn = variable_amt / reference_day if reference_day > 0 else 0.0
    projected_spend = rent_amt + daily_burn * 30.0

    return {
        "currentMonthKey": month_key,
        "previousMonthKey": prev_key,
        "currentMonthLabel": get_month_label(month_key),
        "previousMonthLabel": get_month_label(prev_key),
        "currentIncome": curr_income,
        "previousIncome": prev_income,
        "currentExpenses": curr_expenses,
        "previousExpenses": prev_expenses,
        "expenseDeltaAmount": expense_delta,
        "expenseDeltaPercent": expense_delta_pct,
        "netSavings": net_savings,
        "previousNetSavings": prev_net_savings,
        "savingsRate": savings_rate,
        "previousSavingsRate": prev_savings_rate,
        "recoverableAvoidableTotal": recoverable_total,
        "avoidableTransactions": avoidable_tx,
        "categoryComparisons": category_comparisons,
        "spikeAlerts": spike_alerts,
        "projectedMonthEndSpend": projected_spend,
    }


def calculate_six_month_trends(
    transactions: list[dict[str, Any]], current_month_key: str
) -> list[dict[str, Any]]:
    keys: list[str] = []
    cursor = current_month_key
    for _ in range(6):
        keys.insert(0, cursor)
        cursor = get_previous_month_key(cursor)

    rows = []
    for k in keys:
        m_tx = [t for t in transactions if t["date"].startswith(k)]
        exp_tx = [t for t in m_tx if t["type"] == "expense"]
        expenses = sum(t["amount"] for t in exp_tx)
        avoidable = sum(t["amount"] for t in exp_tx if t["isAvoidable"])
        essential = max(0.0, expenses - avoidable)
        income = sum(t["amount"] for t in m_tx if t["type"] == "income")
        rows.append(
            {
                "monthKey": k,
                "monthLabel": get_month_label(k),
                "Income": income,
                "Essential": essential,
                "Avoidable": avoidable,
                "Expenses": expenses,
                "NetSavings": income - expenses,
            }
        )
    return rows


def calculate_year_breakdown(
    transactions: list[dict[str, Any]], year: int
) -> list[dict[str, Any]]:
    rows = []
    for idx in range(1, 13):
        month_key = f"{year:04d}-{idx:02d}"
        m_tx = [t for t in transactions if t["date"].startswith(month_key)]
        exp_tx = [t for t in m_tx if t["type"] == "expense"]
        inc_tx = [t for t in m_tx if t["type"] == "income"]

        expenses = sum(t["amount"] for t in exp_tx)
        avoidable = sum(t["amount"] for t in exp_tx if t["isAvoidable"])
        essential = max(0.0, expenses - avoidable)
        income = sum(t["amount"] for t in inc_tx)
        net_savings = income - expenses
        savings_rate = (net_savings / income) * 100.0 if income > 0 else 0.0

        by_cat: dict[str, float] = {}
        for t in exp_tx:
            by_cat[t["category"]] = by_cat.get(t["category"], 0.0) + t["amount"]
        top_cat = max(by_cat.items(), key=lambda kv: kv[1])[0] if by_cat else "—"

        rows.append(
            {
                "monthKey": month_key,
                "monthLabel": get_month_label(month_key),
                "shortLabel": datetime(year, idx, 1).strftime("%b"),
                "income": income,
                "expenses": expenses,
                "essentialExpenses": essential,
                "avoidableExpenses": avoidable,
                "netSavings": net_savings,
                "savingsRate": savings_rate,
                "topCategory": top_cat,
                "transactionCount": len(m_tx),
            }
        )
    return rows


def calculate_weekly_avoidable_digest(
    transactions: list[dict[str, Any]], month_key: str
) -> list[dict[str, Any]]:
    month_exp = [
        t
        for t in transactions
        if t["type"] == "expense" and t["date"].startswith(month_key)
    ]
    ranges = [
        ("Week 1", 1, 7),
        ("Week 2", 8, 14),
        ("Week 3", 15, 21),
        ("Week 4", 22, 31),
    ]
    short_m = get_month_label(month_key).split(" ")[0]
    buckets = []
    for label, start, end in ranges:
        w_all = []
        for t in month_exp:
            try:
                day_num = int(t["date"][8:10])
            except ValueError:
                continue
            if start <= day_num <= end:
                w_all.append(t)
        avoidable = sorted(
            [t for t in w_all if t["isAvoidable"]],
            key=lambda x: x["amount"],
            reverse=True,
        )
        total_exp = sum(t["amount"] for t in w_all)
        avoid_total = sum(t["amount"] for t in avoidable)
        if total_exp > 0 or avoid_total > 0:
            buckets.append(
                {
                    "weekLabel": label,
                    "rangeLabel": f"{short_m} {start}–{'End' if end == 31 else end}",
                    "weeklyTotalExpenses": total_exp,
                    "avoidableTotal": avoid_total,
                    "avoidableSharePct": (avoid_total / total_exp) * 100.0
                    if total_exp > 0
                    else 0.0,
                    "items": avoidable,
                }
            )
    return list(reversed(buckets))


# ---------------------------------------------------------------------------
# Gemini AI Functions (Google GenAI Python SDK)
# ---------------------------------------------------------------------------


def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "MY_GEMINI_API_KEY":
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it in your terminal before running Streamlit."
        )
    return genai.Client(api_key=api_key)


def parse_sms_with_gemini(
    messages: list[str], preferred_currency: str, reference_date: str
) -> tuple[list[dict[str, Any]], int]:
    cleaned = [m.strip() for m in messages if m.strip()][:30]
    if not cleaned:
        return [], 0

    client = get_gemini_client()
    prompt = "\n".join(
        [
            f"Reference current date: {reference_date}.",
            f"Default currency if unspecified: {preferred_currency}.",
            f"Allowed categories: {', '.join(ALL_CATEGORIES)}.",
            "Parse each phone SMS / bank notification / UPI alert below into transactions.",
            "Rules:",
            "- Ignore OTPs, login alerts, or promotional spam (set isTransaction: false).",
            "- Extract positive numeric amount and currency code (INR, USD, EUR, or GBP).",
            "- type: 'income' for salary/credits/refunds, 'expense' for debits/purchases/bills.",
            "- Clean merchant name and select the single best category from the allowed list.",
            "- Set isAvoidable=true for discretionary impulse buys, delivery fees, café treats, or overlapping subscriptions.",
            "",
            "Messages:",
            *[f"[{idx}] {msg}" for idx, msg in enumerate(cleaned)],
        ]
    )

    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "sourceIndex": types.Schema(type=types.Type.INTEGER),
                "isTransaction": types.Schema(type=types.Type.BOOLEAN),
                "merchant": types.Schema(type=types.Type.STRING),
                "amount": types.Schema(type=types.Type.NUMBER),
                "currency": types.Schema(type=types.Type.STRING),
                "type": types.Schema(type=types.Type.STRING),
                "category": types.Schema(type=types.Type.STRING),
                "paymentMethod": types.Schema(type=types.Type.STRING),
                "date": types.Schema(type=types.Type.STRING),
                "isAvoidable": types.Schema(type=types.Type.BOOLEAN),
                "avoidableReason": types.Schema(type=types.Type.STRING),
                "confidence": types.Schema(type=types.Type.NUMBER),
            },
            required=[
                "sourceIndex",
                "isTransaction",
                "merchant",
                "amount",
                "currency",
                "type",
                "category",
                "paymentMethod",
                "date",
                "isAvoidable",
                "confidence",
            ],
        ),
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are an accurate personal finance SMS parser. Return strict JSON.",
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        ),
    )

    raw_items = json.loads((response.text or "[]").strip())
    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if not item.get("isTransaction") or float(item.get("amount") or 0) <= 0:
            continue
        src_idx = int(item.get("sourceIndex", 0))
        if src_idx < 0 or src_idx >= len(cleaned):
            src_idx = 0
        tx_type = "income" if item.get("type") == "income" else "expense"
        cat = item.get("category", "")
        if cat not in ALL_CATEGORIES:
            cat = "Salary & Income" if tx_type == "income" else "Shopping"
        curr = str(item.get("currency") or preferred_currency).upper()
        if curr not in CURRENCY_META:
            curr = preferred_currency
        tx_date = str(item.get("date") or reference_date)
        if len(tx_date) != 10 or tx_date[4] != "-" or tx_date[7] != "-":
            tx_date = reference_date

        parsed.append(
            {
                "id": f"sms_{int(datetime.utcnow().timestamp())}_{idx}_{uuid.uuid4().hex[:4]}",
                "merchant": str(item.get("merchant") or "Unknown Merchant")[:80],
                "amount": round(abs(float(item["amount"])), 2),
                "currency": curr,
                "type": tx_type,
                "category": cat,
                "paymentMethod": str(item.get("paymentMethod") or "Bank SMS Alert")[:60],
                "date": tx_date,
                "rawMessage": cleaned[src_idx],
                "sender": "BANK-SMS",
                "isAvoidable": False if tx_type == "income" else bool(item.get("isAvoidable")),
                "avoidableReason": ""
                if tx_type == "income"
                else str(item.get("avoidableReason") or "")[:140],
                "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.9)))),
                "source": "sms-ai",
            }
        )

    ignored_count = max(0, len(cleaned) - len(parsed))
    return parsed, ignored_count


def generate_ai_advice_with_gemini(
    summary: dict[str, Any], preferred_currency: str
) -> dict[str, Any]:
    client = get_gemini_client()

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "headline": types.Schema(type=types.Type.STRING),
            "monthlyDiagnosis": types.Schema(type=types.Type.STRING),
            "spikeAlerts": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "category": types.Schema(type=types.Type.STRING),
                        "increaseAmount": types.Schema(type=types.Type.NUMBER),
                        "explanation": types.Schema(type=types.Type.STRING),
                        "action": types.Schema(type=types.Type.STRING),
                    },
                    required=["category", "increaseAmount", "explanation", "action"],
                ),
            ),
            "avoidableBreakdown": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "title": types.Schema(type=types.Type.STRING),
                        "monthlyCost": types.Schema(type=types.Type.NUMBER),
                        "suggestion": types.Schema(type=types.Type.STRING),
                    },
                    required=["title", "monthlyCost", "suggestion"],
                ),
            ),
            "recommendations": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "title": types.Schema(type=types.Type.STRING),
                        "estimatedMonthlySavings": types.Schema(type=types.Type.NUMBER),
                        "priority": types.Schema(type=types.Type.STRING),
                        "steps": types.Schema(type=types.Type.STRING),
                    },
                    required=[
                        "title",
                        "estimatedMonthlySavings",
                        "priority",
                        "steps",
                    ],
                ),
            ),
        },
        required=[
            "headline",
            "monthlyDiagnosis",
            "spikeAlerts",
            "avoidableBreakdown",
            "recommendations",
        ],
    )

    prompt = (
        f"Analyze the user's current month vs previous month financial summary in {preferred_currency}.\n"
        "Return concrete, quantified money management advice grounded strictly in these numbers.\n"
        + json.dumps(summary)
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a pragmatic personal finance analyst. Return strict JSON.",
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads((response.text or "{}").strip())


def ask_ai_advisor(
    question: str, summary: dict[str, Any], preferred_currency: str
) -> str:
    client = get_gemini_client()
    prompt = (
        f"Financial context ({preferred_currency}): {json.dumps(summary)}\n\n"
        f"User question: {question}"
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                f"You are a concise, helpful personal finance advisor. All amounts are in {preferred_currency}. "
                "Answer directly using exact figures from the ledger context in 3-5 clear sentences."
            ),
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        ),
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# UI Helpers & Views (Streamlit)
# ---------------------------------------------------------------------------


def save_monthly_income(
    username: str, month_key: str, amount: float, currency: str
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            DELETE FROM transactions
            WHERE owner_username = ? AND tx_type = 'income' AND source = 'manual' AND tx_date LIKE ?
            """,
            (username, f"{month_key}%"),
        )
    scope_tag = hashlib.sha1(username.encode("utf-8")).hexdigest()[:6]
    insert_transactions(
        username,
        [
            {
                "id": f"income_{scope_tag}_{month_key}",
                "merchant": "Monthly Income",
                "amount": float(amount),
                "currency": currency,
                "type": "income",
                "category": "Salary & Income",
                "paymentMethod": "Bank Transfer",
                "date": f"{month_key}-01",
                "rawMessage": "",
                "isAvoidable": False,
                "avoidableReason": "",
                "confidence": 1.0,
                "source": "manual",
            }
        ],
    )


def render_monthly_income_section(
    username: str,
    transactions: list[dict[str, Any]],
    currency: str,
    month_key: str,
    form_key: str,
) -> None:
    month_label = get_month_label(month_key)
    existing = next(
        (
            t
            for t in transactions
            if t["date"].startswith(month_key)
            and t["type"] == "income"
            and t["source"] == "manual"
        ),
        None,
    )
    edit_flag_key = f"edit_income_{month_key}"

    if existing and not st.session_state.get(edit_flag_key, False):
        c_text, c_btn = st.columns([4, 1])
        c_text.markdown(
            f"**Monthly Income ({month_label}):** {format_currency(existing['amount'], currency)}"
        )
        if c_btn.button("Update Income", key=f"btn_{form_key}"):
            st.session_state[edit_flag_key] = True
            st.rerun()
        return

    with st.form(form_key):
        c_val, c_save = st.columns([3, 1])
        income_val = c_val.number_input(
            f"Monthly Income for {month_label} ({currency})",
            min_value=0.0,
            step=500.0,
            value=float(existing["amount"]) if existing else 0.0,
        )
        submitted = c_save.form_submit_button("Save Monthly Income", type="primary")
        if submitted:
            if income_val <= 0:
                st.error("Enter a monthly income greater than 0.")
            else:
                save_monthly_income(username, month_key, income_val, currency)
                st.session_state[edit_flag_key] = False
                st.success(f"Saved monthly income ({format_currency(income_val, currency)}).")
                st.rerun()


def render_expense_form(
    username: str,
    form_key: str,
    currency: str,
    default_date: str | None = None,
) -> None:
    with st.form(form_key, clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        merchant = c1.text_input(
            "Shop / Merchant",
            placeholder="e.g., BigBasket, Starbucks",
        )
        amount = c2.number_input(f"Amount ({currency})", min_value=0.0, step=50.0)
        tx_date = c3.date_input(
            "Date",
            value=datetime.strptime(default_date or DEFAULT_REFERENCE_DATE, "%Y-%m-%d").date(),
        )

        c4, c5 = st.columns(2)
        category = c4.selectbox("Category", options=EXPENSE_CATEGORIES)
        method = c5.text_input("Payment Method", value="UPI / Card")

        is_avoidable = st.checkbox("Optional / avoidable spending (could skip)")
        avoid_reason = ""
        if is_avoidable:
            avoid_reason = st.text_input(
                "Why is it optional?",
                placeholder="e.g., Delivery fee or impulse order",
            )

        submitted = st.form_submit_button("Save Expense", type="primary")
        if submitted:
            if not merchant.strip() or amount <= 0:
                st.error("Please enter a merchant name and an amount greater than 0.")
            else:
                insert_transactions(
                    username,
                    [
                        {
                            "id": f"manual_{uuid.uuid4().hex[:10]}",
                            "merchant": merchant.strip(),
                            "amount": float(amount),
                            "currency": currency,
                            "type": "expense",
                            "category": category,
                            "paymentMethod": method.strip() or "UPI / Card",
                            "date": tx_date.isoformat(),
                            "rawMessage": "",
                            "isAvoidable": is_avoidable,
                            "avoidableReason": avoid_reason.strip(),
                            "confidence": 1.0,
                            "source": "manual",
                        }
                    ],
                )
                st.success(f"Saved {merchant.strip()} ({format_currency(amount, currency)}).")
                st.rerun()


def render_home(
    username: str,
    transactions: list[dict[str, Any]],
    metrics: dict[str, Any],
    currency: str,
    current_month_key: str,
) -> None:
    has_data = len(transactions) > 0
    if not has_data:
        st.subheader("Start in 2 Steps")
        st.write("Set your monthly income once, then log expenses or scan SMS alerts.")
        st.markdown("#### 1. Set Monthly Income")
        render_monthly_income_section(username, transactions, currency, current_month_key, "welcome_income")
        st.markdown("#### 2. Log Expense")
        render_expense_form(username, "welcome_expense", currency)
        return

    render_monthly_income_section(username, transactions, currency, current_month_key, "home_income")
    with st.expander("+ Add Expense"):
        render_expense_form(username, "quick_expense", currency)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"Money Saved ({metrics['currentMonthLabel']})",
        format_currency(metrics["netSavings"], currency),
        f"{metrics['savingsRate']:.1f}% saved",
    )
    k2.metric(
        f"Money Spent ({metrics['currentMonthLabel']})",
        format_currency(metrics["currentExpenses"], currency),
        f"{metrics['expenseDeltaAmount']:+,.2f} vs {metrics['previousMonthLabel']}",
        delta_color="inverse",
    )
    k3.metric(
        "Money Received (Income)",
        format_currency(metrics["currentIncome"], currency),
        f"Est. month-end: {format_currency(metrics['projectedMonthEndSpend'], currency)}",
    )
    k4.metric(
        "Optional Spending (Can Cut)",
        format_currency(metrics["recoverableAvoidableTotal"], currency),
        f"{len(metrics['avoidableTransactions'])} item(s)",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Spending Spikes")
        spikes = metrics["spikeAlerts"]
        if not spikes:
            st.info(f"No major spending spikes vs {metrics['previousMonthLabel']}.")
        else:
            for s in spikes[:5]:
                pct = f" (+{s['deltaPercent']:.0f}%)" if s["deltaPercent"] is not None else " (New)"
                st.warning(
                    f"**{s['category']}** +{format_currency(s['deltaAmount'], currency)}{pct} — "
                    f"{format_currency(s['currentAmount'], currency)} vs {format_currency(s['previousAmount'], currency)}"
                )

    with right:
        st.subheader("Optional Expenses")
        avoidable = metrics["avoidableTransactions"]
        if not avoidable:
            st.info("All recorded expenses this month look essential.")
        else:
            for tx in avoidable[:5]:
                st.write(
                    f"- **{tx['merchant']}** ({tx['category']}, {tx['date']}): "
                    f"**{format_currency(tx['amount'], currency)}**"
                    + (f" — *{tx['avoidableReason']}*" if tx["avoidableReason"] else "")
                )

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader(f"Spending by Category ({metrics['currentMonthLabel']} vs {metrics['previousMonthLabel']})")
        if metrics["categoryComparisons"]:
            cat_df = pd.DataFrame(
                [
                    {
                        "Category": c["category"],
                        metrics["currentMonthLabel"]: c["currentAmount"],
                        metrics["previousMonthLabel"]: c["previousAmount"],
                    }
                    for c in metrics["categoryComparisons"]
                ]
            ).set_index("Category")
            st.bar_chart(cat_df)
        else:
            st.caption("No category spending recorded yet.")

    with c_right:
        st.subheader("6-Month Spending Trend")
        trends = calculate_six_month_trends(transactions, current_month_key)
        trend_df = pd.DataFrame(
            [
                {
                    "Month": t["monthLabel"],
                    "Essential": t["Essential"],
                    "Avoidable": t["Avoidable"],
                }
                for t in trends
            ]
        ).set_index("Month")
        st.bar_chart(trend_df)

    st.subheader("Recent Entries")
    recent_df = pd.DataFrame(
        [
            {
                "Date": t["date"],
                "Type": t["type"].title(),
                "Merchant": t["merchant"],
                "Category": t["category"],
                "Method": t["paymentMethod"],
                "Amount": format_currency(t["amount"], currency),
                "Optional": "Yes" if t["isAvoidable"] else "No",
            }
            for t in transactions[:10]
        ]
    )
    st.dataframe(recent_df, use_container_width=True, hide_index=True)


def render_reports(
    transactions: list[dict[str, Any]],
    currency: str,
    savings_goal: float,
    current_month_key: str,
) -> None:
    st.subheader("Financial Reports")
    tab_monthly, tab_annual = st.tabs(["Monthly Statement", "All Months (Jan–Dec)"])

    with tab_monthly:
        month_keys = sorted(
            {
                current_month_key,
                get_previous_month_key(current_month_key),
                *[t["date"][:7] for t in transactions if len(t["date"]) >= 7],
            },
            reverse=True,
        )
        selected_month = st.selectbox(
            "Select Month",
            options=month_keys,
            format_func=get_month_label,
        )
        m_metrics = calculate_monthly_metrics(transactions, selected_month, currency)
        essential = max(0.0, m_metrics["currentExpenses"] - m_metrics["recoverableAvoidableTotal"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Income", format_currency(m_metrics["currentIncome"], currency))
        c2.metric("Total Expenses", format_currency(m_metrics["currentExpenses"], currency))
        c3.metric(
            "Net Savings",
            format_currency(m_metrics["netSavings"], currency),
            f"{m_metrics['savingsRate']:.1f}% savings rate",
        )
        goal_pct = (
            max(0.0, min(100.0, (m_metrics["netSavings"] / savings_goal) * 100.0))
            if savings_goal > 0
            else 0.0
        )
        c4.metric("Savings Goal Progress", f"{goal_pct:.0f}%", f"Goal: {format_currency(savings_goal, currency)}")

        if savings_goal > 0:
            st.progress(goal_pct / 100.0, text=f"Savings goal: {goal_pct:.0f}%")

        st.markdown(
            f"**Cashflow Allocation ({m_metrics['currentMonthLabel']}):** "
            f"Essential **{format_currency(essential, currency)}** · "
            f"Avoidable **{format_currency(m_metrics['recoverableAvoidableTotal'], currency)}** · "
            f"Net Savings **{format_currency(m_metrics['netSavings'], currency)}**"
        )

        col_inc, col_cat = st.columns(2)
        month_tx = [t for t in transactions if t["date"].startswith(selected_month)]
        with col_inc:
            st.markdown("#### Monthly Income Entries")
            inc_rows = [t for t in month_tx if t["type"] == "income"]
            if inc_rows:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Date": t["date"],
                                "Source": t["merchant"],
                                "Method": t["paymentMethod"],
                                "Amount": format_currency(t["amount"], currency),
                            }
                            for t in inc_rows
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No income logged for {m_metrics['currentMonthLabel']}.")

        with col_cat:
            st.markdown("#### Expense Breakdown by Category")
            cat_rows = [c for c in m_metrics["categoryComparisons"] if c["currentAmount"] > 0]
            if cat_rows:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Category": c["category"],
                                "Current": format_currency(c["currentAmount"], currency),
                                "Last Month": format_currency(c["previousAmount"], currency),
                                "Avoidable": format_currency(c["avoidableAmount"], currency),
                                "Txns": c["transactionCount"],
                            }
                            for c in cat_rows
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No expenses recorded for {m_metrics['currentMonthLabel']}.")

        statement_df = pd.DataFrame(
            [
                {
                    "Date": t["date"],
                    "Type": t["type"],
                    "Merchant": t["merchant"],
                    "Category": t["category"],
                    "PaymentMethod": t["paymentMethod"],
                    "Amount": t["amount"],
                    "Avoidable": "Yes" if t["isAvoidable"] else "No",
                }
                for t in month_tx
            ]
        )
        st.download_button(
            "Download Monthly Statement CSV",
            data=statement_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ledgely-report-{selected_month}.csv",
            mime="text/csv",
        )

    with tab_annual:
        default_year = int(current_month_key[:4])
        years = sorted(
            {
                default_year - 1,
                default_year,
                default_year + 1,
                *[int(t["date"][:4]) for t in transactions if t["date"][:4].isdigit()],
            },
            reverse=True,
        )
        selected_year = st.selectbox("Calendar Year", options=years)
        months = calculate_year_breakdown(transactions, selected_year)

        total_inc = sum(m["income"] for m in months)
        total_exp = sum(m["expenses"] for m in months)
        total_avoid = sum(m["avoidableExpenses"] for m in months)
        total_net = total_inc - total_exp
        ann_rate = (total_net / total_inc) * 100.0 if total_inc > 0 else 0.0

        a1, a2, a3, a4 = st.columns(4)
        a1.metric(f"Income ({selected_year})", format_currency(total_inc, currency))
        a2.metric(f"Expenses ({selected_year})", format_currency(total_exp, currency))
        a3.metric(
            "Annual Net Savings",
            format_currency(total_net, currency),
            f"{ann_rate:.1f}% savings rate",
        )
        a4.metric("Annual Avoidable Spend", format_currency(total_avoid, currency))

        chart_df = pd.DataFrame(
            [
                {
                    "Month": m["shortLabel"],
                    "Income": m["income"],
                    "Expenses": m["expenses"],
                }
                for m in months
            ]
        ).set_index("Month")
        st.bar_chart(chart_df)

        table_df = pd.DataFrame(
            [
                {
                    "Month": m["monthLabel"],
                    "Income": round(m["income"], 2),
                    "Expenses": round(m["expenses"], 2),
                    "Avoidable": round(m["avoidableExpenses"], 2),
                    "Net Savings": round(m["netSavings"], 2),
                    "Savings Rate %": round(m["savingsRate"], 1),
                    "Top Category": m["topCategory"],
                    "Txns": m["transactionCount"],
                }
                for m in months
            ]
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export Jan–Dec CSV",
            data=table_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ledgely-annual-{selected_year}.csv",
            mime="text/csv",
        )


def render_transactions(
    username: str, transactions: list[dict[str, Any]], currency: str
) -> None:
    c1, c2, c3 = st.columns(3)
    query = c1.text_input("Search shop, category, or SMS text")
    type_filter = c2.selectbox("Type", ["all", "expense", "income"])
    cat_filter = c3.selectbox("Category", ["All", *ALL_CATEGORIES])

    filtered = transactions
    if query.strip():
        q = query.strip().lower()
        filtered = [
            t
            for t in filtered
            if q in t["merchant"].lower()
            or q in t["category"].lower()
            or q in t["rawMessage"].lower()
        ]
    if type_filter != "all":
        filtered = [t for t in filtered if t["type"] == type_filter]
    if cat_filter != "All":
        filtered = [t for t in filtered if t["category"] == cat_filter]

    if filtered:
        f_inc = sum(t["amount"] for t in filtered if t["type"] == "income")
        f_exp = sum(t["amount"] for t in filtered if t["type"] == "expense")
        f_avoid = sum(
            t["amount"] for t in filtered if t["type"] == "expense" and t["isAvoidable"]
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Filtered Income", format_currency(f_inc, currency))
        m2.metric("Filtered Spend", format_currency(f_exp, currency))
        m3.metric("Filtered Optional Spend", format_currency(f_avoid, currency))

    bar_left, bar_mid, bar_right = st.columns([2, 1, 1])
    bar_left.caption(f"Showing {len(filtered)} of {len(transactions)} transactions")
    if bar_mid.button("Load Sample Data", use_container_width=True):
        demo_rows = build_demo_transactions(DEFAULT_REFERENCE_DATE, currency, username)
        insert_transactions(username, demo_rows)
        st.success("Loaded 6 months of sample data (income, expenses, spikes, and SMS).")
        st.rerun()

    if filtered:
        export_df = pd.DataFrame(
            [
                {
                    "Date": t["date"],
                    "Type": t["type"],
                    "Merchant": t["merchant"],
                    "Category": t["category"],
                    "Amount": t["amount"],
                    "Currency": t["currency"],
                    "PaymentMethod": t["paymentMethod"],
                    "Avoidable": "Yes" if t["isAvoidable"] else "No",
                    "Reason": t["avoidableReason"],
                }
                for t in filtered
            ]
        )
        bar_right.download_button(
            "Download Filtered CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ledgely-transactions-{DEFAULT_REFERENCE_DATE}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        for tx in filtered[:60]:
            with st.container(border=True):
                r1, r2, r3 = st.columns([3, 2, 1])
                sign = "+" if tx["type"] == "income" else "-"
                r1.markdown(
                    f"**{tx['merchant']}** · {tx['date']} · {tx['paymentMethod']}"
                )
                if tx["rawMessage"]:
                    r1.caption(tx["rawMessage"])
                r2.markdown(f"**{sign}{format_currency(tx['amount'], currency)}** · {tx['category']}")
                if tx["isAvoidable"]:
                    r2.caption(f"Optional spend: {tx['avoidableReason'] or 'Marked avoidable'}")

                if r3.button("Delete", key=f"del_{tx['id']}"):
                    delete_transaction(username, tx["id"])
                    st.rerun()

                if tx["type"] == "expense":
                    with st.expander("Edit Category / Optional Tag"):
                        new_cat = st.selectbox(
                            "Category",
                            options=EXPENSE_CATEGORIES,
                            index=EXPENSE_CATEGORIES.index(tx["category"])
                            if tx["category"] in EXPENSE_CATEGORIES
                            else 0,
                            key=f"cat_{tx['id']}",
                        )
                        new_avoid = st.checkbox(
                            "Optional / avoidable spending",
                            value=tx["isAvoidable"],
                            key=f"av_{tx['id']}",
                        )
                        new_reason = st.text_input(
                            "Reason",
                            value=tx["avoidableReason"],
                            key=f"rs_{tx['id']}",
                        )
                        if st.button("Update", key=f"upd_{tx['id']}"):
                            update_transaction_fields(
                                username, tx["id"], new_cat, new_avoid, new_reason
                            )
                            st.rerun()
    else:
        st.info("No transactions match. Add one above or click Load Sample Data.")


def render_sms_sync(
    username: str, transactions: list[dict[str, Any]], currency: str
) -> None:
    st.subheader("Scan Bank & UPI SMS (Gemini AI)")
    st.write("Paste SMS alerts (one per line) or upload a .txt, .csv, or .json file.")

    if "sms_draft" not in st.session_state:
        st.session_state.sms_draft = ""

    cols = st.columns(len(SAMPLE_SMS_EXAMPLES))
    for col, (label, sample_text) in zip(cols, SAMPLE_SMS_EXAMPLES):
        if col.button(f"+ {label}", use_container_width=True):
            st.session_state.sms_draft = sample_text

    sms_text = st.text_area(
        "SMS messages (one per line)",
        value=st.session_state.sms_draft,
        height=140,
        placeholder="Paste bank or UPI SMS messages here...",
    )
    st.session_state.sms_draft = sms_text

    uploaded = st.file_uploader("Or upload SMS export (.txt, .csv, .json)", type=["txt", "csv", "json"])
    if uploaded is not None:
        raw_content = uploaded.read().decode("utf-8", errors="replace")
        if uploaded.name.lower().endswith(".json"):
            try:
                rows = json.loads(raw_content)
                valid = [
                    r
                    for r in rows
                    if isinstance(r, dict)
                    and isinstance(r.get("merchant"), str)
                    and float(r.get("amount") or 0) > 0
                ]
                if st.button(f"Import {len(valid)} JSON Transaction(s)", type="primary"):
                    count = insert_transactions(username, valid)
                    st.success(f"Imported {count} transaction(s) from JSON.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not parse JSON file: {exc}")
        else:
            st.session_state.sms_draft = raw_content

    if st.button("Read & Save Payments with Gemini AI", type="primary"):
        lines = [line.strip() for line in sms_text.splitlines() if line.strip()]
        if not lines:
            st.warning("Paste at least one payment SMS message first.")
        else:
            with st.spinner("Reading SMS alerts with Gemini AI..."):
                try:
                    parsed, ignored = parse_sms_with_gemini(
                        lines, currency, DEFAULT_REFERENCE_DATE
                    )
                    if parsed:
                        insert_transactions(username, parsed)
                        st.session_state.sms_draft = ""
                        st.success(
                            f"Added {len(parsed)} payment(s) ({ignored} non-payment/OTP message ignored)."
                        )
                        st.rerun()
                    else:
                        st.info("No payment transactions detected in those messages.")
                except Exception as exc:
                    st.error(f"AI SMS parsing failed: {exc}")

    st.divider()
    sms_logs = [t for t in transactions if t["source"] != "manual" and t["rawMessage"]]
    st.subheader(f"Ingested Phone Message Log ({len(sms_logs)})")
    if not sms_logs:
        st.caption("No SMS messages parsed yet.")
    else:
        for tx in sms_logs[:25]:
            sign = "+" if tx["type"] == "income" else "-"
            with st.container(border=True):
                st.markdown(
                    f"**{tx['merchant']}** · {tx['category']} · {tx['date']} · "
                    f"**{sign}{format_currency(tx['amount'], currency)}**"
                )
                st.caption(f"SMS: “{tx['rawMessage']}”")


def render_advisor(
    username: str,
    transactions: list[dict[str, Any]],
    metrics: dict[str, Any],
    currency: str,
    current_month_key: str,
) -> None:
    saved_report = load_saved_ai_report(username, current_month_key)

    summary_payload = {
        "month": metrics["currentMonthLabel"],
        "previousMonth": metrics["previousMonthLabel"],
        "income": metrics["currentIncome"],
        "expenses": metrics["currentExpenses"],
        "previousExpenses": metrics["previousExpenses"],
        "netSavings": metrics["netSavings"],
        "savingsRatePct": round(metrics["savingsRate"], 1),
        "recoverableAvoidableTotal": metrics["recoverableAvoidableTotal"],
        "topCategories": metrics["categoryComparisons"][:6],
        "spikeAlerts": metrics["spikeAlerts"][:5],
        "topAvoidableItems": [
            {
                "merchant": t["merchant"],
                "amount": t["amount"],
                "category": t["category"],
                "reason": t["avoidableReason"],
            }
            for t in metrics["avoidableTransactions"][:6]
        ],
    }

    tab_plan, tab_digest, tab_ask = st.tabs(
        ["AI Savings Plan", "Weekly Optional Digest", "Ask Advisor"]
    )

    with tab_plan:
        if st.button("Generate / Refresh AI Savings Plan", type="primary"):
            with st.spinner("Analyzing spending patterns with Gemini AI..."):
                try:
                    report = generate_ai_advice_with_gemini(summary_payload, currency)
                    save_ai_report(username, current_month_key, currency, report)
                    saved_report = report
                    st.success("AI financial plan updated.")
                except Exception as exc:
                    st.error(f"Could not generate AI plan: {exc}")

        if saved_report:
            st.markdown(f"### {saved_report.get('headline', '')}")
            st.write(saved_report.get("monthlyDiagnosis", ""))

            spikes = saved_report.get("spikeAlerts", [])
            avoid_items = saved_report.get("avoidableBreakdown", [])
            if spikes or avoid_items:
                scol, acol = st.columns(2)
                with scol:
                    st.markdown("#### Spending Spike Insights")
                    for sp in spikes:
                        with st.container(border=True):
                            st.markdown(
                                f"**{sp.get('category', '')}** · +{format_currency(float(sp.get('increaseAmount', 0)), currency)}"
                            )
                            st.write(sp.get("explanation", ""))
                            st.caption(f"Action: {sp.get('action', '')}")
                with acol:
                    st.markdown("#### Avoidable Cost Breakdown")
                    for av in avoid_items:
                        with st.container(border=True):
                            st.markdown(
                                f"**{av.get('title', '')}** · {format_currency(float(av.get('monthlyCost', 0)), currency)}/mo"
                            )
                            st.write(av.get("suggestion", ""))

            recs = saved_report.get("recommendations", [])
            if recs:
                st.markdown("#### Recommended Money Moves")
                for rec in recs:
                    with st.container(border=True):
                        st.markdown(
                            f"**{rec.get('title', '')}** · Priority: {rec.get('priority', 'Medium')} · "
                            f"Est. Savings: **{format_currency(float(rec.get('estimatedMonthlySavings', 0)), currency)}/mo**"
                        )
                        st.write(rec.get("steps", ""))
        else:
            st.info("Generate a plan to see personalized recommendations.")

    with tab_digest:
        st.subheader("Weekly Optional Spend Digest")
        buckets = calculate_weekly_avoidable_digest(transactions, current_month_key)
        if not buckets:
            st.caption("No weekly expenses recorded for this month yet.")
        else:
            for b in buckets:
                with st.expander(
                    f"{b['weekLabel']} ({b['rangeLabel']}) — Optional: {format_currency(b['avoidableTotal'], currency)} "
                    f"({b['avoidableSharePct']:.0f}% of weekly spend)"
                ):
                    if not b["items"]:
                        st.write("No avoidable expenses flagged in this week.")
                    else:
                        for item in b["items"]:
                            st.write(
                                f"- **{item['merchant']}** ({item['category']}): "
                                f"{format_currency(item['amount'], currency)} — {item['avoidableReason']}"
                            )

    with tab_ask:
        st.subheader("Ask Your Financial Advisor")
        starters = [
            "How can I save more money this month?",
            "Which expense category jumped the most?",
            "What optional purchases should I cut first?",
        ]
        if "advisor_q" not in st.session_state:
            st.session_state.advisor_q = ""

        scols = st.columns(len(starters))
        for col, q_text in zip(scols, starters):
            if col.button(q_text, use_container_width=True):
                st.session_state.advisor_q = q_text

        question = st.text_input("Ask a question about your finances", value=st.session_state.advisor_q)
        st.session_state.advisor_q = question
        if st.button("Ask Advisor", type="primary") and question.strip():
            with st.spinner("Thinking..."):
                try:
                    answer = ask_ai_advisor(question.strip(), summary_payload, currency)
                    st.success(answer)
                except Exception as exc:
                    st.error(f"Advisor request failed: {exc}")


def _has_google_oidc_config() -> bool:
    try:
        return hasattr(st, "login") and "auth" in st.secrets
    except Exception:
        return False


def _get_google_oidc_user() -> dict[str, str] | None:
    try:
        u = getattr(st, "user", None)
        if u and getattr(u, "is_logged_in", False):
            email = str(getattr(u, "email", "") or getattr(u, "sub", "") or "google-user").strip().lower()
            name = str(getattr(u, "name", "") or email).strip()
            return {"username": email, "display_name": name, "provider": "google"}
    except Exception:
        return None
    return None


def render_auth_gate() -> None:
    st.title("Ledgely")
    st.subheader("Sign in with Google")
    st.write("One-click access to your private expense tracker.")

    if _has_google_oidc_config():
        if st.button("Continue with Google", type="primary", use_container_width=True):
            st.login()
    else:
        st.info(
            "To enable Google Sign-In locally in Streamlit 1.42+, add your Google OAuth credentials "
            "under [auth] in .streamlit/secrets.toml, or use local sign-in below."
        )

    with st.expander("Or sign in locally without Google"):
        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                l_user = st.text_input("Username")
                l_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", type="primary"):
                    account = authenticate_user(l_user, l_pass)
                    if account:
                        st.session_state.auth_user = account
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_register:
            with st.form("register_form"):
                r_name = st.text_input("Display Name", placeholder="e.g., Alex")
                r_user = st.text_input("Username", placeholder="letters, numbers, _ . -")
                r_pass = st.text_input("Password (min 6 characters)", type="password")
                r_confirm = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Create Account", type="primary"):
                    if r_pass != r_confirm:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = create_user(r_user, r_name, r_pass)
                        if ok:
                            st.session_state.auth_user = {
                                "username": r_user.strip().lower(),
                                "display_name": r_name.strip() or r_user.strip().lower(),
                            }
                            st.rerun()
                        else:
                            st.error(msg)


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Ledgely — Python Expense & Savings Tracker",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_db()

    auth_user = _get_google_oidc_user() or st.session_state.get("auth_user")
    if not auth_user:
        render_auth_gate()
        return

    username = auth_user["username"]
    display_name = auth_user.get("display_name") or username
    is_google_user = auth_user.get("provider") == "google"

    currency = get_setting(username, "currency", "INR")
    if currency not in CURRENCY_META:
        currency = "INR"
    savings_goal = float(get_setting(username, "savings_goal", "35000") or 35000)

    transactions = fetch_transactions(username)
    current_month_key = DEFAULT_REFERENCE_DATE[:7]
    ref_day = int(DEFAULT_REFERENCE_DATE[8:10])
    metrics = calculate_monthly_metrics(transactions, current_month_key, currency, ref_day)

    with st.sidebar:
        st.title("Ledgely")
        nav = st.radio(
            "Navigation",
            options=[
                "Home",
                "Reports",
                "All Transactions",
                "SMS & Phone Sync",
                "Tips & Savings Plan",
            ],
        )

        st.divider()
        st.caption("Settings & Goal")
        new_currency = st.selectbox(
            "Currency",
            options=list(CURRENCY_META.keys()),
            index=list(CURRENCY_META.keys()).index(currency),
            format_func=lambda c: CURRENCY_META[c]["label"],
        )
        if new_currency != currency:
            set_setting(username, "currency", new_currency)
            currency = new_currency
            st.rerun()

        new_goal = st.number_input(
            f"Monthly Savings Goal ({currency})",
            min_value=0.0,
            step=500.0,
            value=savings_goal,
        )
        if abs(new_goal - savings_goal) > 0.01:
            set_setting(username, "savings_goal", str(round(new_goal)))
            savings_goal = new_goal

        pct = max(0.0, min(100.0, (metrics["netSavings"] / savings_goal) * 100.0)) if savings_goal > 0 else 0.0
        st.progress(pct / 100.0, text=f"{pct:.0f}% of goal saved · {len(transactions)} entries")

        st.divider()
        st.caption("Google Account" if is_google_user else "Account")
        st.write(f"Signed in as **{display_name}** ({username})")
        if st.button("Sign Out", use_container_width=True):
            if is_google_user and hasattr(st, "logout"):
                st.logout()
            else:
                st.session_state.pop("auth_user", None)
                st.rerun()

        st.divider()
        with st.expander("Clear All Data"):
            st.warning("This removes all saved transactions and cached AI reports for your account.")
            if st.button("Yes, Delete All Data", type="primary"):
                clear_all_transactions(username)
                st.rerun()

    st.header(nav)
    if nav == "Home":
        render_home(username, transactions, metrics, currency, current_month_key)
    elif nav == "Reports":
        render_reports(transactions, currency, savings_goal, current_month_key)
    elif nav == "All Transactions":
        render_transactions(username, transactions, currency)
    elif nav == "SMS & Phone Sync":
        render_sms_sync(username, transactions, currency)
    elif nav == "Tips & Savings Plan":
        render_advisor(username, transactions, metrics, currency, current_month_key)


if __name__ == "__main__":
    main()
