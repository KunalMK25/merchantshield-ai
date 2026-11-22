"""
Case management: the analyst alert queue, simulated account holds, customer notices and
the audit trail.

THE RULE THIS MODULE EXISTS TO ENFORCE
The system only recommends. Monitoring creates alerts; it never touches an account.
An account changes state only through an explicit action by a named person (hold,
release, close), each in one database transaction together with its notice and audit
events, so a failure can never leave a half-applied decision.

Account lifecycle:  active --hold--> held --release--> active
                                     held --close---> closed (final)
Closing needs a hold first, so it is always a second, deliberate step.

Everything here is a simulation: notices are written but never sent, and no real account
exists behind a customer id.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ALERT_SEVERITIES = ("critical", "high")     # watch-level flags stay on the drift page, not in the queue
ANALYST_MIN, ANALYST_MAX = 2, 80
NOTE_MAX = 500
NOTE_MIN_WHEN_REQUIRED = 3
DELIVERY = "Simulated: not sent to any customer"

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL, day INTEGER NOT NULL, severity TEXT NOT NULL,
    points INTEGER, iso_score REAL, signals TEXT, n_txns INTEGER, total_amount REAL,
    recommended_action TEXT, job_id TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    decision TEXT, decided_by TEXT, decided_at TEXT, decision_note TEXT,
    UNIQUE (customer_id, day)
);
CREATE TABLE IF NOT EXISTS accounts (
    customer_id TEXT PRIMARY KEY, status TEXT NOT NULL,
    updated_at TEXT NOT NULL, updated_by TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS notices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL, alert_id INTEGER, kind TEXT NOT NULL,
    message TEXT NOT NULL, delivery TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL, actor TEXT NOT NULL, kind TEXT NOT NULL,
    customer_id TEXT, alert_id INTEGER, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status, severity, day);
CREATE INDEX IF NOT EXISTS idx_events_customer ON events (customer_id, id);
"""


class CaseError(RuntimeError):
    pass


class NotFound(CaseError):
    pass


class Conflict(CaseError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _actor(name) -> str:
    name = (name or "").strip() if isinstance(name, str) else ""
    if not (ANALYST_MIN <= len(name) <= ANALYST_MAX):
        raise ValueError(f"Analyst name must be {ANALYST_MIN}-{ANALYST_MAX} characters.")
    return name


def _note(note, required: bool) -> str:
    note = (note or "").strip() if isinstance(note, (str, type(None))) else ""
    if len(note) > NOTE_MAX:
        raise ValueError(f"Note must be at most {NOTE_MAX} characters.")
    if required and len(note) < NOTE_MIN_WHEN_REQUIRED:
        raise ValueError("A reason is required for this action.")
    return note


def notice_text(kind: str, customer_id: str) -> str:
    """
    What the customer would be told. Deliberately says nothing about how the activity
    was detected: describing detection logic to a possible fraudster helps them evade it.
    """
    prefix = f"Account {customer_id}: "
    return prefix + {
        "hold": ("We noticed unusual activity on your account and have temporarily blocked it to keep "
                 "your money safe. Our team is reviewing it and will call you shortly. You do not need "
                 "to do anything right now. If you need help in the meantime, please contact the "
                 "bank's helpline."),
        "release": "Our review is complete and your account is active again. Thank you for your patience.",
        "close": ("Following our review, your account has been closed to protect you. Please contact the "
                  "bank to open a new account and to discuss any transactions you do not recognise."),
    }[kind]


class CaseStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._read() as conn:
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")   # the API reads while the daily job writes

    # ------------------------------------------------------------ plumbing
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _read(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @staticmethod
    def _event(conn, actor, kind, customer_id=None, alert_id=None, **detail):
        conn.execute("INSERT INTO events (at, actor, kind, customer_id, alert_id, detail) VALUES (?,?,?,?,?,?)",
                     (_now(), actor, kind, customer_id, alert_id, json.dumps(detail)))

    def _insert_notice(self, conn, customer_id, alert_id, kind) -> dict:
        created = _now()
        message = notice_text(kind, customer_id)
        cur = conn.execute(
            "INSERT INTO notices (customer_id, alert_id, kind, message, delivery, created_at) VALUES (?,?,?,?,?,?)",
            (customer_id, alert_id, kind, message, DELIVERY, created))
        return {"id": cur.lastrowid, "customer_id": customer_id, "alert_id": alert_id, "kind": kind,
                "message": message, "delivery": DELIVERY, "created_at": created}

    @staticmethod
    def _account_row(conn, customer_id) -> dict:
        row = conn.execute("SELECT * FROM accounts WHERE customer_id = ?", (customer_id,)).fetchone()
        if row is None:
            return {"customer_id": customer_id, "status": "active", "updated_at": None,
                    "updated_by": None, "reason": None}
        return dict(row)

    @staticmethod
    def _alert_dict(row) -> dict:
        d = dict(row)
        d["signals"] = json.loads(d["signals"]) if d.get("signals") else []
        d["account_status"] = d.pop("account_status", None) or "active"
        return d

    def _alert(self, conn, alert_id) -> dict:
        row = conn.execute(
            "SELECT al.*, COALESCE(ac.status, 'active') AS account_status FROM alerts al "
            "LEFT JOIN accounts ac ON ac.customer_id = al.customer_id WHERE al.id = ?", (alert_id,)).fetchone()
        if row is None:
            raise NotFound(f"No alert {alert_id}")
        return self._alert_dict(row)

    # -------------------------------------------------------------- ingest
    def ingest(self, records, job_id=None) -> int:
        """Create alerts from anomaly flags. Idempotent; never touches accounts."""
        created = 0
        with self._tx() as conn:
            for r in records:
                if r.get("severity") not in ALERT_SEVERITIES:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO alerts (customer_id, day, severity, points, iso_score, signals, "
                    "n_txns, total_amount, recommended_action, job_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (r["customer_id"], int(r["day"]), r["severity"], r.get("points"), r.get("iso_score"),
                     json.dumps(r.get("signals") or []), r.get("n_txns"), r.get("total_amount"),
                     r.get("recommended_action"), job_id, _now()))
                if cur.rowcount:
                    created += 1
                    alert_id = cur.lastrowid
                    self._event(conn, "system", "alert_created", r["customer_id"], alert_id,
                                severity=r["severity"], day=int(r["day"]))
        return created

    # ------------------------------------------------------------- queries
    def list_alerts(self, status=None, severities=None, limit=100) -> list:
        where, args = [], []
        if status:
            where.append("al.status = ?")
            args.append(status)
        if severities:
            where.append(f"al.severity IN ({','.join('?' * len(severities))})")
            args.extend(severities)
        sql = ("SELECT al.*, COALESCE(ac.status, 'active') AS account_status FROM alerts al "
               "LEFT JOIN accounts ac ON ac.customer_id = al.customer_id "
               + (f"WHERE {' AND '.join(where)} " if where else "")
               + "ORDER BY CASE al.severity WHEN 'critical' THEN 0 ELSE 1 END, al.day DESC, al.id DESC LIMIT ?")
        with self._read() as conn:
            return [self._alert_dict(r) for r in conn.execute(sql, (*args, max(1, min(int(limit), 500)))).fetchall()]

    def get_alert(self, alert_id) -> dict:
        with self._read() as conn:
            alert = self._alert(conn, alert_id)
            cust = alert["customer_id"]
            alert["notices"] = [dict(r) for r in conn.execute(
                "SELECT * FROM notices WHERE customer_id = ? ORDER BY id", (cust,)).fetchall()]
            alert["events"] = [self._event_dict(r) for r in conn.execute(
                "SELECT * FROM events WHERE customer_id = ? ORDER BY id DESC", (cust,)).fetchall()]
            return alert

    def account(self, customer_id) -> dict:
        with self._read() as conn:
            return self._account_row(conn, customer_id)

    def list_accounts(self, status=None) -> list:
        with self._read() as conn:
            if status:
                rows = conn.execute("SELECT * FROM accounts WHERE status = ? ORDER BY updated_at DESC, customer_id",
                                    (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM accounts ORDER BY updated_at DESC, customer_id").fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def _event_dict(row) -> dict:
        d = dict(row)
        d["detail"] = json.loads(d["detail"]) if d.get("detail") else {}
        return d

    def events(self, limit=50, customer_id=None) -> list:
        sql, args = "SELECT * FROM events ", []
        if customer_id:
            sql += "WHERE customer_id = ? "
            args.append(customer_id)
        sql += "ORDER BY id DESC LIMIT ?"
        with self._read() as conn:
            return [self._event_dict(r) for r in conn.execute(sql, (*args, max(1, min(int(limit), 500)))).fetchall()]

    def summary(self) -> dict:
        with self._read() as conn:
            one = lambda sql: conn.execute(sql).fetchone()[0]
            return {
                "open_alerts": one("SELECT COUNT(*) FROM alerts WHERE status='open'"),
                "open_critical": one("SELECT COUNT(*) FROM alerts WHERE status='open' AND severity='critical'"),
                "open_high": one("SELECT COUNT(*) FROM alerts WHERE status='open' AND severity='high'"),
                "dismissed_alerts": one("SELECT COUNT(*) FROM alerts WHERE status='dismissed'"),
                "actioned_alerts": one("SELECT COUNT(*) FROM alerts WHERE status='actioned'"),
                "held_accounts": one("SELECT COUNT(*) FROM accounts WHERE status='held'"),
                "closed_accounts": one("SELECT COUNT(*) FROM accounts WHERE status='closed'"),
            }

    # ------------------------------------------------------------- actions
    def hold(self, alert_id, decided_by, note="") -> dict:
        actor, note = _actor(decided_by), _note(note, required=False)
        with self._tx() as conn:
            alert = self._alert(conn, alert_id)
            if alert["status"] != "open":
                raise Conflict(f"Alert {alert_id} was already {alert['status']}.")
            cust = alert["customer_id"]
            acct = self._account_row(conn, cust)
            if acct["status"] == "closed":
                raise Conflict(f"Account {cust} is closed.")

            conn.execute("UPDATE alerts SET status='actioned', decision='hold', decided_by=?, decided_at=?, "
                         "decision_note=? WHERE id=?", (actor, _now(), note, alert_id))
            notice = None
            if acct["status"] == "held":
                self._event(conn, actor, "hold_confirmed", cust, alert_id, note=note)
            else:
                conn.execute("INSERT INTO accounts (customer_id, status, updated_at, updated_by, reason) "
                             "VALUES (?, 'held', ?, ?, ?) ON CONFLICT(customer_id) DO UPDATE SET "
                             "status='held', updated_at=excluded.updated_at, updated_by=excluded.updated_by, "
                             "reason=excluded.reason", (cust, _now(), actor, note))
                self._event(conn, actor, "hold_placed", cust, alert_id, note=note, severity=alert["severity"])
                notice = self._insert_notice(conn, cust, alert_id, "hold")
                self._event(conn, "system", "notice_created", cust, alert_id, notice="hold", simulated=True)
            return {"alert": self._alert(conn, alert_id), "account": self._account_row(conn, cust), "notice": notice}

    def dismiss(self, alert_id, decided_by, note="") -> dict:
        actor, note = _actor(decided_by), _note(note, required=False)
        with self._tx() as conn:
            alert = self._alert(conn, alert_id)
            if alert["status"] != "open":
                raise Conflict(f"Alert {alert_id} was already {alert['status']}.")
            conn.execute("UPDATE alerts SET status='dismissed', decision='dismiss', decided_by=?, decided_at=?, "
                         "decision_note=? WHERE id=?", (actor, _now(), note, alert_id))
            self._event(conn, actor, "alert_dismissed", alert["customer_id"], alert_id, note=note)
            return {"alert": self._alert(conn, alert_id),
                    "account": self._account_row(conn, alert["customer_id"]), "notice": None}

    def _change_account(self, customer_id, decided_by, note, *, require, to, kind, event):
        actor, note = _actor(decided_by), _note(note, required=True)
        with self._tx() as conn:
            acct = self._account_row(conn, customer_id)
            if acct["status"] != require:
                raise Conflict(f"Account {customer_id} is {acct['status']}; it must be {require} to do this.")
            conn.execute("UPDATE accounts SET status=?, updated_at=?, updated_by=?, reason=? WHERE customer_id=?",
                         (to, _now(), actor, note, customer_id))
            self._event(conn, actor, event, customer_id, None, note=note)
            notice = self._insert_notice(conn, customer_id, None, kind)
            self._event(conn, "system", "notice_created", customer_id, None, notice=kind, simulated=True)
            return {"account": self._account_row(conn, customer_id), "notice": notice}

    def release(self, customer_id, decided_by, note) -> dict:
        return self._change_account(customer_id, decided_by, note, require="held", to="active",
                                    kind="release", event="hold_released")

    def close(self, customer_id, decided_by, note) -> dict:
        return self._change_account(customer_id, decided_by, note, require="held", to="closed",
                                    kind="close", event="account_closed")
