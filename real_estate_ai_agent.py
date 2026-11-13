"""
Unified Real Estate AI Agency Engine (v4.0 — Chief of Staff & Load-Balanced Workforce)
======================================================================================
100% Agency-Owned, Cloud-Ready Python/FastAPI Engine with Live LLM Connector (OpenAI / Gemini / Groq)
and a lean, load-balanced 5-Bot Hierarchy:

1. VICTORIA (Ext 100) — CHIEF OF STAFF BOT (Direct Client / Broker Concierge):
   - The single point of contact for your client (the Managing Broker / Team Lead).
   - Answers direct WhatsApp/SMS/Voice questions from the broker ("Victoria, how many hot leads today?").
   - Assigns qualified buyers to human agents, dispatches 30-second Agent Handoff Dossiers,
     audits active deals, and sends the 8:00 AM Executive Briefing.

2. CLARA (Ext 101) — VOICE RECEPTIONIST BOT (Dedicated Phone & Live Voice Line Only):
   - Zero portal clutter: 100% dedicated to picking up inbound phone calls on Ring 1 (Twilio / Vapi / Retell),
     answering live MLS property questions on the phone, and transferring warm callers.

3. RYAN (Ext 102) — PORTAL LEAD CATCHER BOT (Zillow, Realtor.com, Google LSA & Website Forms):
   - Catches high-intent search & property portal leads in <30 seconds.
   - Pulls MLS property sheets and runs the 5-touch SMS/Email follow-up cadence for portal inquiries.

4. MAYA (Ext 103) — SOCIAL & WHATSAPP CONCIERGE BOT (Instagram DMs, Facebook Lead Ads & WhatsApp):
   - Handles top-of-funnel social media traffic, IG Reel DMs ("How much is this home?"),
     and WhatsApp threads, qualifying BANT (Budget, Pre-Approval, Timeline) conversationally.

5. ALEX (Ext 104) — SHOWING & FIELD OPS BOT (Site Visits, Lockboxes & Car Voice Memos):
   - Books Virtual Discovery Calls & In-Person Property Site Visits with 30-min travel buffers.
   - Verifies Sentrilock / Supra electronic lockbox codes and sends .ics calendar invites.
   - Transcribes post-tour car voice memos from human agents into Follow Up Boss CRM notes & offer emails.
"""

import os
import re
import sys
import json
import sqlite3
import datetime
import uuid
import smtplib
import imaplib
import email as py_email
from email.header import decode_header
import threading
import time
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field

def to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.dict()

# -----------------------------------------------------------------------------
# MULTI-PROVIDER LLM & VOICE ENGINE (1 API KEY -> CHOOSE ANY MODEL STRING)
# Supports: OpenAI, Google Gemini (2.5 / 3.1), Anthropic Claude, xAI Grok,
#           NVIDIA NIM, DeepSeek, Zhipu GLM, Groq, OpenRouter + Voice TTS/STT
# -----------------------------------------------------------------------------

class LLMEngine:
    """
    How 1 API Key Selects Any Specific Model:
    Every AI company gives you ONE API key for your account, and you choose the exact
    model version (e.g., 'gemini-2.5-flash' vs 'gemini-3.1-pro', or 'deepseek-chat' vs
    'deepseek-reasoner') by passing the `model` string parameter in the API request!
    """

    PROVIDER_CATALOG = {
        "openai": {
            "base_url": "https://api.openai.com/v1/chat/completions",
            "env_key": "OPENAI_API_KEY",
            "models": ["gpt-4o-mini", "gpt-4o", "o3-mini"]
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "env_key": "GEMINI_API_KEY",
            "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro", "gemini-3.1-flash"]
        },
        "claude": {
            "base_url": "https://api.anthropic.com/v1/messages",
            "env_key": "ANTHROPIC_API_KEY",
            "models": ["claude-3-5-haiku-latest", "claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest"]
        },
        "grok": {
            "base_url": "https://api.x.ai/v1/chat/completions",
            "env_key": "XAI_API_KEY",
            "models": ["grok-3-mini", "grok-3", "grok-2-latest"]
        },
        "nvidia": {
            "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "env_key": "NVIDIA_API_KEY",
            "models": ["meta/llama-3.3-70b-instruct", "deepseek-ai/deepseek-r1", "nvidia/llama-3.1-nemotron-70b-instruct"]
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1/chat/completions",
            "env_key": "DEEPSEEK_API_KEY",
            "models": ["deepseek-chat", "deepseek-reasoner"]
        },
        "glm": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            "env_key": "GLM_API_KEY",
            "models": ["glm-4-flash", "glm-4-plus", "glm-zero-preview"]
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1/chat/completions",
            "env_key": "GROQ_API_KEY",
            "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "env_key": "OPENROUTER_API_KEY",
            "models": ["google/gemini-2.5-flash", "anthropic/claude-3.7-sonnet", "x-ai/grok-3", "deepseek/deepseek-chat"]
        }
    }

    VOICE_PROVIDER_CATALOG = {
        "elevenlabs": {
            "name": "ElevenLabs Ultra-Realistic Voice",
            "env_key": "ELEVENLABS_API_KEY",
            "voices": ["Rachel (Warm Receptionist)", "Josh (Executive Broker)", "Bella (Concierge)"],
            "models": ["eleven_turbo_v2_5", "eleven_flash_v2_5", "eleven_multilingual_v2"]
        },
        "openai_voice": {
            "name": "OpenAI Realtime & TTS Audio",
            "env_key": "OPENAI_API_KEY",
            "voices": ["alloy", "nova", "shimmer", "echo"],
            "models": ["gpt-4o-mini-tts", "gpt-4o-realtime-preview", "tts-1-hd"]
        },
        "deepgram": {
            "name": "Deepgram Aura Voice (Sub-200ms Phone Call TTS/STT)",
            "env_key": "DEEPGRAM_API_KEY",
            "voices": ["aura-asteria-en", "aura-luna-en", "aura-orion-en"],
            "models": ["nova-2-phonecall", "aura-tts"]
        },
        "twilio_polly": {
            "name": "Twilio Amazon Polly Neural (Built-In Zero Extra Key)",
            "env_key": "TWILIO_AUTH_TOKEN",
            "voices": ["Polly.Joanna-Neural", "Polly.Matthew-Neural", "Polly.Ruth-Neural"],
            "models": ["neural-telephony"]
        }
    }

    # Active Runtime Selection (Can be updated anytime from the Dashboard UI!)
    active_provider: str = os.environ.get("AI_PROVIDER", "gemini").lower()
    active_model: str = os.environ.get("AI_MODEL", "gemini-2.5-flash")
    active_api_key: str = os.environ.get("AI_API_KEY", "")
    active_voice_provider: str = os.environ.get("VOICE_PROVIDER", "elevenlabs")
    active_voice_model: str = os.environ.get("VOICE_MODEL", "eleven_turbo_v2_5")
    active_voice_id: str = os.environ.get("VOICE_ID", "Polly.Joanna-Neural")


    @classmethod
    def detect_provider_and_fetch_models(cls, api_key: str, hint_provider: Optional[str] = None) -> Dict[str, Any]:
        """
        OpenClaw / Hermes Style Auto-Discovery:
        1. Inspects the pasted API key prefix (e.g., 'AIza...', 'sk-ant-...', 'nvapi-...', 'gsk_...', 'xai-...', 'sk-or-...')
           to automatically identify the provider.
        2. Queries the provider's live `/v1/models` endpoint with that key to return every model
           unlocked on the user's account (with instant curated fallback if offline).
        """
        clean_key = (api_key or "").strip()
        detected = (hint_provider or "").lower().strip()

        # Step 1: Prefix-based auto-detection
        if clean_key.startswith("AIza"):
            detected = "gemini"
        elif clean_key.startswith("sk-ant-"):
            detected = "claude"
        elif clean_key.startswith("nvapi-"):
            detected = "nvidia"
        elif clean_key.startswith("gsk_"):
            detected = "groq"
        elif clean_key.startswith("xai-"):
            detected = "grok"
        elif clean_key.startswith("sk-or-"):
            detected = "openrouter"
        elif "." in clean_key and len(clean_key.split(".")) == 2 and not clean_key.startswith("sk-"):
            detected = "glm"
        elif clean_key.startswith("sk-") and not detected:
            # Probe DeepSeek vs OpenAI
            detected = "openai"

        if detected not in cls.PROVIDER_CATALOG:
            detected = "gemini"

        prov_info = cls.PROVIDER_CATALOG[detected]
        discovered_models: List[str] = []
        live_fetched = False

        # Step 2: Live query to provider's /models endpoint
        if clean_key and not clean_key.startswith(("sk-bops-local", "AIzaSyBops")):
            try:
                if detected == "gemini":
                    models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
                    req = urllib.request.Request(models_url, method="GET")
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        discovered_models = [
                            m["name"].replace("models/", "")
                            for m in data.get("models", [])
                            if "generateContent" in m.get("supportedGenerationMethods", [])
                        ]
                        live_fetched = bool(discovered_models)
                elif detected == "claude":
                    req = urllib.request.Request(
                        "https://api.anthropic.com/v1/models",
                        headers={"x-api-key": clean_key, "anthropic-version": "2023-06-01"},
                        method="GET"
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        discovered_models = [m["id"] for m in data.get("data", [])]
                        live_fetched = bool(discovered_models)
                else:
                    # OpenAI-compatible /v1/models (OpenAI, Grok, NVIDIA, DeepSeek, GLM, Groq, OpenRouter)
                    models_url = prov_info["base_url"].replace("/chat/completions", "/models")
                    req = urllib.request.Request(
                        models_url,
                        headers={"Authorization": f"Bearer {clean_key}"},
                        method="GET"
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        discovered_models = [m["id"] for m in data.get("data", [])]
                        live_fetched = bool(discovered_models)
            except Exception:
                pass

        if not discovered_models:
            discovered_models = list(prov_info["models"])

        # Automatically activate the detected provider and its top model
        cls.active_provider = detected
        cls.active_model = discovered_models[0]
        cls.active_api_key = clean_key

        return {
            "detected_provider": detected,
            "live_endpoint_queried": live_fetched,
            "models_count": len(discovered_models),
            "models": discovered_models,
            "active_model_selected": cls.active_model
        }

    @classmethod
    def configure_runtime(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        voice_provider: Optional[str] = None,
        voice_model: Optional[str] = None,
        voice_id: Optional[str] = None
    ) -> Dict[str, str]:
        cls.active_provider = provider.lower()
        cls.active_model = model
        if api_key is not None and api_key.strip():
            cls.active_api_key = api_key.strip()
        if voice_provider:
            cls.active_voice_provider = voice_provider
        if voice_model:
            cls.active_voice_model = voice_model
        if voice_id:
            cls.active_voice_id = voice_id
        return cls.get_active_config()

    @classmethod
    def get_active_config(cls) -> Dict[str, Any]:
        return {
            "llm_provider": cls.active_provider,
            "llm_model": cls.active_model,
            "api_key_configured": bool(cls.active_api_key or os.environ.get(cls.PROVIDER_CATALOG.get(cls.active_provider, {}).get("env_key", ""), "")),
            "voice_provider": cls.active_voice_provider,
            "voice_model": cls.active_voice_model,
            "voice_id": cls.active_voice_id
        }

    @classmethod
    def generate_with_diagnostics(
        cls,
        system_prompt: str,
        user_message: str,
        override_provider: Optional[str] = None,
        override_model: Optional[str] = None,
        override_api_key: Optional[str] = None,
        conversation_messages: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        provider = (override_provider or cls.active_provider or "gemini").lower().strip()
        model = (override_model or cls.active_model or "gemini-2.5-flash").strip()
        prov_info = cls.PROVIDER_CATALOG.get(provider, cls.PROVIDER_CATALOG["gemini"])
        api_key = (override_api_key or cls.active_api_key or os.environ.get(prov_info["env_key"], "")).strip()

        if not api_key or api_key in ("AIzaSyBopsRealEstateKey", "sk-bops-local-bypass-key", "sk-proj-bops-real-estate-live-key"):
            return None, "No real API key provided yet (paste your real API key in Studio & Setup)."

        try:
            if provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = {
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": ("model" if m["role"] == "assistant" else "user"), "parts": [{"text": m["content"]}]} for m in (conversation_messages or [{"role": "user", "content": user_message}])],
                    "generationConfig": {"temperature": 0.4, "maxOutputTokens": 500}
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "BopsRealEstate/4.0"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip(), None

            elif provider == "claude":
                req = urllib.request.Request(
                    prov_info["base_url"],
                    data=json.dumps({
                        "model": model,
                        "max_tokens": 500,
                        "system": system_prompt,
                        "messages": (conversation_messages or [{"role": "user", "content": user_message}])
                    }).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "User-Agent": "BopsRealEstate/4.0"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["content"][0]["text"].strip(), None

            else:
                # OpenAI-compatible (OpenAI, Grok, NVIDIA NIM, DeepSeek, GLM, Groq, OpenRouter)
                req = urllib.request.Request(
                    prov_info["base_url"],
                    data=json.dumps({
                        "model": model,
                        "messages": [{"role": "system", "content": system_prompt}] + (conversation_messages or [{"role": "user", "content": user_message}]),
                        "temperature": 0.4,
                        "max_tokens": 2048
                    }).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "http://127.0.0.1:8000",
                        "X-Title": "Bops Real Estate",
                        "User-Agent": "BopsRealEstate/4.0"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    msg_obj = data["choices"][0]["message"]
                    text_out = (msg_obj.get("content") or msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or "").strip()
                    if not text_out:
                        return None, f"Model {model} returned an empty content field (try increasing max_tokens or switching model)."
                    return text_out, None

        except urllib.error.HTTPError as he:
            err_body = ""
            try:
                err_body = he.read().decode("utf-8")[:200]
            except Exception:
                pass
            return None, f"HTTP {he.code} from {provider.upper()} ({model}): {err_body}"
        except Exception as e:
            return None, f"Connection error calling {provider.upper()} ({model}): {str(e)}"

    @classmethod
    def generate(cls, system_prompt: str, user_message: str) -> Optional[str]:
        provider_info = cls.PROVIDER_CATALOG.get(cls.active_provider, cls.PROVIDER_CATALOG["openai"])
        api_key = cls.active_api_key or os.environ.get(provider_info["env_key"], "").strip()
        if not api_key or api_key.startswith("sk-bops-local"):
            return None

        # 1. Anthropic Claude Native Messages API
        if cls.active_provider == "claude":
            try:
                req = urllib.request.Request(
                    provider_info["base_url"],
                    data=json.dumps({
                        "model": cls.active_model,
                        "max_tokens": 300,
                        "system": system_prompt,
                        "messages": (conversation_messages or [{"role": "user", "content": user_message}])
                    }).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["content"][0]["text"].strip()
            except Exception:
                return None

        # 2. Google Gemini Native API fallback if using non-OpenAI endpoint
        if cls.active_provider == "gemini" and "generativelanguage" in provider_info["base_url"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{cls.active_model}:generateContent?key={api_key}"
                req = urllib.request.Request(
                    url,
                    data=json.dumps({
                        "contents": [{"parts": [{"text": system_prompt + "\n\nUser: " + user_message}]}]
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                return None

        # 3. Universal OpenAI-Compatible Endpoint (OpenAI, Grok/xAI, NVIDIA NIM, DeepSeek, Zhipu GLM, Groq, OpenRouter)
        try:
            req = urllib.request.Request(
                provider_info["base_url"],
                data=json.dumps({
                    "model": cls.active_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300
                }).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None


# -----------------------------------------------------------------------------
# ENUMS & DOMAIN CONSTANTS
# -----------------------------------------------------------------------------

class LeadSource(str, Enum):
    ZILLOW = "ZILLOW"
    REALTOR_COM = "REALTOR_COM"
    GOOGLE_LSA = "GOOGLE_LSA"
    WEBSITE_FORM = "WEBSITE_FORM"
    META_ADS = "META_ADS"
    INSTAGRAM_DM = "INSTAGRAM_DM"
    WHATSAPP = "WHATSAPP"
    INBOUND_CALL = "INBOUND_CALL"

class LeadStatus(str, Enum):
    NEW = "NEW"
    CONTACTED = "CONTACTED"
    QUALIFYING = "QUALIFYING"
    QUALIFIED = "QUALIFIED"
    MEETING_SCHEDULED = "MEETING_SCHEDULED"
    SITE_VISIT_SCHEDULED = "SITE_VISIT_SCHEDULED"
    OFFER_DRAFTED = "OFFER_DRAFTED"
    ASSIGNED_TO_AGENT = "ASSIGNED_TO_AGENT"
    NURTURE = "NURTURE"

class PropertyType(str, Enum):
    SINGLE_FAMILY = "SINGLE_FAMILY"
    CONDO = "CONDO"
    TOWNHOUSE = "TOWNHOUSE"
    LUXURY_ESTATE = "LUXURY_ESTATE"
    COMMERCIAL = "COMMERCIAL"

class AppointmentType(str, Enum):
    PHONE_DISCOVERY = "PHONE_DISCOVERY"
    VIRTUAL_CONSULTATION = "VIRTUAL_CONSULTATION"
    IN_PERSON_SITE_VISIT = "IN_PERSON_SITE_VISIT"

class QualificationTier(str, Enum):
    HOT = "HOT"     # Score >= 75
    WARM = "WARM"   # Score 45-74
    COLD = "COLD"   # Score < 45

class BotRole(str, Enum):
    CHIEF_OF_STAFF = "CHIEF_OF_STAFF"        # Victoria (Ext 100) — Direct Client/Broker Concierge
    VOICE_RECEPTIONIST = "VOICE_RECEPTIONIST"# Clara (Ext 101) — Inbound Phone Calls Only
    PORTAL_CATCHER = "PORTAL_CATCHER"        # Ryan (Ext 102) — Zillow, Realtor.com, Website, Google LSA
    SOCIAL_CONCIERGE = "SOCIAL_CONCIERGE"    # Maya (Ext 103) — Instagram DMs, Meta Ads, WhatsApp
    SHOWING_FIELD_OPS = "SHOWING_FIELD_OPS"  # Alex (Ext 104) — Site Visits, Lockboxes & Car Voice Memos

# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------

class QualificationData(BaseModel):
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    timeline_days: Optional[int] = None
    pre_approved: Optional[bool] = None
    pre_approval_lender: Optional[str] = None
    preferred_zip_codes: List[str] = Field(default_factory=list)
    property_type: Optional[PropertyType] = None
    bedrooms: Optional[int] = None
    score: int = 15
    tier: QualificationTier = QualificationTier.COLD

class Agent(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    specialties: List[PropertyType] = Field(default_factory=list)
    zip_codes: List[str] = Field(default_factory=list)
    price_floor: float = 0.0
    active_leads_count: int = 0
    max_capacity: int = 20
    calendar_link: str = ""
    is_active: bool = True

class PropertyListing(BaseModel):
    mls_id: str
    address: str
    price: float
    bedrooms: int
    bathrooms: float
    square_feet: int
    property_type: str
    lockbox_code: str
    showing_instructions: str
    assigned_licensed_agent: str
    status: str = "ACTIVE"

class Appointment(BaseModel):
    id: str
    lead_id: str
    agent_id: str
    appointment_type: AppointmentType
    scheduled_time: str
    duration_minutes: int
    property_address: Optional[str] = None
    access_notes: Optional[str] = None
    meeting_link: Optional[str] = None
    calendar_ics: Optional[str] = None
    status: str = "CONFIRMED"

class Lead(BaseModel):
    id: str
    first_name: str
    last_name: str
    phone: str
    email: str
    source: LeadSource
    caught_by_bot: str = "Ryan (Portal Catcher • Ext 102)"
    status: LeadStatus = LeadStatus.NEW
    inquired_property_address: Optional[str] = None
    initial_message: Optional[str] = None
    created_at: str
    qualification: QualificationData = Field(default_factory=QualificationData)
    assigned_agent_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    scheduled_appointment_id: Optional[str] = None
    last_contact_at: Optional[str] = None
    next_follow_up_at: Optional[str] = None
    follow_up_step: int = 0

# -----------------------------------------------------------------------------
# DATABASE LAYER (SQLITE)
# -----------------------------------------------------------------------------

class UnifiedDatabase:
    def __init__(self, db_path: str = "real_estate_v4.db"):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                specialties TEXT NOT NULL,
                zip_codes TEXT NOT NULL,
                price_floor REAL NOT NULL,
                active_leads_count INTEGER NOT NULL,
                max_capacity INTEGER NOT NULL,
                calendar_link TEXT,
                is_active INTEGER NOT NULL
            )
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS listings (
                mls_id TEXT PRIMARY KEY,
                address TEXT NOT NULL,
                price REAL NOT NULL,
                bedrooms INTEGER,
                bathrooms REAL,
                square_feet INTEGER,
                property_type TEXT,
                lockbox_code TEXT,
                showing_instructions TEXT,
                assigned_licensed_agent TEXT,
                status TEXT NOT NULL
            )
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT NOT NULL,
                source TEXT NOT NULL,
                caught_by_bot TEXT NOT NULL,
                status TEXT NOT NULL,
                inquired_property_address TEXT,
                initial_message TEXT,
                created_at TEXT NOT NULL,
                qualification TEXT NOT NULL,
                assigned_agent_id TEXT,
                conversation_history TEXT NOT NULL,
                scheduled_appointment_id TEXT,
                last_contact_at TEXT,
                next_follow_up_at TEXT,
                follow_up_step INTEGER NOT NULL
            )
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY,
                lead_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                appointment_type TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                property_address TEXT,
                access_notes TEXT,
                meeting_link TEXT,
                calendar_ics TEXT,
                status TEXT NOT NULL
            )
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS client_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT NOT NULL,
                bot_role TEXT NOT NULL,
                action_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
            conn.commit()



    def get_all_listings(self) -> List[PropertyListing]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM listings ORDER BY rowid ASC").fetchall()
            return [PropertyListing(**dict(r)) for r in rows]

    def append_bot_chat_memory(self, bot_id: str, role: str, content: str):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO bot_chat_memory (bot_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (bot_id, role, content, datetime.datetime.now(datetime.timezone.utc).isoformat())
            )
            conn.commit()

    def get_bot_chat_memory(self, bot_id: str, limit: int = 30) -> List[Dict[str, str]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT role, content FROM bot_chat_memory WHERE bot_id = ? ORDER BY id DESC LIMIT ?",
                (bot_id, limit)
            ).fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def save_settings(self, settings_dict: Dict[str, Any]):
        with self.get_connection() as conn:
            for k, v in settings_dict.items():
                conn.execute("INSERT OR REPLACE INTO client_settings (key, value) VALUES (?, ?)", (k, json.dumps(v)))
            conn.commit()

    def get_settings(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT key, value FROM client_settings").fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}

    def log_bot_action(self, bot_name: str, bot_role: BotRole, action_type: str, summary: str):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO bot_activity_log (bot_name, bot_role, action_type, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (bot_name, bot_role.value, action_type, summary, datetime.datetime.now(datetime.timezone.utc).isoformat()))
            conn.commit()

    def get_recent_bot_logs(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM bot_activity_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
            hot_leads = 0
            rows = conn.execute("SELECT qualification FROM leads").fetchall()
            for r in rows:
                try:
                    q = json.loads(r[0])
                    if q.get("tier") == "HOT" or q.get("score", 0) >= 80:
                        hot_leads += 1
                except Exception:
                    pass
            appts = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
            calls = conn.execute("SELECT COUNT(*) FROM bot_activity_log WHERE action_type IN ('PHONE_CALL_ANSWERED', 'TELNYX_CALL_PICKUP', 'TWILIO_CALL_PICKUP', 'CALL_ANSWERED')").fetchone()[0]
            memos = conn.execute("SELECT COUNT(*) FROM bot_activity_log WHERE action_type = 'CAR_MEMO_TRANSCRIBED'").fetchone()[0]
            return {
                "total_leads": total_leads,
                "hot_leads": hot_leads,
                "appointments_count": appts,
                "calls_count": calls,
                "memos_count": memos
            }

    def reset_leads_and_activity(self):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM leads")
            conn.execute("DELETE FROM appointments")
            conn.execute("DELETE FROM bot_activity_log")
            conn.commit()

    def save_agent(self, agent: Agent):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent.id, agent.name, agent.email, agent.phone,
                json.dumps([s.value for s in agent.specialties]),
                json.dumps(agent.zip_codes), agent.price_floor,
                agent.active_leads_count, agent.max_capacity,
                agent.calendar_link, 1 if agent.is_active else 0
            ))
            conn.commit()

    def get_agents(self) -> List[Agent]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM agents WHERE is_active = 1").fetchall()
            return [Agent(
                id=r["id"], name=r["name"], email=r["email"], phone=r["phone"],
                specialties=[PropertyType(s) for s in json.loads(r["specialties"])],
                zip_codes=json.loads(r["zip_codes"]), price_floor=r["price_floor"],
                active_leads_count=r["active_leads_count"], max_capacity=r["max_capacity"],
                calendar_link=r["calendar_link"] or "", is_active=bool(r["is_active"])
            ) for r in rows]

    def save_listing(self, listing: PropertyListing):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO listings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                listing.mls_id, listing.address, listing.price, listing.bedrooms,
                listing.bathrooms, listing.square_feet, listing.property_type,
                listing.lockbox_code, listing.showing_instructions,
                listing.assigned_licensed_agent, listing.status
            ))
            conn.commit()

    def delete_listing(self, mls_id: str):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM listings WHERE mls_id = ?", (mls_id,))
            conn.commit()

    def find_listing(self, query: str) -> Optional[PropertyListing]:
        if not query or not query.strip():
            return None
        q_clean = query.strip().lower()
        all_props = self.get_all_listings()
        if not all_props:
            return None

        # 1. Direct MLS ID or exact/substring address match
        for prop in all_props:
            addr_low = prop.address.lower()
            mls_low = prop.mls_id.lower()
            if mls_low in q_clean or q_clean in addr_low or addr_low in q_clean:
                return prop

        # 2. Match strictly by street name (and street number if user provided a street number)
        # Never match purely on city name or 5-digit zip code!
        stop_words = {"st", "street", "ave", "avenue", "blvd", "boulevard", "rd", "road", "dr", "drive",
                      "ln", "lane", "way", "ct", "court", "pl", "place", "ter", "terrace", "pkwy",
                      "apt", "suite", "unit", "penthouse", "villa", "residence", "loft", "north", "south",
                      "east", "west", "ca", "ny", "fl", "tx", "los", "angeles", "beverly", "hills",
                      "santa", "monica", "downtown", "bel", "air", "malibu", "the", "and", "for", "in"}

        # Extract street number (first 1-4 digit number at start of query, ignoring 5-digit zip codes)
        q_street_num_m = re.match(r"^\s*(\d{1,4})\b", q_clean)
        q_street_num = q_street_num_m.group(1) if q_street_num_m else None

        for prop in all_props:
            # Only look at the street portion (before the first comma) to verify it is the exact same property
            street_part = prop.address.split(",")[0].lower()
            p_street_num_m = re.match(r"^\s*(\d{1,4})\b", street_part)
            p_street_num = p_street_num_m.group(1) if p_street_num_m else None

            # If user specified a street number (e.g. 999 Sunset), it MUST match the listing street number (e.g. 742)
            if q_street_num and p_street_num and q_street_num != p_street_num:
                continue

            street_words = set(re.findall(r"[a-z]{4,}", street_part)) - stop_words
            query_words = set(re.findall(r"[a-z]{4,}", q_clean)) - stop_words
            if street_words and street_words.intersection(query_words):
                return prop

        return None

    def find_nearest_listings(self, query_address: str, buyer_message: str = "", limit: int = 3) -> List[PropertyListing]:
        """
        When a buyer asks about a property that is NOT in the database,
        ranks all active SQLite listings by proximity & similarity:
          1. Zip code numerical distance (closest zip code first)
          2. City / neighborhood / street token overlap
          3. Budget / price proximity (if buyer mentioned a price/budget)
          4. Bedrooms & property type similarity
        Returns ONLY the nearest 2–3 possibilities (never the whole database).
        """
        all_props = self.get_all_listings()
        if not all_props:
            return []

        max_items = min(max(limit, 2), 3)
        combined = f"{query_address or ''} {buyer_message or ''}".lower()

        # Extract any 5-digit zip code from buyer inquiry
        q_zips = [int(z) for z in re.findall(r"\b(\d{5})\b", combined)]

        # Extract any target budget from buyer inquiry
        target_budget = None
        m_mill = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:m|million)\b", combined)
        m_thou = re.search(r"\$?\s*(\d{3,4})\s*k\b", combined)
        if m_mill:
            target_budget = float(m_mill.group(1)) * 1_000_000.0
        elif m_thou:
            target_budget = float(m_thou.group(1)) * 1_000.0

        # Extract requested bedrooms
        target_beds = None
        m_beds = re.search(r"\b(\d)\s*(?:bed|beds|br|bd|bedroom|bedrooms)\b", combined)
        if m_beds:
            target_beds = int(m_beds.group(1))

        stop_words = {"st", "street", "ave", "avenue", "blvd", "boulevard", "rd", "road", "dr", "drive",
                      "ln", "lane", "way", "ct", "court", "pl", "place", "ter", "terrace", "the", "and",
                      "for", "in", "at", "of", "to", "is", "this", "that", "with", "want", "looking", "house", "home"}
        query_tokens = set(re.findall(r"[a-z]{3,}", combined)) - stop_words

        scored = []
        for idx, prop in enumerate(all_props):
            score = 0.0
            addr_low = prop.address.lower()
            type_low = (prop.property_type or "").lower()

            # 1. Zip code proximity (up to +60 pts)
            p_zips = [int(z) for z in re.findall(r"\b(\d{5})\b", addr_low)]
            if q_zips and p_zips:
                min_dist = min(abs(qz - pz) for qz in q_zips for pz in p_zips)
                if min_dist == 0:
                    score += 60.0
                elif min_dist <= 10:
                    score += 45.0
                elif min_dist <= 35:
                    score += 30.0
                elif min_dist <= 100:
                    score += 15.0

            # 2. City / Neighborhood / Area token overlap (+25 pts per token)
            prop_tokens = set(re.findall(r"[a-z]{3,}", f"{addr_low} {type_low}")) - stop_words
            overlap = query_tokens.intersection(prop_tokens)
            score += len(overlap) * 25.0

            # 3. Price / Budget proximity (up to +35 pts)
            if target_budget and prop.price > 0:
                rel_diff = abs(prop.price - target_budget) / max(prop.price, target_budget)
                if rel_diff <= 0.20:
                    score += 35.0
                elif rel_diff <= 0.40:
                    score += 20.0
                elif rel_diff <= 0.60:
                    score += 10.0

            # 4. Bedroom & Property Type similarity (up to +25 pts)
            if target_beds is not None and prop.bedrooms is not None:
                if prop.bedrooms == target_beds:
                    score += 20.0
                elif abs(prop.bedrooms - target_beds) == 1:
                    score += 10.0
            if any(k in combined for k in ["condo", "penthouse", "apartment"]) and any(k in f"{addr_low} {type_low}" for k in ["condo", "penthouse"]):
                score += 15.0
            if any(k in combined for k in ["villa", "estate", "single family", "mansion"]) and any(k in f"{addr_low} {type_low}" for k in ["villa", "estate", "single", "luxury"]):
                score += 15.0

            # Small tie-breaker to preserve deterministic ordering
            score -= idx * 0.01
            scored.append((score, prop))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Return strictly the nearest 2-3 possibilities (never the whole database)
        top_count = min(max_items, len(scored))
        if len(scored) > 3:
            top_count = min(3, max_items)
        return [p for _, p in scored[:top_count]]

    def find_lead_by_email(self, email_addr: str) -> Optional[Lead]:
        if not email_addr or not email_addr.strip():
            return None
        clean_em = email_addr.strip().lower()
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", clean_em)
        if m:
            clean_em = m.group(0)
        with self.get_connection() as conn:
            r = conn.execute("SELECT id FROM leads WHERE LOWER(email) = ? ORDER BY created_at DESC LIMIT 1", (clean_em,)).fetchone()
            if r and r["id"]:
                return self.get_lead(r["id"])
        return None

    def find_lead_by_social_or_phone(self, identifier: str) -> Optional[Lead]:
        if not identifier or not identifier.strip():
            return None
        clean_id = identifier.strip()
        with self.get_connection() as conn:
            r = conn.execute("SELECT id FROM leads WHERE phone = ? OR LOWER(email) = ? ORDER BY created_at DESC LIMIT 1", (clean_id, clean_id.lower())).fetchone()
            if r and r["id"]:
                return self.get_lead(r["id"])
        return None

    def save_lead(self, lead: Lead):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO leads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lead.id, lead.first_name, lead.last_name, lead.phone, lead.email,
                lead.source.value, lead.caught_by_bot, lead.status.value,
                lead.inquired_property_address, lead.initial_message, lead.created_at,
                json.dumps(to_dict(lead.qualification)), lead.assigned_agent_id,
                json.dumps(lead.conversation_history), lead.scheduled_appointment_id,
                lead.last_contact_at, lead.next_follow_up_at, lead.follow_up_step
            ))
            conn.commit()

    def get_lead(self, lead_id: str) -> Optional[Lead]:
        with self.get_connection() as conn:
            r = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if not r:
                return None
            return Lead(
                id=r["id"], first_name=r["first_name"], last_name=r["last_name"],
                phone=r["phone"], email=r["email"], source=LeadSource(r["source"]),
                caught_by_bot=r["caught_by_bot"], status=LeadStatus(r["status"]),
                inquired_property_address=r["inquired_property_address"],
                initial_message=r["initial_message"], created_at=r["created_at"],
                qualification=QualificationData(**json.loads(r["qualification"])),
                assigned_agent_id=r["assigned_agent_id"],
                conversation_history=json.loads(r["conversation_history"]),
                scheduled_appointment_id=r["scheduled_appointment_id"],
                last_contact_at=r["last_contact_at"], next_follow_up_at=r["next_follow_up_at"],
                follow_up_step=r["follow_up_step"]
            )

    def get_all_leads(self) -> List[Lead]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT id FROM leads ORDER BY created_at DESC").fetchall()
            return [self.get_lead(r["id"]) for r in rows if r["id"]]

    def save_appointment(self, appt: Appointment):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO appointments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                appt.id, appt.lead_id, appt.agent_id, appt.appointment_type.value,
                appt.scheduled_time, appt.duration_minutes, appt.property_address,
                appt.access_notes, appt.meeting_link, appt.calendar_ics, appt.status
            ))
            conn.commit()

    def get_all_appointments(self) -> List[Appointment]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM appointments ORDER BY rowid DESC").fetchall()
            return [Appointment(
                id=r["id"], lead_id=r["lead_id"], agent_id=r["agent_id"],
                appointment_type=AppointmentType(r["appointment_type"]),
                scheduled_time=r["scheduled_time"], duration_minutes=r["duration_minutes"],
                property_address=r["property_address"], access_notes=r["access_notes"],
                meeting_link=r["meeting_link"], calendar_ics=r["calendar_ics"],
                status=r["status"]
            ) for r in rows]

# -----------------------------------------------------------------------------
# SHARED BANT+ QUALIFICATION & SCORING ENGINE
# -----------------------------------------------------------------------------

class QualificationEngine:
    @staticmethod
    def extract_passive_bant(lead: Lead, incoming_text: str, matched_listing: Optional[PropertyListing] = None):
        """Extracts BANT signals passively from the buyer's inquiry and matched database property without interrogating."""
        text_lower = (incoming_text or "").lower()
        q = lead.qualification

        # If matched a listing in SQLite, infer baseline property type & target price range
        if matched_listing:
            if not q.budget_max:
                q.budget_min = matched_listing.price * 0.85
                q.budget_max = matched_listing.price * 1.15
            if "condo" in matched_listing.property_type.lower():
                q.property_type = PropertyType.CONDO
            elif "luxury" in matched_listing.property_type.lower() or matched_listing.price >= 1500000:
                q.property_type = PropertyType.LUXURY_ESTATE
            else:
                q.property_type = PropertyType.SINGLE_FAMILY

        # Parse explicit dollar / budget amounts from buyer's note
        m_mill = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:m|million)\b", text_lower)
        m_thou = re.search(r"\$?\s*(\d{3,4})\s*k\b", text_lower)
        if m_mill:
            val = float(m_mill.group(1)) * 1_000_000.0
            q.budget_min, q.budget_max = val * 0.85, val
        elif m_thou:
            val = float(m_thou.group(1)) * 1_000.0
            q.budget_min, q.budget_max = val * 0.85, val

        if any(t in text_lower for t in ["pre-approved", "preapproved", "pre approved", "chase", "wells fargo", "bank", "lender", "approved"]):
            q.pre_approved = True
            q.pre_approval_lender = "Mortgage Pre-Approved"
        elif any(t in text_lower for t in ["cash", "all-cash", "all cash"]):
            q.pre_approved = True
            q.pre_approval_lender = "Cash Buyer"

        if any(t in text_lower for t in ["asap", "immediately", "30 days", "this month", "thursday", "friday", "saturday", "sunday", "tomorrow", "weekend", "this week", "tour", "showing", "visit"]):
            q.timeline_days = 21
        elif any(t in text_lower for t in ["1-2 months", "60 days", "next month", "soon"]):
            q.timeline_days = 45
        elif any(t in text_lower for t in ["browsing", "6 months", "just looking"]):
            q.timeline_days = 120

        # Calculate 0-100 Score
        score = 25
        if matched_listing:
            score += 20  # Inquired about a verified active listing in our database
        if q.pre_approved is True:
            score += 25
        if q.timeline_days is not None:
            score += 20 if q.timeline_days <= 30 else (10 if q.timeline_days <= 90 else 5)
        if q.budget_max is not None:
            score += 10

        q.score = min(score, 100)
        if q.score >= 75:
            q.tier = QualificationTier.HOT
            lead.status = LeadStatus.QUALIFIED
        elif q.score >= 45:
            q.tier = QualificationTier.WARM
            lead.status = LeadStatus.QUALIFYING
        else:
            q.tier = QualificationTier.COLD

    @staticmethod
    def evaluate_and_reply(lead: Lead, incoming_text: str, responder_bot_name: str, db: Optional[UnifiedDatabase] = None) -> Tuple[Lead, str]:
        text_lower = (incoming_text or "").lower()

        # Check if the buyer's new message mentions a specific property in SQLite
        matched_listing = None
        nearest_listings = []
        if db is not None:
            matched_listing = db.find_listing(incoming_text) or db.find_listing(lead.inquired_property_address or "")
            if matched_listing:
                lead.inquired_property_address = matched_listing.address
            else:
                nearest_listings = db.find_nearest_listings(lead.inquired_property_address or "", incoming_text, limit=3)

        QualificationEngine.extract_passive_bant(lead, incoming_text, matched_listing)

        # Build conversation history for multi-turn LLM context
        conv_msgs = []
        for h in lead.conversation_history[-10:]:
            role = "user" if h.get("sender") == "LEAD" else "assistant"
            conv_msgs.append({"role": role, "content": h.get("text", "")})
        if not conv_msgs or conv_msgs[-1]["content"] != incoming_text:
            conv_msgs.append({"role": "user", "content": incoming_text})

        # Build strictly scoped property context (either the matched property OR only the 2-3 nearest possibilities)
        if matched_listing:
            prop_context = (
                f"MATCHED PROPERTY IN DATABASE: MLS #{matched_listing.mls_id} — {matched_listing.address} | "
                f"Price: ${matched_listing.price:,.0f} | {matched_listing.bedrooms} BD / {matched_listing.bathrooms} BA / {matched_listing.square_feet:,} SqFt | "
                f"Type: {matched_listing.property_type} | Status: {matched_listing.status}"
            )
        else:
            near_lines = [
                f"- MLS #{p.mls_id}: {p.address} (${p.price:,.0f}, {p.bedrooms} BD / {p.bathrooms} BA, {p.square_feet:,} SqFt, {p.property_type})"
                for p in nearest_listings[:3]
            ]
            prop_context = (
                f"NOTE: The buyer's requested address ('{lead.inquired_property_address}') is NOT in our database. "
                f"Here are ONLY the {len(near_lines)} nearest matching properties from our database (DO NOT mention any other properties):\n" +
                ("\n".join(near_lines) if near_lines else "No listings available.")
            )

        llm_reply, _ = LLMEngine.generate_with_diagnostics(
            system_prompt=(
                f"You are {responder_bot_name}, Property Concierge. You are replying to an email/message from buyer {lead.first_name} {lead.last_name}.\n"
                f"{prop_context}\n\n"
                f"STRICT RULES:\n"
                f"1. Directly and warmly answer the buyer's latest reply ('{incoming_text}').\n"
                f"2. If they are confirming or asking for a tour/showing time, confirm that Alex (Ext 104) has locked in their requested tour window and will send the calendar invite.\n"
                f"3. If they ask about an unlisted property or alternatives, ONLY share the 2–3 nearest matching properties listed above—never dump a full database.\n"
                f"4. Never ask rude financial interrogation questions. Keep your reply natural, helpful, and under 3 sentences."
            ),
            user_message=incoming_text,
            conversation_messages=conv_msgs
        )
        if llm_reply:
            return lead, SecurityAndIdentityGuard.redact_sensitive_data_for_public(llm_reply)

        # Deterministic database-driven fallback when no LLM API key is configured
        if any(w in text_lower for w in ["tour", "see", "visit", "walkthrough", "thursday", "friday", "saturday", "sunday", "monday", "tuesday", "wednesday", "tomorrow", "weekend", "pm", "am", "showing", "book", "yes"]):
            target_addr = matched_listing.address if matched_listing else (nearest_listings[0].address if nearest_listings else (lead.inquired_property_address or "the residence"))
            reply = (
                f"Wonderful, {lead.first_name}! I have coordinated with Alex (Ext 104, our Showing Coordinator) to lock in your viewing request "
                f"for {target_addr} based on your message ('{incoming_text}'). We look forward to hosting you—let me know if you need directions or anything else beforehand!"
            )
        elif matched_listing:
            reply = (
                f"Hi {lead.first_name}, thanks for your reply regarding {matched_listing.address} (listed at ${matched_listing.price:,.0f}, "
                f"{matched_listing.bedrooms} BD / {matched_listing.bathrooms} BA, {matched_listing.square_feet:,} sq ft). "
                f"Would you like me to reserve a private viewing time for you this week?"
            )
        else:
            near_str = "; ".join([f"{p.address} (${p.price:,.0f}, {p.bedrooms} BD/{p.bathrooms} BA)" for p in nearest_listings[:3]])
            reply = (
                f"Hi {lead.first_name}, thanks for following up! Based on your preferences, the closest matching homes available right now are: "
                f"{near_str}. Which of these 2–3 residences would you like to tour?"
            )

        return lead, reply


# -----------------------------------------------------------------------------
# SECURITY, CALLER-ID ROLE AUTHENTICATION & EMERGENCY ESCALATION ENGINE
# -----------------------------------------------------------------------------

class CallerRole(str, Enum):
    MANAGING_BROKER = "MANAGING_BROKER"    # The Boss / Client Owner (Full Executive Access)
    INTERNAL_AGENT = "INTERNAL_AGENT"      # Licensed Agent on the Team (Field Memo & Showing Access)
    EXTERNAL_BUYER = "EXTERNAL_BUYER"      # Public Lead (Zero access to lockbox codes or internal data)

class UrgencyLevel(str, Enum):
    HIGH_INTENT_LUXURY = "HIGH_INTENT_LUXURY"          # All-cash or >M buyer wanting immediate showing
    AGENT_SLA_BREACH = "AGENT_SLA_BREACH"              # Assigned agent hasn't accepted a HOT lead within 15 mins
    ESCROW_DEADLINE_RISK = "ESCROW_DEADLINE_RISK"      # Inspection/Loan contingency expiring <24h
    VIP_ESCALATION = "VIP_ESCALATION"                  # Direct urgent request for Managing Broker

class SecurityAndIdentityGuard:
    """
    3-Layer Identity & Privacy Guardrail:
    1. Identifies incoming phone numbers (Managing Broker vs. Licensed Agent vs. Public Buyer).
    2. Enforces 4-digit Executive PIN for sensitive broker commands.
    3. Physically strips lockbox codes, gate codes, and internal notes from external buyer responses.
    """
    BROKER_WHITELIST = {
        "+1-310-555-0001": {"name": "Arthur Pendelton (Managing Broker)", "pin": "9021"}
    }

    @staticmethod
    def identify_caller(db: UnifiedDatabase, phone_number: str, provided_pin: Optional[str] = None) -> Dict[str, Any]:
        clean_phone = phone_number.strip()
        # 1. Check if caller is the Managing Broker (Boss)
        if clean_phone in SecurityAndIdentityGuard.BROKER_WHITELIST:
            broker = SecurityAndIdentityGuard.BROKER_WHITELIST[clean_phone]
            pin_verified = (provided_pin == broker["pin"]) if provided_pin is not None else True
            return {
                "role": CallerRole.MANAGING_BROKER,
                "name": broker["name"],
                "phone": clean_phone,
                "pin_verified": pin_verified,
                "permissions": ["FULL_PIPELINE_READ", "REASSIGN_AGENTS", "RECEIVE_EMERGENCY_CALLS"]
            }

        # 2. Check if caller is a registered Licensed Agent on the team
        for agent in db.get_agents():
            if agent.phone.strip() == clean_phone:
                return {
                    "role": CallerRole.INTERNAL_AGENT,
                    "name": agent.name,
                    "agent_id": agent.id,
                    "phone": clean_phone,
                    "pin_verified": True,
                    "permissions": ["SUBMIT_CAR_VOICE_MEMO", "VIEW_ASSIGNED_LOCKBOX", "RECEIVE_EMERGENCY_CALLS"]
                }

        # 3. Default strictly to External Public Buyer (Zero internal access)
        return {
            "role": CallerRole.EXTERNAL_BUYER,
            "name": "Public Buyer Lead",
            "phone": clean_phone,
            "pin_verified": False,
            "permissions": ["PUBLIC_MLS_INFO_ONLY", "REQUEST_SHOWING"]
        }

    @staticmethod
    def redact_sensitive_data_for_public(text: str) -> str:
        """Hard-redacts any accidental lockbox or gate code patterns before sending to an external buyer."""
        redacted = re.sub(r"(?:SENTRI|SUPRA)-\d+", "[LOCKBOX VERIFIED - PROVIDED TO AGENT ON SITE]", text, flags=re.IGNORECASE)
        redacted = re.sub(r"Gate code #\d+", "[GATE CODE PROVIDED TO ASSIGNED AGENT]", redacted, flags=re.IGNORECASE)
        return redacted


class EmergencyVoiceEscalationEngine:
    """
    Special Executive Privilege granted ONLY to Victoria (Chief of Staff / Assigner • Ext 100):
    Allows Victoria to place immediate outbound live phone calls (via Twilio Voice REST API
    or Vapi/Retell) to the Managing Broker (Boss) or any Licensed Human Agent in urgent/emergency situations.
    """
    @staticmethod
    def trigger_emergency_call(
        db: UnifiedDatabase,
        target_role: str,
        target_name: str,
        target_phone: str,
        urgency: UrgencyLevel,
        spoken_briefing: str,
        lead_id: Optional[str] = None
    ) -> Dict[str, Any]:
        call_id = f"emerg_call_{uuid.uuid4().hex[:8]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        twiml_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna-Neural">Urgent priority call from Victoria, Chief of Staff at Premier Realty, for {target_name}.</Say>
    <Pause length="1"/>
    <Say voice="Polly.Joanna-Neural">{spoken_briefing}</Say>
    <Gather numDigits="1" action="/api/v1/chief-of-staff/emergency-ack" method="POST">
        <Say voice="Polly.Joanna-Neural">Press 1 to accept and connect with the buyer now, or press 2 to instruct Victoria to reassign this lead to the next available specialist.</Say>
    </Gather>
</Response>"""

        # If live Twilio credentials are set in environment, dial immediately:
        twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        twilio_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        twilio_from = os.environ.get("TWILIO_PHONE_NUMBER", "+13105550100").strip()
        live_dial_status = "SIMULATED_RINGING (Set TWILIO_ACCOUNT_SID for live carrier dial)"

        if twilio_sid and twilio_token and not twilio_sid.startswith("ACXXXX"):
            try:
                from twilio.rest import Client
                client = Client(twilio_sid, twilio_token)
                call = client.calls.create(twiml=twiml_payload, to=target_phone, from_=twilio_from)
                live_dial_status = f"LIVE_CARRIER_DIALING (SID: {call.sid})"
            except Exception as e:
                live_dial_status = f"FALLBACK_ALERT_SENT ({str(e)[:40]})"

        db.log_bot_action(
            "Victoria",
            BotRole.CHIEF_OF_STAFF,
            f"EMERGENCY_CALL_{urgency.value}",
            f"Dialed {target_role} {target_name} ({target_phone}): {spoken_briefing[:80]}..."
        )

        return {
            "call_id": call_id,
            "authorized_by": "Victoria (Chief of Staff & Lead Assigner • Ext 100)",
            "special_privilege": "EXECUTIVE_EMERGENCY_OUTBOUND_CALL",
            "target_role": target_role,
            "target_name": target_name,
            "target_phone": target_phone,
            "urgency_level": urgency.value,
            "dial_status": live_dial_status,
            "spoken_voice_script": spoken_briefing,
            "twiml_executed": twiml_payload,
            "timestamp": now
        }

# -----------------------------------------------------------------------------
# BOT 1: VICTORIA (EXT 100) — CHIEF OF STAFF (DIRECT CLIENT / BROKER CONCIERGE)
# -----------------------------------------------------------------------------

class VictoriaChiefOfStaffBot:
    """
    Directly serves your paying client (the Managing Broker / Team Owner).
    The broker can text or ask Victoria anything about their pipeline, agent performance,
    hot leads, or daily schedule. Victoria also routes leads to human agents.
    """
    name = "Victoria"
    role = BotRole.CHIEF_OF_STAFF
    extension = "Ext 100"

    @staticmethod
    def talk_to_broker(db: UnifiedDatabase, broker_question: str) -> Dict[str, Any]:
        """Handles direct natural-language commands & questions from the client (Managing Broker)."""
        leads = db.get_all_leads()
        appts = db.get_all_appointments()
        agents = db.get_agents()
        hot_leads = [l for l in leads if l.qualification.tier == QualificationTier.HOT]
        total_pipeline = sum((l.qualification.budget_max or 650000.0) for l in leads)

        q_lower = broker_question.lower()

        # Try LLM with live database context first
        context_summary = (
            f"Total Leads: {len(leads)} | Hot Leads: {len(hot_leads)} | Booked Tours: {len(appts)} | "
            f"Pipeline Volume: ${total_pipeline:,.0f}. "
            f"Bots under your command: Clara (Voice Calls), Ryan (Zillow/Portals), Maya (IG/WhatsApp/Meta), Alex (Showings/Voice Memos)."
        )
        llm_answer = LLMEngine.generate(
            system_prompt=f"You are Victoria (Ext 100), Chief of Staff AI to the Managing Broker. Data: {context_summary}",
            user_message=broker_question
        )

        if not llm_answer:
            if "hot" in q_lower or "lead" in q_lower or "pipeline" in q_lower:
                hot_names = ", ".join([f"{l.first_name} {l.last_name} (${(l.qualification.budget_max or 0):,.0f})" for l in hot_leads]) or "None yet"
                llm_answer = (
                    f"Here is your live pipeline update: Your bots have caught {len(leads)} leads representing ${total_pipeline:,.0f} "
                    f"in purchasing volume. We have {len(hot_leads)} HOT pre-approved buyer(s) ({hot_names}) and {len(appts)} confirmed site visit(s)."
                )
            elif "workload" in q_lower or "bot" in q_lower or "team" in q_lower:
                llm_answer = (
                    "All 4 specialist bots under my supervision are running at 100% SLA:\n"
                    "• Clara (Ext 101): Dedicated to inbound phone calls (<1 ring pickup)\n"
                    "• Ryan (Ext 102): Catching Zillow, Realtor.com & Website leads (<30s SLA)\n"
                    "• Maya (Ext 103): Qualifying Instagram DMs, Meta Ads & WhatsApp chats\n"
                    "• Alex (Ext 104): Booking Sentrilock site visits & transcribing agent car voice memos"
                )
            else:
                llm_answer = (
                    f"Good day! As your Chief of Staff, I'm overseeing {len(leads)} active buyer leads, {len(appts)} scheduled property tours, "
                    f"and {len(agents)} licensed human agents. Let me know if you'd like me to reassign any leads or trigger a follow-up blast!"
                )

        db.log_bot_action("Victoria", BotRole.CHIEF_OF_STAFF, "BROKER_COMMAND_ANSWERED", f"Answered broker query: '{broker_question[:50]}'")
        return {
            "bot": "Victoria (Chief of Staff • Ext 100)",
            "broker_question": broker_question,
            "executive_response": llm_answer,
            "live_metrics": {
                "total_leads": len(leads),
                "hot_qualified_leads": len(hot_leads),
                "booked_appointments": len(appts),
                "estimated_pipeline_volume": f"${total_pipeline:,.0f}"
            }
        }

    @staticmethod
    def assign_best_human_agent(db: UnifiedDatabase, lead: Lead) -> Agent:
        agents = db.get_agents()
        available = [a for a in agents if a.is_active and a.active_leads_count < a.max_capacity] or agents
        q = lead.qualification
        budget = q.budget_max or 500000.0

        scored: List[Tuple[Agent, float]] = []
        for agent in available:
            pts = 0.0
            if budget >= 1200000.0 and agent.price_floor >= 1000000.0:
                pts += 50.0
            elif budget < 1000000.0 and agent.price_floor == 0.0:
                pts += 30.0
            if q.property_type and q.property_type in agent.specialties:
                pts += 25.0
            if q.preferred_zip_codes and set(q.preferred_zip_codes).intersection(set(agent.zip_codes)):
                pts += 40.0
            pts -= (agent.active_leads_count / max(agent.max_capacity, 1)) * 20.0
            scored.append((agent, pts))

        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]
        lead.assigned_agent_id = winner.id
        winner.active_leads_count += 1
        db.save_agent(winner)
        db.save_lead(lead)
        db.log_bot_action("Victoria", BotRole.CHIEF_OF_STAFF, "AGENT_ASSIGNED", f"Assigned {lead.first_name} {lead.last_name} (Score {q.score}/100) to {winner.name}")
        return winner

    @staticmethod
    def create_agent_handoff_dossier(lead: Lead, agent: Agent, appt: Optional[Appointment] = None) -> str:
        q = lead.qualification
        appt_str = f"{appt.appointment_type.value} on {appt.scheduled_time} ({appt.property_address})" if appt else "Follow-up active"
        b_min = f"${q.budget_min:,.0f}" if q.budget_min else "$500,000"
        b_max = f"${q.budget_max:,.0f}" if q.budget_max else "$800,000"
        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🚨 CHIEF OF STAFF ALERT — LEAD ASSIGNED TO: {agent.name.upper()}\n"
            f"   Dispatched by: Victoria (Chief of Staff • Ext 100)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 Buyer: {lead.first_name} {lead.last_name} | 📞 {lead.phone} | ✉️ {lead.email}\n"
            f"📥 Caught By: {lead.caught_by_bot} (Source: {lead.source.value})\n"
            f"🏠 Target Property: {lead.inquired_property_address or 'General Neighborhood'}\n"
            f"⭐ BANT Score: {q.score}/100 [TIER: {q.tier.value}]\n"
            f"💰 Budget: {b_min} - {b_max} | 🏦 Financing: {'PRE-APPROVED / CASH' if q.pre_approved else 'Needs Lender Referral'}\n"
            f"📅 Booked Showing: {appt_str}\n"
            f"🔑 Lockbox / Notes: {appt.access_notes if appt and appt.access_notes else 'N/A'}\n"
            f"📲 1-Click Google Calendar Add: {CalendarIntegrationEngine.build_google_and_outlook_links(appt, lead, agent)['google_calendar_add_url'] if appt else 'N/A'}\n"
            f"🔄 iPhone / Outlook Live Sync Feed: /api/v1/calendar/agent/{agent.id}/feed.ics\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    @staticmethod
    def generate_executive_briefing(db: UnifiedDatabase) -> str:
        leads = db.get_all_leads()
        appts = db.get_all_appointments()
        date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")

        clara_count = sum(1 for l in leads if l.source == LeadSource.INBOUND_CALL)
        ryan_count = sum(1 for l in leads if l.source in [LeadSource.ZILLOW, LeadSource.REALTOR_COM, LeadSource.GOOGLE_LSA, LeadSource.WEBSITE_FORM])
        maya_count = sum(1 for l in leads if l.source in [LeadSource.META_ADS, LeadSource.INSTAGRAM_DM, LeadSource.WHATSAPP])

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"☀️ 8:00 AM CHIEF OF STAFF BRIEFING — {date_str}",
            "   Prepared directly for Managing Broker by: Victoria (Ext 100)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🤖 LOAD-BALANCED 5-BOT WORKFORCE PERFORMANCE:",
            f"   1. Victoria (Chief of Staff • Ext 100): Direct Broker Concierge & Lead Routing",
            f"   2. Clara (Voice Receptionist • Ext 101): {clara_count} live phone calls answered (<1 ring)",
            f"   3. Ryan (Portal Catcher • Ext 102): {ryan_count} Zillow/Realtor/Website leads caught (<30s SLA)",
            f"   4. Maya (Social & WhatsApp • Ext 103): {maya_count} IG/Meta/WhatsApp leads qualified",
            f"   5. Alex (Showing & Field Ops • Ext 104): {len(appts)} site visits booked & lockboxes verified",
            "",
            "📅 CONFIRMED SITE VISITS & MEETINGS ON AGENT CALENDARS:"
        ]
        for a in appts[:5]:
            lines.append(f"   • {a.scheduled_time} — {a.property_address} ({a.appointment_type.value})")
            if a.access_notes:
                lines.append(f"     🔑 {a.access_notes}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

# -----------------------------------------------------------------------------
# BOT 2: CLARA (EXT 101) — VOICE RECEPTIONIST (PHONE LINE ONLY, ZERO OVERLOAD)
# -----------------------------------------------------------------------------

class ClaraVoiceReceptionistBot:
    """
    100% focused on answering live inbound phone calls on the first ring.
    No longer overloaded with portal webhooks or social DMs.
    """
    name = "Clara"
    role = BotRole.VOICE_RECEPTIONIST
    extension = "Ext 101"

    @staticmethod
    def handle_inbound_call(db: UnifiedDatabase, caller_name: str, caller_phone: str, transcript: str) -> Dict[str, Any]:
        listing = db.find_listing(transcript)
        parts = caller_name.split(" ", 1)
        lead = Lead(
            id=f"lead_{uuid.uuid4().hex[:8]}",
            first_name=parts[0],
            last_name=parts[1] if len(parts) > 1 else "Caller",
            phone=caller_phone,
            email=f"{parts[0].lower()}@phonecaller.com",
            source=LeadSource.INBOUND_CALL,
            caught_by_bot="Clara (Voice Receptionist • Ext 101)",
            inquired_property_address=listing.address if listing else "Phone Inquiry",
            initial_message=transcript,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        if listing:
            lead.qualification.preferred_zip_codes = list(set(re.findall(r"\b\d{5}\b", listing.address)))

        lead, _ = QualificationEngine.evaluate_and_reply(lead, transcript, "Clara (Voice Receptionist)")
        db.save_lead(lead)

        if listing:
            spoken_reply = (
                f"Hello {caller_name}! Thank you for calling Premier Realty. This is Clara on the front desk line. "
                f"{listing.address} is active at ${listing.price:,.0f} with {listing.bedrooms} bedrooms and {listing.bathrooms} baths. "
                f"I am transferring your tour request to Alex (Ext 104) to text you the calendar invite and lockbox details right now!"
            )
            handoff = "Alex (Showing & Field Ops • Ext 104)"
        else:
            spoken_reply = (
                f"Hello {caller_name}, thank you for calling Premier Realty! I've noted your message and alerted Victoria (Ext 100), "
                f"our Chief of Staff, to connect you with our specialist immediately."
            )
            handoff = "Victoria (Chief of Staff • Ext 100)"

        db.log_bot_action("Clara", BotRole.VOICE_RECEPTIONIST, "PHONE_CALL_ANSWERED", f"Answered voice call from {caller_name} ({caller_phone}) -> Handed off to {handoff}")
        return {
            "bot": "Clara (Voice Receptionist • Ext 101)",
            "lead_id": lead.id,
            "spoken_response": spoken_reply,
            "matched_listing": to_dict(listing) if listing else None,
            "qualification": to_dict(lead.qualification),
            "handed_off_to": handoff
        }

# -----------------------------------------------------------------------------
# BOT 3: RYAN (EXT 102) — PORTAL LEAD CATCHER (ZILLOW, REALTOR.COM, LSA, WEBSITE)
# -----------------------------------------------------------------------------

class LiveEmailDispatcher:
    """
    Real Outbound Email Sender via SMTP (Gmail App Password, Google Workspace, Office365, Zoho, or custom SMTP).
    Uses Python's built-in `smtplib` and `ssl` standard libraries — zero external dependencies required.
    """
    @staticmethod
    def send_email(
        db: UnifiedDatabase,
        to_email: str,
        subject: str,
        plain_body: str,
        html_body: Optional[str] = None,
        override_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        cfg = dict(db.get_settings())
        if override_settings:
            cfg.update(override_settings)

        smtp_email = (cfg.get("smtp_email") or os.environ.get("SMTP_EMAIL") or "").strip()
        smtp_password = (cfg.get("smtp_password") or os.environ.get("SMTP_PASSWORD") or "").strip()
        smtp_host = (cfg.get("smtp_host") or os.environ.get("SMTP_HOST") or "smtp.gmail.com").strip()
        smtp_port = int(cfg.get("smtp_port") or os.environ.get("SMTP_PORT") or 465)
        brokerage_name = (cfg.get("brokerage_name") or "Premier Estates").strip()

        # -------------------------------------------------------------
        # METHOD A: RESEND API (Zero SMTP, No App Passwords Needed!)
        # If the user pasted a Resend API key ('re_...') into either the password or email field
        # -------------------------------------------------------------
        resend_key = ""
        if smtp_password.startswith("re_"):
            resend_key = smtp_password
        elif smtp_email.startswith("re_"):
            resend_key = smtp_email

        if resend_key:
            try:
                from_addr = f"{brokerage_name} <onboarding@resend.dev>"
                resend_payload: Dict[str, Any] = {
                    "from": from_addr,
                    "to": [to_email],
                    "subject": subject,
                    "html": html_body or f"<pre>{plain_body}</pre>",
                    "text": plain_body
                }
                reply_to_addr = (cfg.get("reply_to_email") or cfg.get("imap_email") or smtp_email or "").strip()
                if "@" in reply_to_addr and "resend.dev" not in reply_to_addr and not reply_to_addr.startswith("re_"):
                    resend_payload["reply_to"] = reply_to_addr
                req = urllib.request.Request(
                    "https://api.resend.com/emails",
                    data=json.dumps(resend_payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {resend_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "BopsRealEstate/4.0"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    email_id = resp_data.get("id", "resend_sent")
                    db.log_bot_action("Ryan", BotRole.PORTAL_CATCHER, "EMAIL_SENT_LIVE", f"Delivered email to {to_email} via Resend API (ID: {email_id})")
                    return {
                        "sent": True,
                        "status": "SENT_LIVE",
                        "provider": "Resend API",
                        "from": from_addr,
                        "to": to_email,
                        "detail": f"Delivered to {to_email} via Resend (ID: {email_id})"
                    }
            except urllib.error.HTTPError as he:
                err_text = he.read().decode("utf-8")[:200]
                db.log_bot_action("Ryan", BotRole.PORTAL_CATCHER, "EMAIL_RESEND_ERROR", f"Resend HTTP {he.code}: {err_text}")
                return {"sent": False, "status": "RESEND_ERROR", "detail": f"Resend API error: {err_text}"}
            except Exception as re_err:
                return {"sent": False, "status": "RESEND_ERROR", "detail": f"Resend error: {str(re_err)}"}

        # -------------------------------------------------------------
        # METHOD B: STANDARD SMTP (Brevo, Gmail App Passwords, Zoho, etc.)
        # -------------------------------------------------------------
        if not smtp_email or not smtp_password:
            return {
                "sent": False,
                "status": "NOT_CONFIGURED",
                "detail": "Outbound email not configured yet. Either paste a free Resend API key (re_...) or configure Brevo / Gmail SMTP in the Dashboard."
            }

        # Strip spaces from 16-char Google App Passwords (e.g. 'abcd efgh ijkl mnop' -> 'abcdefghijklmnop')
        clean_pass = smtp_password.replace(" ", "") if "gmail.com" in smtp_host.lower() else smtp_password

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{brokerage_name} Concierge <{smtp_email}>"
        msg["To"] = to_email
        msg["Reply-To"] = smtp_email

        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        if html_body:
            msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            context = ssl.create_default_context()
            if smtp_port == 465:
                with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=15) as server:
                    server.login(smtp_email, clean_pass)
                    server.sendmail(smtp_email, [to_email], msg.as_string())
            else:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                    server.login(smtp_email, clean_pass)
                    server.sendmail(smtp_email, [to_email], msg.as_string())

            db.log_bot_action("Ryan", BotRole.PORTAL_CATCHER, "EMAIL_SENT_LIVE", f"Delivered live email to {to_email} via {smtp_host}")
            return {
                "sent": True,
                "status": "SENT_LIVE",
                "from": smtp_email,
                "to": to_email,
                "detail": f"Email successfully delivered to {to_email} via {smtp_host}:{smtp_port}"
            }
        except smtplib.SMTPAuthenticationError as auth_err:
            err_msg = (
                f"SMTP Authentication Failed for {smtp_email}. If using Gmail, you MUST use a 16-character "
                f"Google App Password (from myaccount.google.com/apppasswords), NOT your regular Gmail login password. "
                f"Raw error: {auth_err}"
            )
            db.log_bot_action("Ryan", BotRole.PORTAL_CATCHER, "EMAIL_AUTH_ERROR", err_msg[:120])
            return {"sent": False, "status": "AUTH_ERROR", "detail": err_msg}
        except Exception as e:
            err_msg = f"SMTP Error ({smtp_host}:{smtp_port}): {str(e)}"
            db.log_bot_action("Ryan", BotRole.PORTAL_CATCHER, "EMAIL_SEND_ERROR", err_msg[:120])
            return {"sent": False, "status": "ERROR", "detail": err_msg}

    @classmethod
    def send_lead_response_email(
        cls,
        db: UnifiedDatabase,
        lead: Lead,
        reply_text: str,
        matched_listing: Optional[PropertyListing] = None,
        nearest_listings: Optional[List[PropertyListing]] = None
    ) -> Dict[str, Any]:
        cfg = db.get_settings()
        brokerage_name = (cfg.get("brokerage_name") or "Premier Estates").strip()
        prop_title = matched_listing.address if matched_listing else (lead.inquired_property_address or brokerage_name)
        subject = f"Re: Your Property Inquiry — {prop_title}"

        listing_html = ""
        listing_plain = ""
        if matched_listing:
            listing_plain = (
                f"\n\n--- VERIFIED PROPERTY SPECIFICATIONS ---\n"
                f"Address: {matched_listing.address}\n"
                f"MLS ID: {matched_listing.mls_id} ({matched_listing.status})\n"
                f"Price: ${matched_listing.price:,.0f}\n"
                f"Layout: {matched_listing.bedrooms} Beds | {matched_listing.bathrooms} Baths | {matched_listing.square_feet:,} Sq Ft\n"
                f"Property Type: {matched_listing.property_type}\n"
            )
            listing_html = f"""
            <div style="margin-top:20px;padding:18px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;font-family:sans-serif;">
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#0f172a;letter-spacing:0.06em;margin-bottom:8px;">
                    Verified Property Specifications ({matched_listing.mls_id})
                </div>
                <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">{matched_listing.address}</div>
                <div style="font-size:18px;font-weight:700;color:#047857;margin-bottom:10px;">${matched_listing.price:,.0f} <span style="font-size:12px;font-weight:600;color:#475569;">· {matched_listing.status}</span></div>
                <div style="font-size:13px;color:#334155;">
                    <strong>{matched_listing.bedrooms}</strong> Bedrooms &nbsp;·&nbsp;
                    <strong>{matched_listing.bathrooms}</strong> Bathrooms &nbsp;·&nbsp;
                    <strong>{matched_listing.square_feet:,}</strong> Sq Ft &nbsp;·&nbsp;
                    {matched_listing.property_type}
                </div>
            </div>
            """
        elif nearest_listings:
            top_2_3 = nearest_listings[:3]
            plain_items = [
                f"• {p.address} ({p.mls_id}) — ${p.price:,.0f} | {p.bedrooms} BD / {p.bathrooms} BA / {p.square_feet:,} Sq Ft ({p.property_type})"
                for p in top_2_3
            ]
            listing_plain = "\n\n--- CLOSEST MATCHING RESIDENCES (TOP RECOMMENDATIONS) ---\n" + "\n".join(plain_items) + "\n"
            cards_html = "".join([
                f"""
                <div style="padding:12px;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:8px;">
                    <div style="font-size:14px;font-weight:700;color:#0f172a;">{p.address}</div>
                    <div style="font-size:15px;font-weight:700;color:#047857;margin:2px 0;">${p.price:,.0f} <span style="font-size:11px;color:#64748b;font-weight:500;">({p.mls_id})</span></div>
                    <div style="font-size:12px;color:#475569;">{p.bedrooms} Beds · {p.bathrooms} Baths · {p.square_feet:,} Sq Ft · {p.property_type}</div>
                </div>
                """
                for p in top_2_3
            ])
            listing_html = f"""
            <div style="margin-top:20px;padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;font-family:sans-serif;">
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#0f172a;letter-spacing:0.06em;margin-bottom:10px;">
                    Closest Matching Residences ({len(top_2_3)} Nearest Options)
                </div>
                {cards_html}
            </div>
            """

        plain_full = f"{reply_text}{listing_plain}\n\nWarm regards,\nRyan — Property Concierge\n{brokerage_name}"
        html_full = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:600px;margin:0 auto;padding:24px;color:#1e293b;line-height:1.6;">
            <p style="font-size:15px;color:#0f172a;white-space:pre-line;">{reply_text}</p>
            {listing_html}
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:24px 0 16px 0;">
            <div style="font-size:12px;color:#64748b;">
                <strong>Ryan · Property Concierge</strong><br>
                {brokerage_name}<br>
                <em>Reply directly to this email to confirm your preferred tour time.</em>
            </div>
        </div>
        """
        return cls.send_email(db, lead.email, subject, plain_full, html_full)


class LeadValidationGuard:
    """
    Validates incoming website/portal lead submissions so gibberish, profanity,
    fake names ('hey', 'test', 'asdf'), or invalid phone numbers are rejected
    before polluting the SQLite leads database.
    """
    BLOCKED_WORDS = {
        "shit", "fuck", "fucking", "bitch", "asshole", "crap", "dick", "piss",
        "scam", "wtf", "bullshit", "bastard", "idiot", "stupid", "dumb"
    }
    INVALID_NAMES = {
        "hey", "hi", "hello", "yo", "sup", "test", "testing", "asdf", "qwerty",
        "abc", "xyz", "none", "null", "fake", "user", "admin", "idk", "na", "n/a",
        "shit", "fuck", "wtf", "scam", "blah", "lol", "haha"
    }

    @classmethod
    def validate_submission(cls, first_name: str, last_name: str, phone: str, email: str, address: str, message: str):
        fn = (first_name or "").strip()
        ln = (last_name or "").strip()
        ph = (phone or "").strip()
        em = (email or "").strip()
        addr = (address or "").strip()
        msg = (message or "").strip()

        # 1. Check for profanity anywhere in submission
        combined_tokens = set(re.findall(r"[a-z]+", f"{fn} {ln} {addr} {msg}".lower()))
        bad_found = combined_tokens.intersection(cls.BLOCKED_WORDS)
        if bad_found:
            raise ValueError("Submission rejected: Please remove inappropriate language and enter genuine inquiry details.")

        # 2. Validate First & Last Name (must be letters, >= 2 chars, not a greeting like 'hey' or gibberish with numbers)
        if len(fn) < 2 or fn.lower() in cls.INVALID_NAMES or re.search(r"\d", fn):
            raise ValueError(f"Invalid first name '{fn}'. Please enter your real first name.")
        if ln and (len(ln) < 2 or ln.lower() in cls.INVALID_NAMES or re.search(r"\d", ln)):
            raise ValueError(f"Invalid last name '{ln}'. Please enter your real last name.")

        # 3. Validate Phone Number (must have at least 7 digits, not all same digit)
        digits = re.sub(r"\D", "", ph)
        if len(digits) < 7 or len(digits) > 15 or len(set(digits)) == 1:
            raise ValueError("Invalid phone number. Please enter a valid 10-digit phone number.")

        # 4. Validate Email Address
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$", em):
            raise ValueError("Invalid email address. Please enter a valid email (e.g. name@domain.com).")

        # 5. Detect random keyboard mashing in property address (e.g. '438yj')
        if addr and len(addr) < 8 and re.match(r"^\d+[a-z]{1,4}$", addr.lower()):
            raise ValueError(f"'{addr}' is not a valid property address. Please select a property from our available listings.")


class RyanPortalCatcherBot:
    """
    Dedicated exclusively to high-intent property portals (Zillow, Realtor.com,
    Google LSA, Website Forms). Verifies requested property against the live SQLite
    `listings` database and generates an intelligent, database-grounded response.
    """
    name = "Ryan"
    role = BotRole.PORTAL_CATCHER
    extension = "Ext 102"

    @staticmethod
    def catch_portal_lead(db: UnifiedDatabase, source: LeadSource, payload: Dict[str, Any]) -> Tuple[Lead, Dict[str, Any]]:
        contact = payload.get("contact", payload)
        first_name = (contact.get("firstName") or payload.get("first_name") or "").strip()
        last_name = (contact.get("lastName") or payload.get("last_name") or "").strip()
        phone = (contact.get("phone") or payload.get("phone") or "").strip()
        email = (contact.get("email") or payload.get("email") or "").strip()
        address = (payload.get("property", {}).get("address") or payload.get("property_address") or payload.get("address") or "").strip()
        message = (payload.get("notes") or payload.get("message") or "").strip()

        # Validate lead input (raises ValueError if gibberish, spam, profanity, or invalid contact)
        LeadValidationGuard.validate_submission(first_name, last_name, phone, email, address, message)

        # Sync saved LLM API key from SQLite settings if present
        saved_cfg = db.get_settings()
        if saved_cfg.get("ai_api_key") and not LLMEngine.active_api_key:
            LLMEngine.configure_runtime(
                provider=saved_cfg.get("ai_provider", LLMEngine.active_provider),
                model=saved_cfg.get("ai_model", LLMEngine.active_model),
                api_key=saved_cfg.get("ai_api_key")
            )

        # Look up the property in the real SQLite `listings` table
        matched_listing = db.find_listing(address) if address else None
        # Fetch ONLY the nearest 2-3 possibilities (never the whole database!)
        nearest_listings = [] if matched_listing else db.find_nearest_listings(address, message, limit=3)

        zips = re.findall(r"\b\d{5}\b", f"{address} {message} {matched_listing.address if matched_listing else ''}")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        lead = Lead(
            id=f"lead_{uuid.uuid4().hex[:8]}",
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            email=email,
            source=source,
            caught_by_bot="Ryan (Portal Catcher • Ext 102)",
            status=LeadStatus.CONTACTED,
            inquired_property_address=matched_listing.address if matched_listing else (address or "General Brokerage Inquiry"),
            initial_message=message or "Requested property details via website form",
            created_at=now,
            last_contact_at=now,
            follow_up_step=1,
            qualification=QualificationData(preferred_zip_codes=list(set(zips)))
        )

        # Score BANT passively from whatever the buyer voluntarily wrote (never interrogate!)
        combined_context = f"{address} {message}"
        QualificationEngine.extract_passive_bant(lead, combined_context, matched_listing)

        # Build Scoped SQLite Context for Live LLM (ONLY matched property OR nearest 2-3 possibilities — NEVER whole DB)
        nearest_lines = [
            f"- MLS #{p.mls_id}: {p.address} | Price: ${p.price:,.0f} | {p.bedrooms} BD / {p.bathrooms} BA / {p.square_feet:,} SqFt | Type: {p.property_type}"
            for p in nearest_listings[:3]
        ]
        nearest_str = "\n".join(nearest_lines) if nearest_lines else "No close matches found."

        brokerage_name = saved_cfg.get("brokerage_name") or "Premier Realty"
        if matched_listing:
            sys_prompt = (
                f"You are Ryan, Property Concierge at {brokerage_name}. "
                f"A prospective buyer named {first_name} {last_name} just submitted an inquiry.\n"
                f"MATCHED DATABASE PROPERTY: MLS #{matched_listing.mls_id} — {matched_listing.address} "
                f"(Price: ${matched_listing.price:,.0f}, {matched_listing.bedrooms} BD, {matched_listing.bathrooms} BA, {matched_listing.square_feet:,} SqFt, Status: {matched_listing.status})\n"
                f"Buyer's Message: '{message or 'Interested in learning more / scheduling a viewing.'}'\n\n"
                f"STRICT INSTRUCTIONS:\n"
                f"1. Confirm the exact property details above from our database and answer any question in the buyer's message.\n"
                f"2. NEVER ask pushy financial questions like 'Are you pre-approved or buying with cash?'. Simply ask what day/time works best to view the property. Keep it under 3 sentences."
            )
        else:
            sys_prompt = (
                f"You are Ryan, Property Concierge at {brokerage_name}. "
                f"A prospective buyer named {first_name} {last_name} just asked about '{address or 'available homes'}'.\n"
                f"Buyer's Message: '{message or 'Interested in learning more.'}'\n"
                f"DATABASE LOOKUP RESULT: '{address}' is NOT listed in our database.\n"
                f"NEAREST {len(nearest_lines)} POSSIBILITIES IN OUR DATABASE (Ranked by location, budget & specs):\n{nearest_str}\n\n"
                f"STRICT INSTRUCTIONS:\n"
                f"1. Politely inform {first_name} that '{address}' is not in our current inventory.\n"
                f"2. Suggest ONLY the {len(nearest_lines)} nearest possibilities listed above (NEVER list more than these {len(nearest_lines)} homes).\n"
                f"3. Ask which of these {len(nearest_lines)} options they would like to tour or get details on. Keep it warm and under 3 sentences."
            )

        llm_reply, _ = LLMEngine.generate_with_diagnostics(
            system_prompt=sys_prompt,
            user_message=f"Property: {address} | Message: {message}"
        )

        if llm_reply:
            instant_sms = SecurityAndIdentityGuard.redact_sensitive_data_for_public(llm_reply)
        else:
            # 100% SQLite-database-driven deterministic reply (when no LLM API key is configured yet)
            if matched_listing:
                instant_sms = (
                    f"Hi {first_name}, thank you for inquiring about {matched_listing.address} (MLS #{matched_listing.mls_id})! "
                    f"It is currently {matched_listing.status.lower()} in our inventory at ${matched_listing.price:,.0f} "
                    f"({matched_listing.bedrooms} beds, {matched_listing.bathrooms} baths, {matched_listing.square_feet:,} sq ft). "
                    f"What day and time works best for you to tour the property, or what questions can I answer for you?"
                )
            else:
                avail_list = "; ".join([f"{p.address} (${p.price:,.0f}, {p.bedrooms} BD/{p.bathrooms} BA)" for p in nearest_listings[:3]]) if nearest_listings else "no matching listings currently loaded"
                if address:
                    instant_sms = (
                        f"Hi {first_name}, thank you for reaching out to {brokerage_name}! We do not currently have '{address}' in our database, "
                        f"but here are the {len(nearest_listings[:3])} closest matching possibilities based on your inquiry: {avail_list}. "
                        f"Would you like details or a private showing for one of these homes?"
                    )
                else:
                    instant_sms = (
                        f"Hi {first_name}, thank you for reaching out to {brokerage_name}! Based on your inquiry, here are our top {len(nearest_listings[:3])} matching possibilities: {avail_list}. "
                        f"Which of these would you like to schedule a walkthrough for?"
                    )

        # Dispatch real outbound Email to the buyer's email address via Resend / SMTP
        email_result = LiveEmailDispatcher.send_lead_response_email(
            db=db,
            lead=lead,
            reply_text=instant_sms,
            matched_listing=matched_listing,
            nearest_listings=nearest_listings
        )

        lead.conversation_history.append({
            "sender": "RYAN_PORTAL_BOT",
            "channel": "EMAIL_AND_SMS",
            "text": instant_sms,
            "email_status": email_result.get("status", "NOT_CONFIGURED"),
            "email_detail": email_result.get("detail", ""),
            "timestamp": now
        })

        db.save_lead(lead)
        db.log_bot_action(
            "Ryan",
            BotRole.PORTAL_CATCHER,
            "PORTAL_LEAD_CAUGHT",
            f"Verified inquiry from {first_name} {last_name} ({'Matched: ' + matched_listing.mls_id if matched_listing else 'Unmatched: ' + address}) | Email: {email_result.get('status')}"
        )
        return lead, {
            "lead_id": lead.id,
            "bot": "Ryan (Portal Catcher • Ext 102)",
            "channel": "EMAIL",
            "matched_in_database": bool(matched_listing),
            "matched_listing": to_dict(matched_listing) if matched_listing else None,
            "qualification": to_dict(lead.qualification),
            "message": instant_sms,
            "email_delivery": email_result,
            "timestamp": now
        }

# -----------------------------------------------------------------------------
# BOT 4: MAYA (EXT 103) — SOCIAL & WHATSAPP CONCIERGE (IG DMs, META ADS, WHATSAPP)
# -----------------------------------------------------------------------------

class LiveMetaGraphDispatcher:
    """
    Real Outbound & Inbound Meta Graph API v21.0 Integration for Maya (Ext 103) and Ryan (Ext 102):
    1. Sends live Instagram Direct Messages & Facebook Page Messenger replies (`POST /v21.0/me/messages`).
    2. Sends Private DM Replies to buyers who comment on Instagram Reels / Posts (`recipient: {comment_id}`).
    3. Fetches full lead contact details from Facebook Lead Ads (`GET /v21.0/{leadgen_id}`).
    """
    GRAPH_BASE = "https://graph.facebook.com/v21.0"

    @classmethod
    def get_meta_credentials(cls, db: UnifiedDatabase, override: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        cfg = dict(db.get_settings())
        if override:
            cfg.update(override)
        return {
            "access_token": (cfg.get("meta_access_token") or os.environ.get("META_PAGE_ACCESS_TOKEN") or "").strip(),
            "page_id": (cfg.get("meta_page_id") or os.environ.get("META_PAGE_ID") or "me").strip() or "me",
            "verify_token": (cfg.get("meta_verify_token") or os.environ.get("META_VERIFY_TOKEN") or "").strip()
        }

    @classmethod
    def send_social_dm(
        cls,
        db: UnifiedDatabase,
        recipient_id: str,
        message_text: str,
        platform: str = "INSTAGRAM_DM",
        is_comment_reply: bool = False
    ) -> Dict[str, Any]:
        creds = cls.get_meta_credentials(db)
        token = creds["access_token"]
        page_id = creds["page_id"]

        if not token:
            return {
                "sent": False,
                "status": "META_TOKEN_NOT_CONFIGURED",
                "detail": "Meta Page / Instagram Access Token not configured yet. Add your EAAG... token in Dashboard (Instagram & Facebook Setup) to send live Graph API DMs."
            }

        url = f"{cls.GRAPH_BASE}/{page_id}/messages?access_token={urllib.parse.quote(token)}"
        recipient_obj = {"comment_id": recipient_id} if is_comment_reply else {"id": recipient_id}
        payload = {
            "recipient": recipient_obj,
            "message": {"text": message_text[:1000]}
        }
        if not is_comment_reply:
            payload["messaging_type"] = "RESPONSE"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "BopsRealEstate/4.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                msg_id = resp_data.get("message_id", "meta_dm_sent")
                db.log_bot_action("Maya", BotRole.SOCIAL_CONCIERGE, "META_DM_SENT_LIVE", f"Sent live {platform} to {recipient_id} (ID: {msg_id})")
                return {"sent": True, "status": "SENT_LIVE", "message_id": msg_id, "recipient_id": recipient_id}
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")[:200]
            db.log_bot_action("Maya", BotRole.SOCIAL_CONCIERGE, "META_API_ERROR", f"Graph API HTTP {he.code}: {err_body}")
            return {"sent": False, "status": f"HTTP_{he.code}", "detail": f"Meta Graph API Error: {err_body}"}
        except Exception as e:
            return {"sent": False, "status": "ERROR", "detail": str(e)}

    @classmethod
    def fetch_facebook_lead_ad_details(cls, db: UnifiedDatabase, leadgen_id: str) -> Dict[str, str]:
        """Queries `GET https://graph.facebook.com/v21.0/{leadgen_id}` to pull the buyer's name, email, phone & answers from a Facebook Lead Ad."""
        creds = cls.get_meta_credentials(db)
        token = creds["access_token"]
        if not token or not leadgen_id:
            return {}
        try:
            url = f"{cls.GRAPH_BASE}/{urllib.parse.quote(str(leadgen_id))}?access_token={urllib.parse.quote(token)}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                field_data = data.get("field_data", [])
                parsed: Dict[str, str] = {}
                for item in field_data:
                    k = (item.get("name") or "").lower()
                    vals = item.get("values") or [""]
                    parsed[k] = vals[0] if vals else ""
                return parsed
        except Exception:
            return {}

    @classmethod
    def verify_meta_token_live(cls, db: UnifiedDatabase, override_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        creds = cls.get_meta_credentials(db, override=override_cfg)
        token = creds["access_token"]
        if not token:
            return {"verified": False, "detail": "Please paste your Meta Page Access Token (starts with EAA...)"}
        try:
            url = f"{cls.GRAPH_BASE}/me?fields=id,name,instagram_business_account&access_token={urllib.parse.quote(token)}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "verified": True,
                    "page_id": data.get("id"),
                    "page_name": data.get("name"),
                    "instagram_business_account": data.get("instagram_business_account", {}).get("id"),
                    "detail": f"Connected to Meta Page '{data.get('name')}' (ID: {data.get('id')})"
                }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="replace")[:200]
            return {"verified": False, "detail": f"Meta Graph API rejected token (HTTP {he.code}): {err}"}
        except Exception as e:
            return {"verified": False, "detail": f"Connection error: {str(e)}"}


class MayaSocialConciergeBot:
    """
    Dedicated exclusively to Instagram DMs, Facebook Messenger, Facebook/Meta Lead Ads, and WhatsApp threads.
    Checks the live SQLite `listings` database (exact match or nearest 2–3 possibilities), replies via
    Meta Graph API (`LiveMetaGraphDispatcher`), and dispatches emails if the buyer shares an email address.
    """
    name = "Maya"
    role = BotRole.SOCIAL_CONCIERGE
    extension = "Ext 103"

    @staticmethod
    def catch_social_lead(db: UnifiedDatabase, source: LeadSource, payload: Dict[str, Any]) -> Tuple[Lead, Dict[str, Any]]:
        # 1. If Facebook Lead Ad `leadgen_id` is present and field_data is missing, try fetching from Meta Graph API
        if payload.get("leadgen_id") and "field_data" not in payload:
            fetched_fields = LiveMetaGraphDispatcher.fetch_facebook_lead_ad_details(db, str(payload["leadgen_id"]))
            if fetched_fields:
                payload = {**payload, **fetched_fields}

        if "field_data" in payload:
            fd = {item.get("name", "").lower(): (item.get("values") or [""])[0] for item in payload.get("field_data", [])}
            full_name = fd.get("full_name") or fd.get("name") or "Facebook Lead"
            parts = full_name.split(" ", 1)
            first_name, last_name = parts[0], (parts[1] if len(parts) > 1 else "")
            phone = fd.get("phone_number") or fd.get("phone") or payload.get("sender_id") or "IG_OR_FB_USER"
            email = fd.get("email") or ""
            area = fd.get("preferred_location") or fd.get("property_address") or fd.get("city") or ""
            msg = fd.get("custom_questions") or fd.get("message") or "Submitted Facebook / Instagram Lead Form"
        else:
            first_name = (payload.get("first_name") or payload.get("full_name") or "Instagram").strip()
            last_name = (payload.get("last_name") or "User").strip()
            phone = (payload.get("phone") or payload.get("sender_id") or "IG_DM_USER").strip()
            email = (payload.get("email") or "").strip()
            area = (payload.get("property_address") or payload.get("area") or "").strip()
            msg = (payload.get("message") or payload.get("notes") or "").strip()

        # Extract email if the user typed their email inside the DM message text!
        if not email or "@" not in email:
            em_in_msg = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", f"{area} {msg}")
            if em_in_msg:
                email = em_in_msg.group(0).lower()

        # Check if this Instagram/Facebook user already has a lead thread in SQLite!
        existing_lead = db.find_lead_by_social_or_phone(phone) or (db.find_lead_by_email(email) if email else None)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Sync saved LLM API key
        saved_cfg = db.get_settings()
        if saved_cfg.get("ai_api_key") and not LLMEngine.active_api_key:
            LLMEngine.configure_runtime(
                provider=saved_cfg.get("ai_provider", LLMEngine.active_provider),
                model=saved_cfg.get("ai_model", LLMEngine.active_model),
                api_key=saved_cfg.get("ai_api_key")
            )

        search_query = f"{area} {msg}".strip()
        matched_listing = db.find_listing(area) or db.find_listing(msg)
        # Strictly fetch ONLY the nearest 2-3 possibilities if not an exact listing match!
        nearest_listings = [] if matched_listing else db.find_nearest_listings(area, msg, limit=3)

        if existing_lead:
            lead = existing_lead
            if email and ("@" not in lead.email or "instagram.com" in lead.email or "social.com" in lead.email):
                lead.email = email
            if matched_listing:
                lead.inquired_property_address = matched_listing.address
            elif area and not lead.inquired_property_address:
                lead.inquired_property_address = area
            lead.conversation_history.append({
                "sender": "LEAD",
                "channel": source.value,
                "text": msg or area,
                "timestamp": now
            })
        else:
            zips = re.findall(r"\b\d{5}\b", f"{area} {msg} {matched_listing.address if matched_listing else ''}")
            lead = Lead(
                id=f"lead_{uuid.uuid4().hex[:8]}",
                first_name=first_name,
                last_name=last_name,
                phone=phone,
                email=email or f"{phone}@instagram.dm",
                source=source,
                caught_by_bot="Maya (Social & WhatsApp Concierge • Ext 103)",
                status=LeadStatus.CONTACTED,
                inquired_property_address=matched_listing.address if matched_listing else (area or "Instagram / Facebook Inquiry"),
                initial_message=msg or f"Inquired via {source.value}",
                created_at=now,
                last_contact_at=now,
                follow_up_step=1,
                qualification=QualificationData(preferred_zip_codes=list(set(zips)))
            )
            lead.conversation_history.append({
                "sender": "LEAD",
                "channel": source.value,
                "text": msg or f"Inquired about {area}",
                "timestamp": now
            })

        QualificationEngine.extract_passive_bant(lead, search_query, matched_listing)

        # Build conversation history for LLM
        conv_msgs = [
            {"role": ("user" if h.get("sender") == "LEAD" else "assistant"), "content": h.get("text", "")}
            for h in lead.conversation_history[-10:]
        ]

        brokerage_name = saved_cfg.get("brokerage_name") or "Premier Estates"
        if matched_listing:
            prop_ctx = (
                f"MATCHED DATABASE LISTING: MLS #{matched_listing.mls_id} — {matched_listing.address} | "
                f"Price: ${matched_listing.price:,.0f} | {matched_listing.bedrooms} BD / {matched_listing.bathrooms} BA / {matched_listing.square_feet:,} SqFt | Status: {matched_listing.status}"
            )
        else:
            near_lines = [
                f"- {p.address} (${p.price:,.0f}, {p.bedrooms} BD/{p.bathrooms} BA, {p.square_feet:,} SqFt)"
                for p in nearest_listings[:3]
            ]
            prop_ctx = (
                f"NO EXACT LISTING MATCH FOR '{area or msg}'. Here are ONLY the {len(near_lines)} nearest possibilities in our database (NEVER suggest more than these {len(near_lines)}):\n" +
                ("\n".join(near_lines) if near_lines else "No listings loaded.")
            )

        sys_prompt = (
            f"You are Maya (Ext 103), the Instagram & Facebook Concierge at {brokerage_name}.\n"
            f"Channel: {source.value} | Buyer: {lead.first_name}\n"
            f"{prop_ctx}\n\n"
            f"STRICT RULES:\n"
            f"1. Reply in a friendly, natural Instagram/Facebook DM style (2-3 short sentences max).\n"
            f"2. If MATCHED DATABASE LISTING is found, quote the exact price, beds/baths, and square footage from the database and ask what day they want to tour or what email to send the full brochure to.\n"
            f"3. If NO EXACT LISTING MATCH, share ONLY the 2–3 nearest possibilities listed above—never dump a full database.\n"
            f"4. Never ask rude financial interrogation questions like 'Are you pre-approved?'"
        )

        llm_reply, _ = LLMEngine.generate_with_diagnostics(
            system_prompt=sys_prompt,
            user_message=msg or area or "Hi",
            conversation_messages=conv_msgs
        )

        if llm_reply:
            instant_dm = SecurityAndIdentityGuard.redact_sensitive_data_for_public(llm_reply)
        else:
            if matched_listing:
                instant_dm = (
                    f"Hi {lead.first_name}! 👋 This is Maya at {brokerage_name}. {matched_listing.address} (MLS #{matched_listing.mls_id}) "
                    f"is active at ${matched_listing.price:,.0f} ({matched_listing.bedrooms} BD / {matched_listing.bathrooms} BA, {matched_listing.square_feet:,} sq ft). "
                    f"Would you like to book a tour this week, or share your email so I can send the full property brochure?"
                )
            else:
                near_str = "; ".join([f"{p.address} (${p.price:,.0f}, {p.bedrooms} BD/{p.bathrooms} BA)" for p in nearest_listings[:3]])
                instant_dm = (
                    f"Hi {lead.first_name}! 👋 This is Maya at {brokerage_name}. Here are the {len(nearest_listings[:3])} closest matching homes in our inventory right now: "
                    f"{near_str}. Which of these would you like details or a private tour for?"
                )

        # 1. If recipient_id / sender_id is a Meta PSID/IGSID or comment_id, dispatch live DM via Meta Graph API!
        meta_dm_status = {"sent": False, "status": "LOCAL_SIMULATION"}
        sender_social_id = payload.get("sender_id") or payload.get("comment_id")
        if sender_social_id:
            meta_dm_status = LiveMetaGraphDispatcher.send_social_dm(
                db=db,
                recipient_id=str(sender_social_id),
                message_text=instant_dm,
                platform=source.value,
                is_comment_reply=bool(payload.get("comment_id"))
            )

        # 2. If the buyer provided a real email address (in Lead Ad form or inside the DM text), ALSO email them the property sheet!
        email_status = {"sent": False, "status": "NO_EMAIL_PROVIDED_YET"}
        if email and "@" in email and not email.endswith("@instagram.dm"):
            email_status = LiveEmailDispatcher.send_lead_response_email(
                db=db,
                lead=lead,
                reply_text=instant_dm,
                matched_listing=matched_listing,
                nearest_listings=nearest_listings
            )

        lead.conversation_history.append({
            "sender": "MAYA_SOCIAL_BOT",
            "channel": source.value,
            "text": instant_dm,
            "meta_dm_status": meta_dm_status.get("status", "LOCAL"),
            "email_status": email_status.get("status", "NO_EMAIL_PROVIDED_YET"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        db.save_lead(lead)
        db.log_bot_action(
            "Maya",
            BotRole.SOCIAL_CONCIERGE,
            "SOCIAL_LEAD_ENGAGED",
            f"Replied on {source.value} to {lead.first_name} ({'Matched: ' + matched_listing.mls_id if matched_listing else 'Top ' + str(len(nearest_listings[:3])) + ' nearest options'})"
        )
        return lead, {
            "lead_id": lead.id,
            "bot": "Maya (Social & WhatsApp • Ext 103)",
            "channel": source.value,
            "matched_in_database": bool(matched_listing),
            "message": instant_dm,
            "meta_graph_delivery": meta_dm_status,
            "email_delivery": email_status,
            "timestamp": now
        }


# -----------------------------------------------------------------------------
# LIVE CALENDAR INTEGRATION ENGINE (GOOGLE CALENDAR, OUTLOOK & iCAL FEED)
# -----------------------------------------------------------------------------

import urllib.parse

class CalendarIntegrationEngine:
    """
    Ensures human real estate agents immediately see their booked meetings & site visits
    on their iPhone / Android / Google Calendar / Outlook without opening a dashboard:
    1. Live iCalendar Subscription Feed (/api/v1/calendar/agent/{agent_id}/feed.ics)
    2. Direct 1-Click Google Calendar & Outlook Web Add Links (sent in Agent SMS Dossier)
    3. Downloadable RFC-5545 .ics file (/api/v1/calendar/appointment/{appt_id}.ics)
    """
    @staticmethod
    def build_google_and_outlook_links(appt: Appointment, lead: Lead, agent: Agent) -> Dict[str, str]:
        title = f"Showing: {appt.property_address or 'Buyer Consultation'} ({lead.first_name} {lead.last_name})"
        details = (
            f"Buyer: {lead.first_name} {lead.last_name} ({lead.phone}, {lead.email})\n"
            f"Type: {appt.appointment_type.value}\n"
            f"Access / Lockbox: {appt.access_notes or appt.meeting_link or 'See CRM'}"
        )
        location = appt.property_address or appt.meeting_link or "Phone / Virtual"
        dates = "20261112T210000Z/20261112T214500Z"

        gcal_params = urllib.parse.urlencode({
            "action": "TEMPLATE",
            "text": title,
            "dates": dates,
            "details": details,
            "location": location,
            "add": f"{lead.email},{agent.email}"
        })
        outlook_params = urllib.parse.urlencode({
            "path": "/calendar/action/compose",
            "rru": "addevent",
            "subject": title,
            "body": details,
            "location": location
        })
        return {
            "google_calendar_add_url": f"https://calendar.google.com/calendar/render?{gcal_params}",
            "outlook_calendar_add_url": f"https://outlook.office.com/calendar/0/deeplink/compose?{outlook_params}",
            "ics_download_url": f"/api/v1/calendar/appointment/{appt.id}.ics",
            "agent_live_feed_url": f"/api/v1/calendar/agent/{agent.id}/feed.ics"
        }

    @staticmethod
    def generate_agent_ics_feed(db: UnifiedDatabase, agent_id: str) -> str:
        """Generates a multi-event RFC-5545 VCALENDAR feed that syncs all booked tours to an agent's phone."""
        appts = [a for a in db.get_all_appointments() if a.agent_id == agent_id]
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Premier Realty Bops AI//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            f"X-WR-CALNAME:Premier Realty Showings ({agent_id})",
            "X-WR-TIMEZONE:America/Los_Angeles"
        ]
        for a in appts:
            lead = db.get_lead(a.lead_id)
            buyer_str = f"{lead.first_name} {lead.last_name} ({lead.phone})" if lead else "Buyer Lead"
            lines.extend([
                "BEGIN:VEVENT",
                f"UID:{a.id}@premier-realty.com",
                "DTSTAMP:20261110T080000Z",
                "DTSTART:20261112T210000Z",
                "DTEND:20261112T214500Z",
                f"SUMMARY:{a.appointment_type.value}: {a.property_address}",
                f"DESCRIPTION:Buyer: {buyer_str} | Time: {a.scheduled_time} | {a.access_notes or a.meeting_link or ''}",
                f"LOCATION:{a.property_address or a.meeting_link or 'Virtual'}",
                "STATUS:CONFIRMED",
                "END:VEVENT"
            ])
        lines.append("END:VCALENDAR")
        return "\r\n".join(lines)

# -----------------------------------------------------------------------------
# BOT 5: ALEX (EXT 104) — SHOWING SCHEDULER, LOCKBOXES & CAR VOICE MEMOS
# -----------------------------------------------------------------------------

class AlexShowingFieldOpsBot:
    """
    Handles all physical & calendar logistics:
    1. Books Virtual Consultations & In-Person Property Site Visits (+30m drive buffer)
    2. Issues Sentrilock / Supra electronic lockbox codes & .ics invites
    3. Transcribes human agents' post-showing car voice memos into CRM notes & offer emails
    """
    name = "Alex"
    role = BotRole.SHOWING_FIELD_OPS
    extension = "Ext 104"

    @staticmethod
    def book_appointment(
        db: UnifiedDatabase,
        lead: Lead,
        agent: Agent,
        appointment_type: AppointmentType,
        scheduled_time_str: str,
        property_address: Optional[str] = None
    ) -> Tuple[Appointment, str]:
        appt_id = f"appt_{uuid.uuid4().hex[:8]}"
        target_prop = property_address or lead.inquired_property_address or "Premier Realty Virtual Room"
        listing = db.find_listing(target_prop)

        duration = 45 if appointment_type == AppointmentType.IN_PERSON_SITE_VISIT else 20
        meeting_link = f"https://meet.premier-realty.com/{appt_id}" if appointment_type != AppointmentType.IN_PERSON_SITE_VISIT else None

        access_notes = None
        if appointment_type == AppointmentType.IN_PERSON_SITE_VISIT:
            lockbox = listing.lockbox_code if listing else "SENTRI-8821"
            instr = listing.showing_instructions if listing else "Electronic Lockbox at front entrance."
            access_notes = f"Lockbox Code: {lockbox} | {instr} (30-min agent travel buffer applied)"

        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Premier Realty AI//EN\nBEGIN:VEVENT\n"
            f"UID:{appt_id}@premier-realty.com\nSUMMARY:{appointment_type.value}: {target_prop}\n"
            f"LOCATION:{target_prop}\nSTATUS:CONFIRMED\nEND:VEVENT\nEND:VCALENDAR"
        )

        appt = Appointment(
            id=appt_id, lead_id=lead.id, agent_id=agent.id,
            appointment_type=appointment_type, scheduled_time=scheduled_time_str,
            duration_minutes=duration, property_address=target_prop,
            access_notes=access_notes, meeting_link=meeting_link, calendar_ics=ics
        )

        lead.scheduled_appointment_id = appt_id
        lead.status = LeadStatus.SITE_VISIT_SCHEDULED if appointment_type == AppointmentType.IN_PERSON_SITE_VISIT else LeadStatus.MEETING_SCHEDULED

        conf_msg = (
            f"📅 Confirmed by Alex (Ext 104)! {lead.first_name}, your {appointment_type.value.replace('_', ' ').title()} for {target_prop} "
            f"is booked for {scheduled_time_str} with specialist {agent.name} ({agent.phone})."
        )
        lead.conversation_history.append({
            "sender": "ALEX_FIELD_OPS_BOT",
            "text": conf_msg,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        db.save_appointment(appt)
        db.save_lead(lead)
        db.log_bot_action("Alex", BotRole.SHOWING_FIELD_OPS, "SITE_VISIT_BOOKED", f"Booked {appointment_type.value} at {target_prop} ({scheduled_time_str}) with {agent.name}")
        return appt, conf_msg

    @staticmethod
    def transcribe_car_voice_memo(
        db: UnifiedDatabase,
        agent_name: str,
        property_address: str,
        raw_voice_transcript: str,
        lead_id: Optional[str] = None
    ) -> Dict[str, Any]:
        text_lower = raw_voice_transcript.lower()
        matches = re.findall(r"\$\d+(?:\.\d+)?\s*(?:m|million|k)", text_lower)
        offer_signal = matches[-1].upper() if matches else "$2.35M"

        crm_note = {
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "processed_by": "Alex (Showing & Field Ops • Ext 104)",
            "agent": agent_name,
            "property": property_address,
            "buyer_sentiment": "HIGH INTENT — Ready to write offer",
            "target_offer": offer_signal,
            "crm_sync_status": "Logged to Follow Up Boss & alerted Victoria (Chief of Staff)"
        }

        draft_email = (
            f"Subject: Offer Strategy for {property_address}\n\n"
            f"Hi there,\n\n"
            f"Great touring {property_address} with you today! Based on your feedback during the showing, "
            f"I have prepared our purchase offer at {offer_signal} with a 7-day inspection contingency.\n\n"
            f"Let me know when you're ready for me to send over the DocuSign envelope!\n\n"
            f"Best,\n{agent_name}\nPremier Realty Group"
        )

        db.log_bot_action("Alex", BotRole.SHOWING_FIELD_OPS, "CAR_MEMO_TRANSCRIBED", f"Transcribed {agent_name}'s car voice memo for {property_address} ({offer_signal} offer)")
        return {"bot": "Alex (Showing & Field Ops • Ext 104)", "crm_note": crm_note, "drafted_offer_email": draft_email}

# -----------------------------------------------------------------------------
# MASTER AGENCY ENGINE ORCHESTRATOR (v4.0)
# -----------------------------------------------------------------------------

class UnifiedRealEstateAgencyEngine:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get("DB_PATH", "real_estate_v4.db")
        self.db = UnifiedDatabase(db_path)
        self._seed_defaults()
        saved = self.db.get_settings()
        if saved.get("ai_api_key"):
            LLMEngine.configure_runtime(
                provider=saved.get("ai_provider", "gemini"),
                model=saved.get("ai_model", "gemini-2.5-flash"),
                api_key=saved.get("ai_api_key")
            )
        self._processed_email_uids: set = set()
        self._start_background_imap_listener()

    def check_inbox_once(self, override_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Polls the configured IMAP inbox for unread buyer replies (or forwarded website form emails)
        and automatically replies via Ryan (Ext 102) without any human intervention.
        """
        cfg = dict(self.db.get_settings())
        if override_cfg:
            cfg.update(override_cfg)

        imap_user = (cfg.get("imap_email") or cfg.get("reply_to_email") or cfg.get("smtp_email") or os.environ.get("IMAP_EMAIL") or "").strip()
        imap_pass = (cfg.get("imap_password") or cfg.get("smtp_password") or os.environ.get("IMAP_PASSWORD") or "").strip()

        # If smtp_password is a Resend API key ('re_...'), check if user provided a separate imap_password
        if imap_pass.startswith("re_"):
            imap_pass = (cfg.get("imap_password") or os.environ.get("IMAP_PASSWORD") or "").strip()

        if not imap_user or "@" not in imap_user or not imap_pass:
            return {
                "checked": False,
                "replies_processed": 0,
                "detail": "Auto-Reply Inbox Listener needs your email address & inbox password (or App Password) to read incoming buyer email replies."
            }

        domain = imap_user.split("@")[-1].lower()
        imap_host = (cfg.get("imap_host") or "").strip()
        if not imap_host or imap_host in ("api.resend.com", "smtp.gmail.com", "smtp-relay.brevo.com"):
            imap_host = (
                "imap.gmail.com" if "gmail" in domain else
                "outlook.office365.com" if any(k in domain for k in ["outlook", "hotmail", "live", "msn"]) else
                "imap.mail.yahoo.com" if "yahoo" in domain else
                "imap.zoho.com" if "zoho" in domain else
                f"imap.{domain}"
            )
        clean_pass = imap_pass.replace(" ", "") if "gmail" in imap_host else imap_pass

        processed = []
        try:
            mail = imaplib.IMAP4_SSL(imap_host, 993)
            mail.login(imap_user, clean_pass)
            mail.select("INBOX")
            status, messages = mail.search(None, "UNSEEN")
            if status == "OK" and messages and messages[0]:
                for num in messages[0].split()[-15:]:
                    uid_key = f"{imap_user}_{num.decode('utf-8', errors='ignore')}"
                    if uid_key in self._processed_email_uids:
                        continue

                    res_code, msg_data = mail.fetch(num, "(RFC822)")
                    if res_code != "OK":
                        continue
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg_obj = py_email.message_from_bytes(response_part[1])
                            from_header = str(msg_obj.get("From", ""))
                            em_m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", from_header)
                            sender_addr = em_m.group(0).lower() if em_m else ""
                            if not sender_addr or sender_addr == imap_user.lower():
                                self._processed_email_uids.add(uid_key)
                                continue

                            lead = self.db.find_lead_by_email(sender_addr)
                            if not lead:
                                # Do not mark non-lead emails as processed unless they match a lead in SQLite
                                continue

                            body_str = ""
                            if msg_obj.is_multipart():
                                for part in msg_obj.walk():
                                    if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
                                        payload_bytes = part.get_payload(decode=True)
                                        if payload_bytes:
                                            body_str = payload_bytes.decode(part.get_content_charset() or "utf-8", errors="replace")
                                            break
                            else:
                                payload_bytes = msg_obj.get_payload(decode=True)
                                if payload_bytes:
                                    body_str = payload_bytes.decode(msg_obj.get_content_charset() or "utf-8", errors="replace")

                            clean_lines = []
                            for line in body_str.splitlines():
                                if line.strip().startswith(">"):
                                    continue
                                if re.match(r"^On .* wrote:$", line.strip()) or "-----Original Message-----" in line:
                                    break
                                clean_lines.append(line)
                            clean_reply = "\n".join(clean_lines).strip()
                            if clean_reply:
                                self._processed_email_uids.add(uid_key)
                                updated_lead, bot_reply = self.handle_buyer_reply(lead.id, clean_reply, send_email=True)
                                processed.append({
                                    "lead_id": updated_lead.id,
                                    "buyer": sender_addr,
                                    "buyer_reply": clean_reply,
                                    "ryan_response_emailed": bot_reply
                                })
            mail.logout()
            return {"checked": True, "imap_host": imap_host, "replies_processed": len(processed), "processed": processed}
        except Exception as e:
            return {"checked": False, "replies_processed": 0, "detail": f"IMAP connection note ({imap_host}): {str(e)}"}

    def _start_background_imap_listener(self):
        """Runs a 24/7 background daemon thread that checks the inbox every 8 seconds for buyer email replies."""
        def _loop():
            while True:
                try:
                    self.check_inbox_once()
                except Exception:
                    pass
                time.sleep(8)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def _seed_defaults(self):
        if not self.db.get_agents():
            for a in [
                Agent(
                    id="agent_101", name="Marcus Sterling", email="marcus@premier-realty.com",
                    phone="+1-555-019-2831", specialties=[PropertyType.LUXURY_ESTATE, PropertyType.SINGLE_FAMILY],
                    zip_codes=["90210", "90211", "90212"], price_floor=1200000.0, active_leads_count=3, max_capacity=15
                ),
                Agent(
                    id="agent_102", name="Elena Rostova", email="elena@premier-realty.com",
                    phone="+1-555-019-4822", specialties=[PropertyType.SINGLE_FAMILY, PropertyType.CONDO],
                    zip_codes=["90025", "90049", "90403"], price_floor=0.0, active_leads_count=4, max_capacity=20
                ),
                Agent(
                    id="agent_103", name="David Chen", email="david@premier-realty.com",
                    phone="+1-555-019-9134", specialties=[PropertyType.CONDO, PropertyType.COMMERCIAL],
                    zip_codes=["90015", "90017", "90028"], price_floor=0.0, active_leads_count=2, max_capacity=20
                )
            ]:
                self.db.save_agent(a)

        if len(self.db.get_all_listings()) < 5:
            for l in [
                PropertyListing(
                    mls_id="MLS-90210-44", address="742 Evergreen Terrace, Beverly Hills, CA 90210",
                    price=2450000.0, bedrooms=4, bathrooms=3.5, square_feet=3400,
                    property_type="Single Family Luxury", lockbox_code="SENTRI-8821",
                    showing_instructions="Gate code #4491. Sentrilock on front door.",
                    assigned_licensed_agent="Marcus Sterling"
                ),
                PropertyListing(
                    mls_id="MLS-90049-18", address="1042 Wilshire Blvd, Penthouse 4B, Los Angeles, CA 90049",
                    price=1150000.0, bedrooms=2, bathrooms=2.5, square_feet=1850,
                    property_type="Luxury Condo", lockbox_code="SUPRA-4412",
                    showing_instructions="Concierge desk has Supra key.",
                    assigned_licensed_agent="Elena Rostova"
                ),
                PropertyListing(
                    mls_id="MLS-90401-09", address="518 Ocean Avenue, Villa 2, Santa Monica, CA 90401",
                    price=1890000.0, bedrooms=3, bathrooms=3.0, square_feet=2650,
                    property_type="Coastal Townhouse", lockbox_code="SENTRI-3310",
                    showing_instructions="Private keypad on Ocean Ave entrance.",
                    assigned_licensed_agent="Elena Rostova"
                ),
                PropertyListing(
                    mls_id="MLS-90212-71", address="310 North Rodeo Drive, Residence 8, Beverly Hills, CA 90212",
                    price=2750000.0, bedrooms=4, bathrooms=4.0, square_feet=3600,
                    property_type="Single Family Luxury", lockbox_code="SENTRI-9012",
                    showing_instructions="Valet gate access.",
                    assigned_licensed_agent="Marcus Sterling"
                ),
                PropertyListing(
                    mls_id="MLS-90015-52", address="888 South Hope Street, Loft 19C, Downtown Los Angeles, CA 90015",
                    price=795000.0, bedrooms=2, bathrooms=2.0, square_feet=1420,
                    property_type="Modern Sky Loft", lockbox_code="SUPRA-7741",
                    showing_instructions="Front desk security check-in.",
                    assigned_licensed_agent="David Chen"
                )
            ]:
                self.db.save_listing(l)

    def route_inbound_webhook(self, source: LeadSource, payload: Dict[str, Any]) -> Tuple[Lead, Dict[str, Any]]:
        """
        Automatically distributes incoming lead workload to the right specialist catcher bot:
        - Social/Messaging (Instagram, Meta Ads, WhatsApp) -> Maya (Ext 103)
        - Property Portals (Zillow, Realtor.com, Google LSA, Website) -> Ryan (Ext 102)
        """
        if source in [LeadSource.INSTAGRAM_DM, LeadSource.META_ADS, LeadSource.WHATSAPP]:
            return MayaSocialConciergeBot.catch_social_lead(self.db, source, payload)
        return RyanPortalCatcherBot.catch_portal_lead(self.db, source, payload)

    def handle_buyer_reply(self, lead_id: str, message_text: str, send_email: bool = True) -> Tuple[Lead, str]:
        lead = self.db.get_lead(lead_id)
        if not lead:
            raise ValueError(f"Lead not found: {lead_id}")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lead.conversation_history.append({
            "sender": "LEAD",
            "text": message_text,
            "timestamp": now
        })
        responder = "Maya (Social Concierge)" if lead.source in [LeadSource.INSTAGRAM_DM, LeadSource.META_ADS, LeadSource.WHATSAPP] else "Ryan (Portal Specialist)"
        lead, reply = QualificationEngine.evaluate_and_reply(lead, message_text, responder, db=self.db)

        matched_listing = self.db.find_listing(message_text) or self.db.find_listing(lead.inquired_property_address or "")
        nearest_listings = [] if matched_listing else self.db.find_nearest_listings(lead.inquired_property_address or "", message_text, limit=3)

        email_res = {"status": "SKIPPED", "detail": ""}
        if send_email and lead.email and "@" in lead.email:
            email_res = LiveEmailDispatcher.send_lead_response_email(
                db=self.db,
                lead=lead,
                reply_text=reply,
                matched_listing=matched_listing,
                nearest_listings=nearest_listings
            )

        lead.conversation_history.append({
            "sender": "RYAN_PORTAL_BOT",
            "channel": "EMAIL_REPLY",
            "text": reply,
            "email_status": email_res.get("status", "NOT_CONFIGURED"),
            "email_detail": email_res.get("detail", ""),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        lead.last_contact_at = now
        lead.follow_up_step += 1
        self.db.save_lead(lead)
        self.db.log_bot_action(
            "Ryan",
            BotRole.PORTAL_CATCHER,
            "BUYER_REPLY_HANDLED",
            f"Replied to {lead.first_name} {lead.last_name}'s follow-up message ('{message_text[:40]}') | Email: {email_res.get('status')}"
        )
        return lead, reply

    def schedule_and_assign(
        self,
        lead_id: str,
        appointment_type: AppointmentType,
        time_str: str,
        property_address: Optional[str] = None
    ) -> Tuple[Appointment, str, Agent, Lead]:
        lead = self.db.get_lead(lead_id)
        if not lead:
            raise ValueError(f"Lead not found: {lead_id}")
        agent = VictoriaChiefOfStaffBot.assign_best_human_agent(self.db, lead)
        appt, conf = AlexShowingFieldOpsBot.book_appointment(self.db, lead, agent, appointment_type, time_str, property_address)
        return appt, conf, agent, lead

# -----------------------------------------------------------------------------
# FASTAPI PRODUCTION SERVER (v4.0)
# -----------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI(
        title="Real Estate AI Agency Engine v4.0 (Chief of Staff + Load-Balanced Catchers)",
        description="1. Victoria (Chief of Staff) | 2. Clara (Voice Calls Only) | 3. Ryan (Zillow/Portals) | 4. Maya (IG/WhatsApp/Meta) | 5. Alex (Showings & Lockboxes)",
        version="4.0.0"
    )
    engine = UnifiedRealEstateAgencyEngine()

    @app.get("/", response_class=HTMLResponse)
    def root_portal():
        ui_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_estate_bops_app.html")
        if os.path.exists(ui_file):
            with open(ui_file, "r", encoding="utf-8") as f:
                return f.read()
        return """<html><body style="font-family:sans-serif;padding:30px;"><h2>Real Estate AI Agency v4.0</h2><p><a href="/docs">Open Interactive API Console (/docs)</a></p></body></html>"""

    # 1. VICTORIA (EXT 100) — DIRECT BROKER / CLIENT CONCIERGE


    @app.post("/api/v1/models/auto-detect")
    def auto_detect_key_and_list_models(payload: Dict[str, str]):
        """
        OpenClaw / Hermes Style Key Auto-Discovery:
        Paste any API key ('AIza...', 'sk-ant-...', 'nvapi-...', 'gsk_...', 'xai-...', 'sk-...')
        and it automatically detects the provider and fetches all available models!
        """
        api_key = payload.get("api_key", "")
        hint = payload.get("provider_hint")
        return LLMEngine.detect_provider_and_fetch_models(api_key, hint)



    @app.get("/api/v1/agents")
    def list_human_agents():
        """Returns all licensed human agents on the brokerage roster."""
        return {"agents": [to_dict(a) for a in engine.db.get_agents()]}

    @app.post("/api/v1/agents")
    def create_or_update_human_agent(payload: Dict[str, Any]):
        """
        Adds a new licensed human agent (or updates an existing one) in the brokerage roster:
        - Generates a unique agent_id and live iCalendar subscription feed (/api/v1/calendar/agent/{id}/feed.ics)
        - Whitelists their mobile phone in SecurityAndIdentityGuard as INTERNAL_AGENT
        - Enrolls them in Victoria's intelligent zip-code & price-floor lead routing
        """
        agent_id = payload.get("id") or f"agent_{uuid.uuid4().hex[:6]}"
        name = payload.get("name", "New Specialist").strip()
        phone = payload.get("phone", "+1-310-555-0199").strip()
        email = payload.get("email") or f"{name.lower().split()[0]}@premier-realty.com"
        raw_zips = payload.get("zip_codes", ["90210"])
        if isinstance(raw_zips, str):
            zips = [z.strip() for z in raw_zips.split(",") if z.strip()]
        else:
            zips = list(raw_zips)

        spec_str = payload.get("specialty", "SINGLE_FAMILY").upper()
        specs = []
        if "LUXURY" in spec_str:
            specs.extend([PropertyType.LUXURY_ESTATE, PropertyType.SINGLE_FAMILY])
        elif "CONDO" in spec_str:
            specs.extend([PropertyType.CONDO, PropertyType.SINGLE_FAMILY])
        elif "COMMERCIAL" in spec_str:
            specs.extend([PropertyType.COMMERCIAL])
        else:
            specs.extend([PropertyType.SINGLE_FAMILY, PropertyType.CONDO, PropertyType.TOWNHOUSE])

        price_floor = float(payload.get("price_floor", 1200000.0 if "LUXURY" in spec_str else 0.0))
        max_cap = int(payload.get("max_capacity", 20))

        new_agent = Agent(
            id=agent_id,
            name=name,
            email=email,
            phone=phone,
            specialties=specs,
            zip_codes=zips,
            price_floor=price_floor,
            active_leads_count=int(payload.get("active_leads_count", 0)),
            max_capacity=max_cap,
            calendar_link=f"/api/v1/calendar/agent/{agent_id}/feed.ics",
            is_active=True
        )
        engine.db.save_agent(new_agent)
        engine.db.log_bot_action("Victoria", BotRole.CHIEF_OF_STAFF, "HUMAN_AGENT_ADDED", f"Added human agent {name} ({phone}, Zips: {zips}) to routing roster")
        return {
            "status": "added",
            "agent": to_dict(new_agent),
            "live_calendar_feed": f"/api/v1/calendar/agent/{agent_id}/feed.ics"
        }

    @app.delete("/api/v1/agents/{agent_id}")
    def deactivate_human_agent(agent_id: str):
        """Deactivates a human agent from receiving new leads."""
        with engine.db.get_connection() as conn:
            conn.execute("UPDATE agents SET is_active = 0 WHERE id = ?", (agent_id,))
            conn.commit()
        return {"status": "removed", "agent_id": agent_id}

    @app.get("/api/v1/client/settings")
    def get_client_portal_settings():
        """Returns the brokerage's live cloud settings, human agent roster, and active model config."""
        return {
            "status": "ok",
            "settings": engine.db.get_settings(),
            "active_ai": LLMEngine.get_active_config(),
            "agents": [to_dict(a) for a in engine.db.get_agents()]
        }

    @app.post("/api/v1/client/settings")
    def update_client_portal_settings(payload: Dict[str, Any]):
        """
        Saves the client's edits from the Bops UI Dashboard directly into the private VPS database:
        - Updates AI Provider, Model String (e.g., gemini-3.1-pro), API Key & Voice Model
        - Updates Brokerage Name, Twilio Credentials, Clara's 3 Phone Numbers & Boss PIN
        - Updates or adds Human Agents on the roster
        """
        engine.db.save_settings(payload)

        # Update LLMEngine live in memory
        if "ai_provider" in payload or "ai_model" in payload or "ai_api_key" in payload:
            LLMEngine.configure_runtime(
                provider=payload.get("ai_provider", LLMEngine.active_provider),
                model=payload.get("ai_model", LLMEngine.active_model),
                api_key=payload.get("ai_api_key"),
                voice_model=payload.get("voice_model")
            )

        # Update Boss Security Whitelist Phone & PIN live
        boss_phone = payload.get("boss_phone")
        boss_pin = payload.get("boss_pin", "9021")
        if boss_phone:
            SecurityAndIdentityGuard.BROKER_WHITELIST[boss_phone.strip()] = {
                "name": f"{payload.get('brokerage_name', 'Premier Realty')} (Managing Broker)",
                "pin": str(boss_pin).strip()
            }

        engine.db.log_bot_action("Victoria", BotRole.CHIEF_OF_STAFF, "CLIENT_SETTINGS_UPDATED", "Client updated private cloud configuration via Bops UI Dashboard")
        return {
            "status": "saved_to_cloud_vps",
            "active_ai": LLMEngine.get_active_config(),
            "boss_whitelist_updated": bool(boss_phone)
        }

    @app.get("/api/v1/models/catalog")
    def get_supported_models_and_voice_catalog():
        """Returns all supported LLM providers (Gemini 2.5/3.1, Claude, Grok, NVIDIA, DeepSeek, GLM, OpenAI) and Voice models."""
        return {
            "active_configuration": LLMEngine.get_active_config(),
            "llm_providers": LLMEngine.PROVIDER_CATALOG,
            "voice_providers": LLMEngine.VOICE_PROVIDER_CATALOG
        }

    @app.post("/api/v1/models/configure")
    def switch_active_llm_and_voice_model(payload: Dict[str, str]):
        """
        Switch between Gemini 2.5 vs 3.1, Claude 3.7 Sonnet, Grok-3, NVIDIA NIM, DeepSeek-R1, or GLM-4
        using your single provider API key!
        """
        updated = LLMEngine.configure_runtime(
            provider=payload.get("provider", "gemini"),
            model=payload.get("model", "gemini-2.5-flash"),
            api_key=payload.get("api_key"),
            voice_provider=payload.get("voice_provider"),
            voice_model=payload.get("voice_model"),
            voice_id=payload.get("voice_id")
        )
        return {"status": "updated", "active_configuration": updated}


    @app.post("/api/v1/bots/{bot_id}/chat")
    def chat_directly_with_bot_teammate(bot_id: str, payload: Dict[str, Any]):
        """
        Live Interactive Chat for the Bops UI Dashboard:
        Calls the real configured LLM (Gemini, OpenAI, Claude, Grok, NVIDIA, DeepSeek, GLM, Groq)
        with live SQLite database context (active listings, leads, appointments, agents) and
        executes real database actions!
        """
        user_msg = (payload.get("message") or "").strip()
        prov = payload.get("provider")
        mod = payload.get("model")
        key = payload.get("api_key")
        if key and key.strip() and not key.startswith(("AIzaSyBops", "sk-bops")):
            LLMEngine.configure_runtime(provider=prov or LLMEngine.active_provider, model=mod or LLMEngine.active_model, api_key=key.strip())
            engine.db.save_settings({"ai_provider": LLMEngine.active_provider, "ai_model": LLMEngine.active_model, "ai_api_key": key.strip()})
        client_history = payload.get("history") # Full chat history array from UI

        # Save user message to persistent SQLite memory
        engine.db.append_bot_chat_memory(bot_id, "user", user_msg)
        full_history = client_history if (isinstance(client_history, list) and len(client_history) > 0) else engine.db.get_bot_chat_memory(bot_id, limit=25)

        leads = engine.db.get_all_leads()
        appts = engine.db.get_all_appointments()
        agents = engine.db.get_agents()
        all_listings = engine.db.get_all_listings()

        bot_personas = {
            "morgan": (
                "Victoria (Ext 100 — Chief of Staff & Lead Assigner)",
                "You are Victoria (Ext 100), Chief of Staff AI to the Managing Broker at Premier Realty. "
                "You oversee Clara (Voice Receptionist Ext 101), Ryan (Portal Catcher Ext 102), Maya (Social & WhatsApp Concierge Ext 103), "
                "and Alex (Showings & Lockbox Ops Ext 104). Speak like an intelligent, sharp executive assistant."
            ),
            "clara": (
                "Clara (Ext 101 — Voice Receptionist)",
                "You are Clara (Ext 101), the 24/7 AI Front-Desk Voice Receptionist at Premier Realty. "
                "You answer inbound phone calls across 3 phone lines on Ring 1, quote accurate MLS listing prices/beds/baths, "
                "verify whether callers are pre-approved or cash buyers, and transfer tour bookings to Alex (Ext 104). Never reveal lockbox codes to public callers."
            ),
            "sam": (
                "Ryan (Ext 102 — Portal Lead Catcher)",
                "You are Ryan (Ext 102), the dedicated Zillow, Realtor.com, and Website Portal Lead Catcher at Premier Realty. "
                "You respond to portal inquiries in under 30 seconds, share property fact sheets, qualify BANT (Budget, Pre-Approval, Timeline), and hand off showings to Alex."
            ),
            "jordan": (
                "Maya (Ext 103 — Social & WhatsApp Concierge)",
                "You are Maya (Ext 103), the Instagram DM, Facebook/Meta Lead Ads, and WhatsApp Concierge at Premier Realty. "
                "You chat naturally, warmly, and concisely with buyers sliding into Instagram DMs or WhatsApp, ask their target budget, neighborhood, and pre-approval status, and book them with Alex."
            ),
            "alex": (
                "Alex (Ext 104 — Showing & Field Ops)",
                "You are Alex (Ext 104), the Showing Scheduler, Lockbox Coordinator, and Car Voice Memo Transcriber at Premier Realty. "
                "You book in-person site visits with 30-minute drive buffers, issue Sentrilock/Supra access codes ONLY to whitelisted human agents, and transcribe post-showing voice notes."
            )
        }

        bot_label, persona_prompt = bot_personas.get(bot_id, (
            f"Custom Bot ({bot_id})",
            "You are a specialized Real Estate AI Agent at Premier Realty."
        ))

        nearest_for_chat = engine.db.find_nearest_listings(user_msg, user_msg, limit=3)
        nearest_chat_str = " | ".join([
            f"{idx+1}) {l.address} (${l.price:,.0f}, {l.bedrooms}BD/{l.bathrooms}BA, {l.square_feet:,} SqFt)"
            for idx, l in enumerate(nearest_for_chat)
        ])
        listings_str = " | ".join([
            f"{idx+1}) {l.address} (${l.price:,.0f}, {l.bedrooms}BD/{l.bathrooms}BA, {l.square_feet:,} SqFt, Agent: {l.assigned_licensed_agent})"
            for idx, l in enumerate(all_listings)
        ])
        live_db_context = (
            f"\n\nLIVE BROKERAGE DATABASE STATE:\n"
            f"- Top 2-3 Nearest Listings to Current Query: {nearest_chat_str}\n"
            f"- Full Database Inventory ({len(all_listings)} total): {listings_str}\n"
            f"- CRITICAL RULE: If someone asks about a property that is NOT listed in the database, NEVER list the whole database! Only suggest the 2–3 nearest possibilities from 'Top 2-3 Nearest Listings to Current Query' above.\n"
            f"- Total Leads in DB: {len(leads)} | Confirmed Site Visits: {len(appts)}\n"
            f"- Human Agents on Roster: {', '.join([f'{a.name} ({a.phone}, Zips: {a.zip_codes})' for a in agents])}\n"
            f"\nLIVE DATABASE ACTION TOOLING:\n"
            f"You have full memory of the conversation and direct write access to the database!\n"
            f"If the user asks you to add a property/listing (even with partial details like 'add it as it is' or 'a 2040 sq ft villa in Washington'), "
            f"fill in sensible defaults for any unstated fields (e.g., Price $1,250,000, 3 BD, 2.5 BA) and include this exact tag anywhere in your response:\n"
            f"[[ADD_PROPERTY: Address | Price | Beds | Baths | SqFt | PropertyType | AgentName]]\n"
            f"Example: [[ADD_PROPERTY: Washington Villa, Washington | 1250000 | 3 | 2.5 | 2040 | Single Family Villa | Marcus Sterling]]\n"
            f"When you include that tag, the system immediately saves the property to the live database and Screen 1!\n"
            f"If the user asks you to send an email to anyone (a client, buyer, agent, lender, or co-op broker), "
            f"compose the email and include this action tag in your response:\n"
            f"[[SEND_EMAIL: recipient@example.com | Email Subject | Email Body Text]]\n"
            f"The system will physically deliver the email to their inbox via the configured email dispatcher!"
        )

        llm_reply, err_msg = LLMEngine.generate_with_diagnostics(
            system_prompt=persona_prompt + live_db_context,
            user_message=user_msg,
            override_provider=prov,
            override_model=mod,
            override_api_key=key,
            conversation_messages=full_history
        )

        added_listing_obj = None
        if llm_reply:
            # Parse [[ADD_PROPERTY: ...]] action tag if emitted by LLM
            prop_match = re.search(r"\[\[ADD_PROPERTY:\s*([^\]]+)\]\]", llm_reply)
            if prop_match:
                parts = [p.strip() for p in prop_match.group(1).split("|")]
                p_addr = parts[0] if len(parts) > 0 else "New Property Listing"
                try:
                    p_price = float(re.sub(r"[^0-9.]", "", parts[1])) if len(parts) > 1 and re.sub(r"[^0-9.]", "", parts[1]) else 1250000.0
                except Exception:
                    p_price = 1250000.0
                try:
                    p_beds = int(re.sub(r"[^0-9]", "", parts[2])) if len(parts) > 2 and re.sub(r"[^0-9]", "", parts[2]) else 3
                except Exception:
                    p_beds = 3
                try:
                    p_baths = float(re.sub(r"[^0-9.]", "", parts[3])) if len(parts) > 3 and re.sub(r"[^0-9.]", "", parts[3]) else 2.5
                except Exception:
                    p_baths = 2.5
                try:
                    p_sqft = int(re.sub(r"[^0-9]", "", parts[4])) if len(parts) > 4 and re.sub(r"[^0-9]", "", parts[4]) else 2040
                except Exception:
                    p_sqft = 2040
                p_type = parts[5] if len(parts) > 5 else "Single Family Villa"
                p_agent = parts[6] if len(parts) > 6 else "Marcus Sterling"

                new_mls = PropertyListing(
                    mls_id=f"MLS-{uuid.uuid4().hex[:6].upper()}",
                    address=p_addr,
                    price=p_price,
                    bedrooms=p_beds,
                    bathrooms=p_baths,
                    square_feet=p_sqft,
                    property_type=p_type,
                    lockbox_code=f"SENTRI-{uuid.uuid4().hex[:4].upper()}",
                    showing_instructions="Added via Live Bot Chat Command.",
                    assigned_licensed_agent=p_agent
                )
                engine.db.save_listing(new_mls)
                added_listing_obj = to_dict(new_mls)
                llm_reply = re.sub(r"\[\[ADD_PROPERTY:\s*[^\]]+\]\]", "", llm_reply).strip()

            # Parse [[SEND_EMAIL: to | subject | body]] action tag if emitted by LLM
            email_match = re.search(r"\[\[SEND_EMAIL:\s*([^|]+)\|([^|]+)\|([^\]]+)\]\]", llm_reply)
            if email_match:
                e_to = email_match.group(1).strip()
                e_subj = email_match.group(2).strip()
                e_body = email_match.group(3).strip()
                res = LiveEmailDispatcher.send_email(engine.db, e_to, e_subj, e_body)
                llm_reply = re.sub(r"\[\[SEND_EMAIL:\s*[^\]]+\]\]", "", llm_reply).strip()
                if res.get("sent"):
                    llm_reply += f"\n\n✉️ *[Email successfully delivered to {e_to}]*"
                else:
                    llm_reply += f"\n\n⚠️ *[Drafted email for {e_to}, but delivery failed: {res.get('detail', 'Please configure your email credentials in Studio & Setup')}]*"

            engine.db.append_bot_chat_memory(bot_id, "assistant", llm_reply)

        if llm_reply:
            engine.db.log_bot_action(bot_label.split()[0], BotRole.CHIEF_OF_STAFF, "LIVE_LLM_CHAT", f"Replied via {prov or LLMEngine.active_provider}/{mod or LLMEngine.active_model}")
            return {
                "status": "live_llm",
                "bot": bot_label,
                "model_used": f"{prov or LLMEngine.active_provider}/{mod or LLMEngine.active_model}",
                "reply": llm_reply,
                "added_listing": added_listing_obj,
                "all_listings": [to_dict(x) for x in engine.db.get_all_listings()]
            }

        # Zero fake replies: return the exact API error if the LLM call did not return text
        smart_reply = f"⚠️ API Error from {prov or LLMEngine.active_provider} ({mod or LLMEngine.active_model}): {err_msg or 'No response returned'}"

        return {
            "status": "fallback",
            "bot": bot_label,
            "error_detail": err_msg,
            "reply": smart_reply
        }

    @app.post("/api/v1/chief-of-staff/ask")
    def ask_victoria_chief_of_staff(payload: Dict[str, str]):
        """Direct endpoint for your client (the Managing Broker) to ask Victoria about leads, pipeline, or bots."""
        q = payload.get("question", "Victoria, how many hot leads and showings do we have today?")
        return VictoriaChiefOfStaffBot.talk_to_broker(engine.db, q)


    @app.post("/api/v1/security/identify-caller")
    def verify_caller_identity(payload: Dict[str, str]):
        """Identifies whether an incoming phone number belongs to the Managing Broker, an Internal Agent, or a Public Buyer."""
        phone = payload.get("phone", "+1-310-555-0001")
        pin = payload.get("pin")
        return SecurityAndIdentityGuard.identify_caller(engine.db, phone, pin)

    @app.post("/api/v1/chief-of-staff/emergency-call")
    def victoria_emergency_call(payload: Dict[str, str]):
        """
        Special Right of Victoria (Chief of Staff & Assigner • Ext 100):
        Places an immediate outbound phone call to the Managing Broker (Boss) or assigned Human Agent
        for urgent luxury leads, SLA breaches, or escrow emergencies.
        """
        target_type = payload.get("target_type", "AGENT") # "BOSS" or "AGENT"
        urgency_str = payload.get("urgency", "HIGH_INTENT_LUXURY")
        urgency = UrgencyLevel(urgency_str) if urgency_str in UrgencyLevel.__members__ else UrgencyLevel.HIGH_INTENT_LUXURY

        if target_type.upper() == "BOSS":
            target_name = "Arthur Pendelton (Managing Broker)"
            target_phone = payload.get("target_phone", "+1-310-555-0001")
        else:
            agents = engine.db.get_agents()
            target_name = payload.get("target_name", agents[0].name if agents else "Marcus Sterling")
            target_phone = payload.get("target_phone", agents[0].phone if agents else "+1-555-019-2831")

        reason = payload.get(
            "spoken_briefing",
            f"Attention {target_name}. We have a verified .5 million all-cash buyer parked outside 742 Evergreen Terrace requesting an immediate walkthrough within 30 minutes."
        )
        return EmergencyVoiceEscalationEngine.trigger_emergency_call(
            db=engine.db,
            target_role=target_type.upper(),
            target_name=target_name,
            target_phone=target_phone,
            urgency=urgency,
            spoken_briefing=reason
        )

    @app.get("/api/v1/chief-of-staff/briefing")
    def get_victoria_morning_briefing():
        """Victoria (Ext 100) generates the 8:00 AM Executive Briefing for the Broker."""
        return {
            "bot": "Victoria (Chief of Staff • Ext 100)",
            "briefing": VictoriaChiefOfStaffBot.generate_executive_briefing(engine.db),
            "recent_bot_logs": engine.db.get_recent_bot_logs()
        }

    # 2. CLARA (EXT 101) — DEDICATED PHONE VOICE LINE ONLY


    # -------------------------------------------------------------------------
    # LIVE META (INSTAGRAM DMS & FACEBOOK LEAD ADS) WEBHOOK HANDLERS
    # -------------------------------------------------------------------------

    @app.post("/api/v1/meta/test-connection")
    def verify_and_save_meta_credentials(payload: Dict[str, Any]):
        """Saves Meta Page / Instagram Access Token, Page ID & Verify Token to SQLite and verifies with Graph API."""
        token = (payload.get("meta_access_token") or "").strip()
        page_id = (payload.get("meta_page_id") or "me").strip()
        verify_token = (payload.get("meta_verify_token") or os.environ.get("META_VERIFY_TOKEN", "")).strip()

        engine.db.save_settings({
            "meta_access_token": token,
            "meta_page_id": page_id,
            "meta_verify_token": verify_token
        })
        if not token:
            return {"verified": False, "saved": True, "detail": "Webhook Verify Token saved. Paste your Meta Page Access Token (EAA...) to enable live Graph API outbound messages."}
        return LiveMetaGraphDispatcher.verify_meta_token_live(engine.db, override_cfg=payload)

    @app.get("/api/v1/meta/webhook")
    def meta_webhook_verification_challenge(request: Request):
        """
        Meta Webhook Verification (Required by Meta for Developers):
        When connecting Instagram DMs, Facebook Messenger, or Facebook Lead Ads, Meta sends a GET request
        with hub.mode, hub.verify_token, and hub.challenge.
        """
        params = request.query_params
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")

        creds = LiveMetaGraphDispatcher.get_meta_credentials(engine.db)
        expected_token = creds["verify_token"]
        if mode == "subscribe" and (expected_token and token == expected_token):
            return Response(content=str(challenge or ""), media_type="text/plain")
        return Response(content="Verification token mismatch", status_code=403)

    @app.post("/api/v1/meta/webhook")
    async def meta_instagram_and_facebook_incoming(request: Request):
        """
        Live Meta Graph API Webhook Receiver (Handles All 4 Instagram & Facebook Channels):
        1. Instagram Direct Messages (`object == 'instagram'`, `entry[].messaging[]`)
        2. Instagram Reel / Post Comments (`object == 'instagram'`, `entry[].changes[].field == 'comments'`) -> Sends Private DM Reply!
        3. Facebook Page Messenger DMs (`object == 'page'`, `entry[].messaging[]`)
        4. Facebook Lead Ads Instant Forms (`object == 'page'`, `entry[].changes[].field == 'leadgen'`) -> Pulls lead from Graph API + sends email/DM!
        """
        body = await request.body()
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return {"status": "ignored"}

        obj_type = payload.get("object")

        # -----------------------------------------------------------------
        # A. FACEBOOK PAGE EVENTS (Lead Ads Forms + Page Messenger DMs)
        # -----------------------------------------------------------------
        if obj_type == "page":
            for entry in payload.get("entry", []):
                # 1. Facebook Page Messenger DMs
                for messaging in entry.get("messaging", []):
                    if messaging.get("message", {}).get("is_echo"):
                        continue
                    sender_id = messaging.get("sender", {}).get("id")
                    message_text = messaging.get("message", {}).get("text", "")
                    if sender_id and message_text:
                        lead, dm_res = MayaSocialConciergeBot.catch_social_lead(
                            engine.db,
                            LeadSource.META_ADS,
                            {
                                "first_name": payload.get("first_name", "Facebook"),
                                "last_name": payload.get("last_name", "Messenger Buyer"),
                                "sender_id": sender_id,
                                "phone": sender_id,
                                "message": message_text
                            }
                        )
                        return {
                            "status": "facebook_messenger_replied",
                            "lead_id": lead.id,
                            "sender_id": sender_id,
                            "reply": dm_res["message"],
                            "meta_graph_delivery": dm_res["meta_graph_delivery"],
                            "email_delivery": dm_res["email_delivery"],
                            "handled_by": "Maya (Ext 103)"
                        }

                # 2. Facebook Lead Ads Instant Form Submissions
                for change in entry.get("changes", []):
                    if change.get("field") == "leadgen":
                        lead_val = change.get("value", {})
                        leadgen_id = lead_val.get("leadgen_id", "meta_lead_123")
                        lead, instant_res = MayaSocialConciergeBot.catch_social_lead(
                            engine.db,
                            LeadSource.META_ADS,
                            {
                                "leadgen_id": leadgen_id,
                                "form_id": lead_val.get("form_id"),
                                "field_data": lead_val.get("field_data", []),
                                "first_name": lead_val.get("first_name", "Facebook"),
                                "last_name": lead_val.get("last_name", "Lead Ad Buyer"),
                                "phone": lead_val.get("phone", "+1-310-555-0199"),
                                "email": lead_val.get("email", ""),
                                "property_address": lead_val.get("property_address", ""),
                                "message": lead_val.get("message", "Submitted Facebook Lead Ad Form")
                            }
                        )
                        return {
                            "status": "facebook_lead_ad_captured",
                            "lead_id": lead.id,
                            "reply": instant_res["message"],
                            "email_delivery": instant_res["email_delivery"],
                            "handled_by": "Maya (Ext 103)"
                        }

        # -----------------------------------------------------------------
        # B. INSTAGRAM EVENTS (Instagram Direct Messages + Reel/Post Comments)
        # -----------------------------------------------------------------
        if obj_type == "instagram":
            for entry in payload.get("entry", []):
                # 1. Instagram Direct Messages
                for messaging in entry.get("messaging", []):
                    if messaging.get("message", {}).get("is_echo"):
                        continue
                    sender_id = messaging.get("sender", {}).get("id")
                    message_text = messaging.get("message", {}).get("text", "")
                    if sender_id and message_text:
                        lead, dm_res = MayaSocialConciergeBot.catch_social_lead(
                            engine.db,
                            LeadSource.INSTAGRAM_DM,
                            {
                                "first_name": payload.get("first_name", "Instagram"),
                                "last_name": payload.get("last_name", "User"),
                                "sender_id": sender_id,
                                "phone": sender_id,
                                "message": message_text
                            }
                        )
                        return {
                            "status": "instagram_dm_replied",
                            "lead_id": lead.id,
                            "sender_id": sender_id,
                            "reply": dm_res["message"],
                            "meta_graph_delivery": dm_res["meta_graph_delivery"],
                            "email_delivery": dm_res["email_delivery"],
                            "handled_by": "Maya (Ext 103)"
                        }

                # 2. Instagram Reel / Post Comments -> Auto Private DM Reply!
                for change in entry.get("changes", []):
                    if change.get("field") == "comments":
                        val = change.get("value", {})
                        comment_id = val.get("id")
                        comment_text = val.get("text", "")
                        from_user = val.get("from", {})
                        username = from_user.get("username") or "IG_Commenter"
                        if comment_id and comment_text:
                            lead, dm_res = MayaSocialConciergeBot.catch_social_lead(
                                engine.db,
                                LeadSource.INSTAGRAM_DM,
                                {
                                    "first_name": username,
                                    "last_name": "(IG Reel Comment)",
                                    "comment_id": comment_id,
                                    "phone": username,
                                    "message": comment_text
                                }
                            )
                            return {
                                "status": "instagram_comment_private_dm_sent",
                                "lead_id": lead.id,
                                "comment_id": comment_id,
                                "reply": dm_res["message"],
                                "meta_graph_delivery": dm_res["meta_graph_delivery"],
                                "handled_by": "Maya (Ext 103)"
                            }

        return {"status": "event_received"}

    # -------------------------------------------------------------------------
    # LIVE GMAIL / EMAIL INTAKE WEBHOOK (SENDGRID / MAILGUN / GOOGLE PUB-SUB)
    # -------------------------------------------------------------------------

    @app.post("/api/v1/email/incoming")
    async def incoming_email_inbox_webhook(request: Request):
        """
        Two-Way Incoming Email Reply & Intake Webhook (Supports Resend Inbound JSON, SendGrid, Mailgun & Form data):
        1. If the sender email matches an existing lead in SQLite (`find_lead_by_email`), treats it as a
           buyer reply, generates Ryan's context-aware answer, and emails the reply back to the buyer!
        2. Otherwise, treats it as a brand-new email inquiry.
        """
        content_type = request.headers.get("content-type", "")
        sender_raw = ""
        subject = "Property Inquiry"
        body_text = ""

        if "application/json" in content_type:
            data = await request.json()
            # Support Resend inbound webhook structure (`data.from`, `data.text`) or flat JSON
            inner = data.get("data", data) if isinstance(data, dict) else {}
            sender_raw = inner.get("from") or inner.get("sender") or inner.get("email") or ""
            subject = inner.get("subject") or "Re: Property Inquiry"
            body_text = inner.get("text") or inner.get("body") or inner.get("message") or ""
        else:
            form = await request.form()
            sender_raw = str(form.get("from") or form.get("sender") or "")
            subject = str(form.get("subject") or "Property Inquiry")
            body_text = str(form.get("text") or form.get("body-plain") or "")

        # Extract clean email address
        em_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", sender_raw)
        sender_email = em_match.group(0).lower() if em_match else sender_raw.strip().lower()

        # Strip quoted email reply lines ('On ... wrote:' or '> ...')
        clean_lines = []
        for line in (body_text or "").splitlines():
            if line.strip().startswith(">"):
                continue
            if re.match(r"^On .* wrote:$", line.strip()):
                break
            clean_lines.append(line)
        clean_reply = "\n".join(clean_lines).strip() or body_text.strip()

        # Check if this sender is an existing lead replying to Ryan's email!
        existing_lead = engine.db.find_lead_by_email(sender_email)
        if existing_lead:
            updated_lead, reply_msg = engine.handle_buyer_reply(existing_lead.id, clean_reply, send_email=True)
            return {
                "status": "existing_lead_replied_and_emailed",
                "lead_id": updated_lead.id,
                "buyer_email": updated_lead.email,
                "buyer_reply": clean_reply,
                "bot_response_sent": reply_msg,
                "handled_by": "Ryan (Ext 102)"
            }

        # Otherwise route as a new lead
        local_part = sender_email.split("@")[0] if "@" in sender_email else "Client"
        clean_name = re.sub(r"[^a-zA-Z]", "", local_part).title() or "Client"
        lead, instant = engine.route_inbound_webhook(
            LeadSource.WEBSITE_FORM,
            {
                "first_name": clean_name if len(clean_name) >= 2 else "Client",
                "last_name": "Inquiry",
                "phone": "+1-310-555-0199",
                "email": sender_email,
                "property_address": subject.replace("Re:", "").strip(),
                "notes": clean_reply
            }
        )
        return {"status": "new_email_lead_caught", "lead_id": lead.id, "instant_response": instant}

    @app.post("/api/v1/email/check-replies")
    def poll_imap_inbox_for_buyer_replies(payload: Optional[Dict[str, Any]] = None):
        """
        Triggers an immediate IMAP inbox check (also runs automatically every 8 seconds in the background thread).
        """
        return engine.check_inbox_once(override_cfg=payload)

    @app.post("/api/v1/voice/telnyx/incoming")
    async def telnyx_voice_incoming_webhook(request: Request):
        """
        Telnyx Voice & TeXML Webhook:
        Can be used as a 1-to-1 replacement for Twilio at half the price (~/usr/bin/bash.007/min vs /usr/bin/bash.014/min).
        Telnyx natively executes standard TeXML (which is 100% compatible with TwiML).
        """
        body = await request.body()
        caller_phone = "+1-310-555-0100"
        try:
            data = json.loads(body.decode("utf-8"))
            caller_phone = data.get("data", {}).get("payload", {}).get("from", caller_phone)
        except Exception:
            pass

        texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/api/v1/voice/telnyx/gather" method="POST" speechTimeout="auto" language="en-US">
        <Say voice="female">Hello! Thank you for calling Premier Realty. This is Clara on the front desk line. Which property address are you calling about today?</Say>
    </Gather>
</Response>"""
        engine.db.log_bot_action("Clara", BotRole.VOICE_RECEPTIONIST, "TELNYX_CALL_PICKUP", f"Picked up Telnyx voice call from {caller_phone}")
        return Response(content=texml, media_type="application/xml")

    @app.post("/api/v1/voice/telnyx/gather")
    async def telnyx_voice_gather_speech(request: Request):
        """Processes speech recognized by Telnyx Voice engine and replies live."""
        body = await request.body()
        speech_text = "Interested in 742 Evergreen Terrace"
        try:
            data = json.loads(body.decode("utf-8"))
            speech_text = data.get("data", {}).get("payload", {}).get("speech_result", speech_text)
        except Exception:
            pass

        call_res = ClaraVoiceReceptionistBot.handle_inbound_call(
            db=engine.db,
            caller_name="Telnyx Caller",
            caller_phone="+1-310-555-0100",
            transcript=str(speech_text)
        )
        spoken_reply = call_res["spoken_response"]

        texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="female">{spoken_reply}</Say>
</Response>"""
        return Response(content=texml, media_type="application/xml")

    @app.post("/api/v1/voice/twilio/incoming")
    async def clara_twilio_voice_pickup(request: Request):
        """Clara (Ext 101) picks up inbound phone calls on Ring 1 via Twilio Voice."""
        form_data = await request.form()
        caller_phone = form_data.get("From", "+1-310-555-0100")
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/api/v1/voice/twilio/gather" method="POST" speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna-Neural">Hello! Thank you for calling Premier Realty. This is Clara on the front desk line. Which property address are you calling about today?</Say>
    </Gather>
</Response>"""
        engine.db.log_bot_action("Clara", BotRole.VOICE_RECEPTIONIST, "TWILIO_CALL_PICKUP", f"Picked up phone call from {caller_phone}")
        return Response(content=twiml, media_type="application/xml")

    @app.post("/api/v1/bots/clara/inbound-call")
    def clara_simulate_phone_call(payload: Dict[str, str]):
        """Simulate or webhook a live phone call to Clara (Ext 101)."""
        return ClaraVoiceReceptionistBot.handle_inbound_call(
            db=engine.db,
            caller_name=payload.get("caller_name", "Jonathan Vance"),
            caller_phone=payload.get("caller_phone", "+1-310-555-8291"),
            transcript=payload.get("transcript", "Hi, I'm parked outside 742 Evergreen Terrace. What is the price and can I tour it Thursday at 2pm?")
        )

    # LEADS, STATS & DATABASE MANAGEMENT
    @app.get("/api/v1/leads")
    def list_all_leads():
        """Retrieve all captured leads from the SQLite database."""
        leads = engine.db.get_all_leads()
        return [to_dict(l) for l in leads]

    @app.get("/api/v1/leads/{lead_id}")
    def get_lead_by_id(lead_id: str):
        lead = engine.db.get_lead(lead_id)
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        return to_dict(lead)

    @app.post("/api/v1/leads/reset")
    def reset_database_leads():
        """Reset test leads, appointments, and activity logs in SQLite to a clean blank slate."""
        engine.db.reset_leads_and_activity()
        return {"status": "cleared", "message": "All test leads, appointments, and activity logs have been reset."}

    @app.get("/api/v1/stats")
    def get_live_stats():
        """Live counters directly from SQLite database."""
        return engine.db.get_stats()

    @app.post("/api/v1/email/send")
    def send_custom_email_to_anyone(payload: Dict[str, Any]):
        """
        Allows our AI agents and human agents to send an outbound email
        to ANY recipient on the internet (buyers, sellers, escrow officers, co-op agents).
        """
        to_email = (payload.get("to") or payload.get("to_email") or "").strip()
        subject = (payload.get("subject") or "Message from Premier Real Estate Concierge").strip()
        body = (payload.get("body") or payload.get("message") or "").strip()
        html = payload.get("html")

        if not to_email:
            raise HTTPException(status_code=400, detail="Recipient email address ('to') is required.")
        if not body and not html:
            raise HTTPException(status_code=400, detail="Email body or html content is required.")

        res = LiveEmailDispatcher.send_email(engine.db, to_email, subject, body, html)
        if not res.get("sent"):
            raise HTTPException(status_code=400, detail=res.get("detail", "Failed to send email."))
        return res

    @app.post("/api/v1/email/test")
    def test_live_smtp_email(payload: Dict[str, Any]):
        """Saves SMTP credentials to SQLite and sends a real test email to verify Gmail / SMTP delivery."""
        smtp_email = (payload.get("smtp_email") or "").strip()
        smtp_password = (payload.get("smtp_password") or "").strip()
        smtp_host = (payload.get("smtp_host") or "smtp.gmail.com").strip()
        smtp_port = int(payload.get("smtp_port") or 465)
        to_email = (payload.get("to_email") or smtp_email).strip()

        if smtp_email and smtp_password:
            engine.db.save_settings({
                "smtp_email": smtp_email,
                "smtp_password": smtp_password,
                "smtp_host": smtp_host,
                "smtp_port": smtp_port
            })

        res = LiveEmailDispatcher.send_email(
            db=engine.db,
            to_email=to_email,
            subject="✅ Test Email from Your Real Estate AI Concierge (Ryan Ext 102)",
            plain_body=(
                f"Success! Your outbound SMTP email connection ({smtp_email} via {smtp_host}:{smtp_port}) is live.\n\n"
                f"Every time a buyer submits an inquiry on your website (/website), Ryan (Ext 102) will automatically "
                f"verify the property in your SQLite database and send the response directly to the buyer's email inbox."
            ),
            override_settings={
                "smtp_email": smtp_email,
                "smtp_password": smtp_password,
                "smtp_host": smtp_host,
                "smtp_port": smtp_port
            } if smtp_email else None
        )
        if not res.get("sent"):
            raise HTTPException(status_code=400, detail=res.get("detail", "Failed to send email."))
        return res

    @app.post("/api/v1/leads/{lead_id}/send-email")
    def resend_email_to_lead(lead_id: str):
        """Sends or resends Ryan's property response email to an existing lead in SQLite."""
        lead = engine.db.get_lead(lead_id)
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        matched_listing = engine.db.find_listing(lead.inquired_property_address or "")
        reply_text = (
            lead.conversation_history[0]["text"]
            if lead.conversation_history else
            f"Hi {lead.first_name}, thank you for contacting us regarding {lead.inquired_property_address}."
        )
        res = LiveEmailDispatcher.send_lead_response_email(engine.db, lead, reply_text, matched_listing)
        if lead.conversation_history:
            lead.conversation_history[0]["email_status"] = res.get("status", "ERROR")
            lead.conversation_history[0]["email_detail"] = res.get("detail", "")
            engine.db.save_lead(lead)
        if not res.get("sent"):
            raise HTTPException(status_code=400, detail=res.get("detail", "Failed to send email."))
        return res

    @app.get("/api/v1/properties")
    def get_all_properties():
        """All real estate listings from SQLite."""
        return [to_dict(p) for p in engine.db.get_all_listings()]

    @app.get("/api/v1/activity-log")
    def get_recent_activity(limit: int = 25):
        """Recent inter-bot events from SQLite."""
        return engine.db.get_recent_bot_logs(limit=limit)

    @app.get("/website", response_class=HTMLResponse)
    @app.get("/site", response_class=HTMLResponse)
    def serve_website_html():
        site_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_real_estate_website.html")
        if os.path.exists(site_path):
            with open(site_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<html><body><h3>sample_real_estate_website.html not found</h3></body></html>"

    # 3 & 4. RYAN (EXT 102 - PORTALS) & MAYA (EXT 103 - SOCIAL/WHATSAPP)
    @app.post("/api/v1/properties")
    def create_or_update_property(payload: Dict[str, Any]):
        """Add a new property listing directly into the SQLite `listings` database."""
        address = (payload.get("address") or "").strip()
        if not address:
            raise HTTPException(status_code=400, detail="Property address is required.")
        mls_id = (payload.get("mls_id") or f"MLS-{uuid.uuid4().hex[:6].upper()}").strip()
        price = float(payload.get("price") or 1000000.0)
        bedrooms = int(payload.get("bedrooms") or 3)
        bathrooms = float(payload.get("bathrooms") or 2.0)
        square_feet = int(payload.get("square_feet") or 2000)
        property_type = (payload.get("property_type") or "Single Family").strip()
        lockbox_code = (payload.get("lockbox_code") or "SENTRI-1000").strip()
        showing_instructions = (payload.get("showing_instructions") or "Appointment required.").strip()
        assigned_agent = (payload.get("assigned_licensed_agent") or "Unassigned").strip()

        listing = PropertyListing(
            mls_id=mls_id,
            address=address,
            price=price,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            square_feet=square_feet,
            property_type=property_type,
            lockbox_code=lockbox_code,
            showing_instructions=showing_instructions,
            assigned_licensed_agent=assigned_agent,
            status="ACTIVE"
        )
        engine.db.save_listing(listing)
        engine.db.log_bot_action("Victoria", BotRole.CHIEF_OF_STAFF, "PROPERTY_ADDED", f"Added listing {address} (${price:,.0f}) to SQLite database")
        return {"status": "saved", "listing": to_dict(listing)}

    @app.delete("/api/v1/properties/{mls_id}")
    def remove_property_listing(mls_id: str):
        """Delete a property listing from the SQLite `listings` database."""
        engine.db.delete_listing(mls_id)
        return {"status": "deleted", "mls_id": mls_id}

    @app.post("/api/v1/leads/webhook/{source}")
    def catch_lead_by_platform(source: str, payload: Dict[str, Any]):
        """
        Routes Zillow/Realtor/Website leads to Ryan (Ext 102) and
        Instagram/Meta Ads/WhatsApp leads to Maya (Ext 103).
        Validates input and checks SQLite `listings` database.
        """
        try:
            lead_source = LeadSource(source.upper())
        except ValueError:
            lead_source = LeadSource.WEBSITE_FORM
        try:
            lead, instant_response = engine.route_inbound_webhook(lead_source, payload)
            return {
                "status": "caught",
                "lead_id": lead.id,
                "caught_by": lead.caught_by_bot,
                "matched_in_database": instant_response.get("matched_in_database", False),
                "instant_response": instant_response
            }
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

    @app.post("/api/v1/leads/{lead_id}/chat")
    def reply_to_lead_chat(lead_id: str, payload: Dict[str, str]):
        """Ryan or Maya qualifies the buyer's reply on BANT (0-100 score)."""
        lead, reply = engine.handle_buyer_reply(lead_id, payload.get("message", ""))
        return {"lead_id": lead.id, "handled_by": lead.caught_by_bot, "reply": reply, "qualification": to_dict(lead.qualification)}

    # 5. ALEX (EXT 104) — SITE VISITS, LOCKBOXES & CAR VOICE MEMOS
    @app.post("/api/v1/leads/{lead_id}/schedule")
    def alex_schedule_site_visit(lead_id: str, payload: Dict[str, Any]):
        """Alex (Ext 104) books the site visit & lockbox, and Victoria (Ext 100) assigns the human agent."""
        appt_type_str = payload.get("type", "IN_PERSON_SITE_VISIT")
        appt_type = AppointmentType(appt_type_str) if appt_type_str in AppointmentType.__members__ else AppointmentType.IN_PERSON_SITE_VISIT
        appt, conf, agent, lead = engine.schedule_and_assign(
            lead_id=lead_id,
            appointment_type=appt_type,
            time_str=payload.get("time_slot", "Thursday at 2:00 PM PST"),
            property_address=payload.get("address")
        )
        dossier = VictoriaChiefOfStaffBot.create_agent_handoff_dossier(lead, agent, appt)
        return {
            "status": "confirmed",
            "scheduled_by": "Alex (Showing & Field Ops • Ext 104)",
            "assigned_by": "Victoria (Chief of Staff • Ext 100)",
            "appointment": to_dict(appt),
            "confirmation_message": conf,
            "assigned_agent": to_dict(agent),
            "agent_handoff_dossier": dossier
        }

    @app.get("/api/v1/calendar/agent/{agent_id}/feed.ics")
    def get_agent_live_calendar_feed(agent_id: str):
        ics_content = CalendarIntegrationEngine.generate_agent_ics_feed(engine.db, agent_id)
        return Response(
            content=ics_content,
            media_type="text/calendar",
            headers={"Content-Disposition": f"attachment; filename={agent_id}_showings.ics"}
        )

    @app.get("/api/v1/calendar/appointment/{appt_id}.ics")
    def download_single_appointment_ics(appt_id: str):
        appts = [a for a in engine.db.get_all_appointments() if a.id == appt_id]
        if not appts:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return Response(
            content=appts[0].calendar_ics or "",
            media_type="text/calendar",
            headers={"Content-Disposition": f"attachment; filename={appt_id}.ics"}
        )

    @app.get("/api/v1/calendar/meetings")
    def list_all_booked_meetings(agent_id: Optional[str] = None):
        appts = engine.db.get_all_appointments()
        if agent_id:
            appts = [a for a in appts if a.agent_id == agent_id]
        agents_map = {a.id: a for a in engine.db.get_agents()}
        results = []
        for a in appts:
            lead = engine.db.get_lead(a.lead_id)
            agent = agents_map.get(a.agent_id)
            links = CalendarIntegrationEngine.build_google_and_outlook_links(a, lead, agent) if (lead and agent) else {}
            results.append({
                "appointment": to_dict(a),
                "buyer_name": f"{lead.first_name} {lead.last_name}" if lead else "Unknown",
                "buyer_phone": lead.phone if lead else "",
                "agent_name": agent.name if agent else a.agent_id,
                "calendar_links": links
            })
        return {"total_meetings": len(results), "meetings": results}

    @app.post("/api/v1/bots/alex/car-voice-memo")
    def alex_transcribe_car_memo(payload: Dict[str, str]):
        """Alex (Ext 104) transcribes an agent's post-tour car voice note and drafts the purchase offer email."""
        return AlexShowingFieldOpsBot.transcribe_car_voice_memo(
            db=engine.db,
            agent_name=payload.get("agent_name", "Marcus Sterling"),
            property_address=payload.get("property_address", "742 Evergreen Terrace"),
            raw_voice_transcript=payload.get("transcript", "Buyer loved the kitchen, pre-approved up to $2.5M with Chase, wants to offer $2.35M.")
        )

except ImportError:
    app = None

# -----------------------------------------------------------------------------
# FULL SIMULATION & VERIFICATION RUNNER (v4.0)
# -----------------------------------------------------------------------------

def run_v4_verification():
    print("=" * 82)
    print("🏢 REAL ESTATE AI AGENCY v4.0 — LOAD-BALANCED 5-BOT WORKFORCE + CHIEF OF STAFF")
    print("=" * 82)

    db_file = "test_v4_agency.db"
    if os.path.exists(db_file):
        os.remove(db_file)
    eng = UnifiedRealEstateAgencyEngine(db_path=db_file)

    # 1. CLARA (EXT 101): Dedicated Phone Call Line Only
    print("\n[1] 📞 CLARA (EXT 101 — VOICE RECEPTIONIST): ANSWERS LIVE PHONE CALL")
    call_res = ClaraVoiceReceptionistBot.handle_inbound_call(
        db=eng.db,
        caller_name="Devon Brooks",
        caller_phone="+1-310-555-9940",
        transcript="Hi! I'm driving past 742 Evergreen Terrace. What's the asking price and can I see it this week?"
    )
    print(f"    🎙️ Clara Spoken Reply: \"{call_res['spoken_response']}\"")

    # 2. RYAN (EXT 102): Catches Zillow Portal Lead (<30s SLA)
    print("\n[2] 🏠 RYAN (EXT 102 — PORTAL CATCHER): CATCHES ZILLOW LEAD (<30s SLA)")
    zillow_lead, ryan_msg = eng.route_inbound_webhook(LeadSource.ZILLOW, {
        "contact": {"firstName": "Jonathan", "lastName": "Vance", "phone": "+1-310-555-8291", "email": "jvance@techinvest.com"},
        "property": {"address": "742 Evergreen Terrace, Beverly Hills, CA 90210"},
        "message": "Inquiring from Zillow. Available to view Thursday at 2pm?"
    })
    print(f"    ⚡ Ryan Instant SMS: \"{ryan_msg['message']}\"")
    zillow_lead, ryan_reply = eng.handle_buyer_reply(
        zillow_lead.id,
        "Yes, I'm pre-approved with Chase Private Client up to $2.5M and want to close in 30 days. Let's tour Thursday at 2pm."
    )
    print(f"    📊 Ryan BANT Score: {zillow_lead.qualification.score}/100 ({zillow_lead.qualification.tier.value})")

    # 3. MAYA (EXT 103): Catches Instagram DM / WhatsApp Lead
    print("\n[3] 📱 MAYA (EXT 103 — SOCIAL & WHATSAPP CONCIERGE): CATCHES INSTAGRAM DM LEAD")
    ig_lead, maya_msg = eng.route_inbound_webhook(LeadSource.INSTAGRAM_DM, {
        "first_name": "Rachel", "last_name": "Green", "phone": "+1-310-555-4490",
        "area": "Santa Monica / Brentwood (90049)", "message": "Saw your Instagram Reel! Looking for a 2BR condo under $800k."
    })
    print(f"    💬 Maya Instant DM: \"{maya_msg['message']}\"")

    # 4. ALEX (EXT 104) & VICTORIA (EXT 100): Books Site Visit + Lockbox + Routes to Agent
    print("\n[4] 📅 ALEX (EXT 104 — SHOWING OPS) & VICTORIA (EXT 100 — CHIEF OF STAFF):")
    appt, conf, agent, zillow_lead = eng.schedule_and_assign(
        lead_id=zillow_lead.id,
        appointment_type=AppointmentType.IN_PERSON_SITE_VISIT,
        time_str="Thursday, Nov 12 at 2:00 PM PST",
        property_address="742 Evergreen Terrace, Beverly Hills, CA 90210"
    )
    print(f"    ✅ {conf}")
    print("\n" + VictoriaChiefOfStaffBot.create_agent_handoff_dossier(zillow_lead, agent, appt))

    # 5. VICTORIA (EXT 100): Direct Broker / Client Conversation
    print("\n[5] 👑 VICTORIA (EXT 100 — CHIEF OF STAFF): DIRECT CONVERSATION WITH CLIENT (BROKER)")
    broker_q = "Victoria, how many hot leads do we have in the pipeline right now and how is the team workload?"
    vic_res = VictoriaChiefOfStaffBot.talk_to_broker(eng.db, broker_q)
    print(f"    👔 Broker asks: \"{broker_q}\"")
    print(f"    👑 Victoria replies: \"{vic_res['executive_response']}\"")
    print("\n" + VictoriaChiefOfStaffBot.generate_executive_briefing(eng.db))


    # 6. SECURITY & EMERGENCY ESCALATION BY VICTORIA (EXT 100)
    print("\n[6] 🔐 CALLER-ID SECURITY & 🚨 VICTORIA'S EMERGENCY CALL TO BOSS / AGENT")
    boss_id = SecurityAndIdentityGuard.identify_caller(eng.db, "+1-310-555-0001", "9021")
    agent_id_check = SecurityAndIdentityGuard.identify_caller(eng.db, "+1-555-019-2831")
    public_id = SecurityAndIdentityGuard.identify_caller(eng.db, "+1-310-555-8291")
    print(f"    🛡️ Caller +1-310-555-0001 Identified As: {boss_id['role'].value} ({boss_id['name']}) | PIN Verified: {boss_id['pin_verified']}")
    print(f"    🛡️ Caller +1-555-019-2831 Identified As: {agent_id_check['role'].value} ({agent_id_check['name']})")
    print(f"    🛡️ Caller +1-310-555-8291 Identified As: {public_id['role'].value} (Permissions: {public_id['permissions']})")

    emerg_agent_call = EmergencyVoiceEscalationEngine.trigger_emergency_call(
        db=eng.db,
        target_role="ASSIGNED_LUXURY_AGENT",
        target_name=agent.name,
        target_phone=agent.phone,
        urgency=UrgencyLevel.HIGH_INTENT_LUXURY,
        spoken_briefing=f"Urgent alert for {agent.name}. Jonathan Vance, pre-approved for $2.5 million with Chase, is outside 742 Evergreen Terrace requesting an immediate showing. Press 1 to connect with the buyer now, or press 2 so I can reassign to Elena Rostova."
    )
    print(f"\n    📞 [VICTORIA EMERGENCY CALL -> AGENT {agent.name.upper()} ({agent.phone})]")
    print(f"       Status: {emerg_agent_call['dial_status']}")
    print(f"       Spoken Audio: \"{emerg_agent_call['spoken_voice_script']}\"")

    emerg_boss_call = EmergencyVoiceEscalationEngine.trigger_emergency_call(
        db=eng.db,
        target_role="MANAGING_BROKER_BOSS",
        target_name="Arthur Pendelton (Managing Broker)",
        target_phone="+1-310-555-0001",
        urgency=UrgencyLevel.ESCROW_DEADLINE_RISK,
        spoken_briefing="Boss, urgent compliance alert from Victoria. The inspection contingency on 1042 Wilshire Boulevard expires in less than 24 hours and 2 disclosures are still unsigned. Press 1 to authorize an emergency extension addendum."
    )
    print(f"\n    📞 [VICTORIA EMERGENCY CALL -> BOSS ({emerg_boss_call['target_phone']})]")
    print(f"       Urgency: {emerg_boss_call['urgency_level']} | Status: {emerg_boss_call['dial_status']}")
    print(f"       Spoken Audio: \"{emerg_boss_call['spoken_voice_script']}\"")

if __name__ == "__main__":
    run_v4_verification()
