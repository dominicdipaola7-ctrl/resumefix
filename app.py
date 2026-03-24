import os
import json
import hashlib
import re
from datetime import datetime, date, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from dotenv import load_dotenv
import stripe
import openai
import PyPDF2
from fpdf import FPDF
import io
import base64

load_dotenv()

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///resumeai.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

# Mail config
app.config["MAIL_SERVER"]   = os.getenv("MAIL_SERVER", "smtp.gmail.com")
app.config["MAIL_PORT"]     = int(os.getenv("MAIL_PORT", 587))
app.config["MAIL_USE_TLS"]  = os.getenv("MAIL_USE_TLS", "True") == "True"
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD", "")

db   = SQLAlchemy(app)
mail = Mail(app)

# Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUB_KEY       = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID      = os.getenv("STRIPE_PRO_PRICE_ID", "")

# OpenAI
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", 1))

# ── Models ───────────────────────────────────────────────────────────────────
class FreeUsage(db.Model):
    __tablename__ = "free_usage"
    id         = db.Column(db.Integer, primary_key=True)
    ip_hash    = db.Column(db.String(64), nullable=False, index=True)
    usage_date = db.Column(db.Date, nullable=False, default=date.today)
    count      = db.Column(db.Integer, default=0)

class ProUser(db.Model):
    __tablename__ = "pro_users"
    id                  = db.Column(db.Integer, primary_key=True)
    email               = db.Column(db.String(255), unique=True, nullable=False, index=True)
    stripe_customer_id  = db.Column(db.String(120), unique=True)
    stripe_subscription_id = db.Column(db.String(120), unique=True)
    subscription_status = db.Column(db.String(50), default="inactive")
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at          = db.Column(db.DateTime)

class ResumeJob(db.Model):
    __tablename__ = "resume_jobs"
    id             = db.Column(db.Integer, primary_key=True)
    ip_hash        = db.Column(db.String(64))
    pro_email      = db.Column(db.String(255))
    match_score    = db.Column(db.Integer)
    keywords_found = db.Column(db.Text)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_ip_hash():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if ip and "," in ip:
        ip = ip.split(",")[0].strip()
    return hashlib.sha256(ip.encode()).hexdigest()

def check_free_limit():
    """Returns (allowed: bool, remaining: int)"""
    ip_hash = get_ip_hash()
    today   = date.today()
    record  = FreeUsage.query.filter_by(ip_hash=ip_hash, usage_date=today).first()
    if not record:
        return True, FREE_DAILY_LIMIT
    remaining = FREE_DAILY_LIMIT - record.count
    return remaining > 0, max(0, remaining)

def increment_free_usage():
    ip_hash = get_ip_hash()
    today   = date.today()
    record  = FreeUsage.query.filter_by(ip_hash=ip_hash, usage_date=today).first()
    if not record:
        record = FreeUsage(ip_hash=ip_hash, usage_date=today, count=0)
        db.session.add(record)
    record.count += 1
    db.session.commit()

def is_pro_user(email):
    if not email:
        return False
    user = ProUser.query.filter_by(email=email.lower()).first()
    if not user:
        return False
    if user.subscription_status == "active":
        return True
    if user.expires_at and user.expires_at > datetime.utcnow():
        return True
    return False

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text   = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()

def call_openai(system_prompt, user_content, temperature=0.4):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=temperature,
        max_tokens=4000,
    )
    return response.choices[0].message.content.strip()

def analyze_resume(resume_text, job_desc, is_pro=False, include_cover=False):
    """Main AI analysis – returns structured dict."""

    system_prompt = """You are a world-class resume strategist, ATS expert, and career coach.
Your job is to analyze a resume against a job description and return ONLY valid JSON.
All keys and string values must use double quotes. Do not include markdown fences."""

    ats_instruction = ""
    if is_pro:
        ats_instruction = """
Also include:
- "ats_optimized_resume": A fully rewritten ATS-friendly version of the resume using standard section headers (Summary, Experience, Education, Skills), bullet points starting with action verbs, and incorporating keywords naturally. Preserve all factual content.
- "ats_tips": Array of 5 specific ATS optimization tips for this resume.
"""

    cover_instruction = ""
    if include_cover and is_pro:
        cover_instruction = """
Also include:
- "cover_letter": A professional, personalized cover letter (3-4 paragraphs) written in first person, tailored specifically to this job description. Include [Company Name] as a placeholder where needed.
"""

    user_content = f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{job_desc}

Return a JSON object with exactly these fields:
- "match_score": integer 0-100 representing how well the resume matches the job
- "score_breakdown": object with keys "skills_match" (0-100), "experience_match" (0-100), "keyword_density" (0-100), "formatting" (0-100)
- "rewritten_resume": A polished, improved version of the resume tailored to this job. Keep all true facts, restructure and reword to match the job requirements. Use clear sections and bullet points.
- "keywords_found": array of important keywords from the job description that ARE in the resume
- "keywords_missing": array of important keywords from the job description that are NOT in the resume
- "strengths": array of 3-4 specific strengths of this resume for this job
- "improvements": array of 4-5 specific actionable improvement suggestions
- "summary_recommendation": 2-3 sentence executive summary of fit and top recommendation
{ats_instruction}
{cover_instruction}

Respond with ONLY the JSON object. No explanation, no markdown.
"""

    raw = call_openai(system_prompt, user_content, temperature=0.3)

    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt a second, stricter parse request
        retry_prompt = "Return ONLY the JSON object, no other text, no markdown, no explanation."
        raw2 = call_openai(system_prompt, user_content + "\n\n" + retry_prompt, temperature=0.1)
        raw2 = re.sub(r"^```(?:json)?", "", raw2.strip())
        raw2 = re.sub(r"```$", "", raw2.strip()).strip()
        result = json.loads(raw2)

    return result

def generate_pdf_resume(resume_text, title="Rewritten Resume"):
    """Generates a simple PDF from text and returns base64 string."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    # Body
    pdf.set_font("Helvetica", "", 10)
    lines = resume_text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
        # Detect section headers (ALL CAPS or ending with :)
        if (line.isupper() and len(line) > 2) or (line.endswith(":") and len(line) < 40):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(240, 245, 255)
            pdf.cell(0, 8, line, ln=True, fill=True)
            pdf.set_font("Helvetica", "", 10)
        elif line.startswith("•") or line.startswith("-"):
            pdf.set_x(25)
            pdf.multi_cell(0, 6, line)
        else:
            pdf.multi_cell(0, 6, line)

    pdf_bytes = bytes(pdf.output())
    return base64.b64encode(pdf_bytes).decode("utf-8")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    pro_email  = session.get("pro_email", "")
    pro_active = is_pro_user(pro_email)
    _, free_remaining = check_free_limit()
    return render_template(
        "index.html",
        stripe_pub_key=STRIPE_PUB_KEY,
        pro_active=pro_active,
        pro_email=pro_email,
        free_remaining=free_remaining,
        free_limit=FREE_DAILY_LIMIT,
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    pro_email  = session.get("pro_email", "")
    pro_active = is_pro_user(pro_email)

    # Rate limiting for free tier
    if not pro_active:
        allowed, remaining = check_free_limit()
        if not allowed:
            return jsonify({
                "error": "free_limit_reached",
                "message": "You've used your free resume today. Upgrade to Pro for unlimited rewrites!",
            }), 429

    # Validate inputs
    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided."}), 400
    resume_file = request.files["resume"]
    job_desc    = request.form.get("job_description", "").strip()
    include_cover = request.form.get("include_cover", "false").lower() == "true"

    if not resume_file or resume_file.filename == "":
        return jsonify({"error": "Please upload a PDF resume."}), 400
    if not job_desc:
        return jsonify({"error": "Please paste a job description."}), 400
    if len(job_desc) < 50:
        return jsonify({"error": "Job description is too short. Please paste the full description."}), 400
    if not resume_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Extract text
    file_bytes = resume_file.read()
    if len(file_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not read PDF: {str(e)}"}), 400

    if len(resume_text) < 100:
        return jsonify({"error": "Could not extract enough text from your PDF. Please ensure it's a text-based PDF (not scanned image)."}), 400

    # Limit job_desc to 3000 chars to save tokens
    job_desc_trimmed = job_desc[:3000]

    # AI analysis
    try:
        result = analyze_resume(
            resume_text,
            job_desc_trimmed,
            is_pro=pro_active,
            include_cover=include_cover and pro_active,
        )
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except openai.RateLimitError:
        return jsonify({"error": "AI service is busy. Please try again in a moment."}), 503
    except openai.AuthenticationError:
        return jsonify({"error": "AI service configuration error. Please contact support."}), 500
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    # Generate PDF
    try:
        pdf_b64 = generate_pdf_resume(
            result.get("rewritten_resume", resume_text),
            title="AI-Optimized Resume"
        )
        result["pdf_b64"] = pdf_b64
    except Exception:
        result["pdf_b64"] = None

    # Generate ATS PDF if pro
    if pro_active and result.get("ats_optimized_resume"):
        try:
            ats_pdf_b64 = generate_pdf_resume(
                result["ats_optimized_resume"],
                title="ATS-Optimized Resume"
            )
            result["ats_pdf_b64"] = ats_pdf_b64
        except Exception:
            result["ats_pdf_b64"] = None

    # Log usage
    try:
        if not pro_active:
            increment_free_usage()
        rj = ResumeJob(
            ip_hash=get_ip_hash(),
            pro_email=pro_email if pro_active else None,
            match_score=result.get("match_score"),
            keywords_found=json.dumps(result.get("keywords_found", [])),
        )
        db.session.add(rj)
        db.session.commit()
    except Exception:
        pass  # Don't fail the request over logging

    result["is_pro"] = pro_active
    _, remaining = check_free_limit()
    result["free_remaining"] = remaining

    return jsonify(result)

# ── Stripe / Billing ──────────────────────────────────────────────────────────
@app.route("/create-checkout", methods=["POST"])
def create_checkout():
    data  = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "Valid email required."}), 400
    if not STRIPE_PRICE_ID:
        return jsonify({"error": "Billing not configured."}), 500

    try:
        checkout = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=email,
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=os.getenv("APP_URL", request.host_url.rstrip("/"))
                        + "/success?session_id={CHECKOUT_SESSION_ID}&email=" + email,
            cancel_url=os.getenv("APP_URL", request.host_url.rstrip("/")) + "/?canceled=true",
            metadata={"email": email},
            subscription_data={"metadata": {"email": email}},
        )
        return jsonify({"url": checkout.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/success")
def success():
    session_id = request.args.get("session_id", "")
    email      = request.args.get("email", "").strip().lower()

    if session_id and email:
        try:
            checkout = stripe.checkout.Session.retrieve(session_id)
            if checkout.payment_status == "paid":
                _activate_pro(email, checkout.customer, checkout.subscription)
                session["pro_email"] = email
        except Exception:
            pass

    return redirect(url_for("index") + "?pro=activated")

def _activate_pro(email, customer_id, subscription_id):
    user = ProUser.query.filter_by(email=email).first()
    if not user:
        user = ProUser(email=email)
        db.session.add(user)
    user.stripe_customer_id     = customer_id
    user.stripe_subscription_id = subscription_id
    user.subscription_status    = "active"
    user.expires_at             = datetime.utcnow() + timedelta(days=35)
    db.session.commit()

@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload   = request.data
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return jsonify({"error": "Invalid signature"}), 400

    if event["type"] == "checkout.session.completed":
        obj   = event["data"]["object"]
        email = obj.get("metadata", {}).get("email") or obj.get("customer_email", "")
        if email:
            _activate_pro(email.lower(), obj.get("customer"), obj.get("subscription"))

    elif event["type"] in ("customer.subscription.updated", "invoice.paid"):
        obj   = event["data"]["object"]
        sub_id = obj.get("id") or obj.get("subscription")
        if sub_id:
            user = ProUser.query.filter_by(stripe_subscription_id=sub_id).first()
            if user:
                user.subscription_status = "active"
                user.expires_at = datetime.utcnow() + timedelta(days=35)
                db.session.commit()

    elif event["type"] in ("customer.subscription.deleted", "customer.subscription.paused"):
        obj = event["data"]["object"]
        user = ProUser.query.filter_by(stripe_subscription_id=obj.get("id")).first()
        if user:
            user.subscription_status = "inactive"
            db.session.commit()

    return jsonify({"status": "ok"})

@app.route("/pro-login", methods=["POST"])
def pro_login():
    data  = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email required."}), 400
    if is_pro_user(email):
        session["pro_email"] = email
        return jsonify({"success": True, "email": email})
    return jsonify({"error": "No active Pro subscription found for this email."}), 404

@app.route("/pro-logout", methods=["POST"])
def pro_logout():
    session.pop("pro_email", None)
    return jsonify({"success": True})

@app.route("/status")
def status():
    pro_email  = session.get("pro_email", "")
    pro_active = is_pro_user(pro_email)
    _, free_remaining = check_free_limit()
    return jsonify({
        "pro_active":     pro_active,
        "pro_email":      pro_email if pro_active else "",
        "free_remaining": free_remaining,
        "free_limit":     FREE_DAILY_LIMIT,
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 5 MB."}), 413

@app.errorhandler(429)
def rate_limit_error(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

if __name__ == "__main__":
    app.run(debug=os.getenv("DEBUG", "False") == "True", port=5000)
