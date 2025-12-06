import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    
    Response,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from detection import pipeline


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"

for directory in (TEMPLATES_DIR, STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "replace-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    detections = db.relationship("DetectionResult", backref="user", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    input_type = db.Column(db.String(20), nullable=False)
    input_filename = db.Column(db.String(255), nullable=True)
    output_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    average_fps = db.Column(db.Float, nullable=True)
    average_inference = db.Column(db.Float, nullable=True)
    elapsed_time = db.Column(db.Float, nullable=True)


@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    return User.query.get(int(user_id))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Username or email already registered.", "danger")
            return redirect(url_for("register"))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Registration successful. You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        ALLOWED_VIDEO_EXTENSIONS=sorted(ALLOWED_VIDEO_EXTENSIONS),
        ALLOWED_IMAGE_EXTENSIONS=sorted(ALLOWED_IMAGE_EXTENSIONS),
    )


def _save_detection(
    input_type: str,
    output_filename: str,
    stats: Optional[dict] = None,
    input_filename: Optional[str] = None,
) -> None:
    output_relative = Path("outputs") / output_filename
    avg_fps = None
    avg_inference = None
    elapsed = None

    if stats:
        avg_fps = stats.get("average_fps") or stats.get("fps")
        avg_inference = stats.get("average_inference_time") or stats.get("inference_time")
        elapsed = stats.get("elapsed_time")

    result = DetectionResult(
        user_id=current_user.id,
        input_type=input_type,
        input_filename=input_filename,
        output_path=output_relative.as_posix(),
        average_fps=avg_fps,
        average_inference=avg_inference,
        elapsed_time=elapsed,
    )
    db.session.add(result)
    db.session.commit()


@app.route("/upload_image", methods=["POST"])
@login_required
def upload_image():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please select an image file.", "danger")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        flash("Unsupported image format.", "danger")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    input_path = UPLOAD_DIR / unique_name
    file.save(input_path)

    output_filename = f"{uuid.uuid4().hex}_processed.png"
    output_path = OUTPUT_DIR / output_filename

    try:
        stats = pipeline.process_image(str(input_path), str(output_path))
    except Exception as exc:  # pragma: no cover - runtime errors surfaced to user
        flash(f"Processing failed: {exc}", "danger")
        return redirect(url_for("dashboard"))

    _save_detection(
        input_type="image",
        output_filename=output_filename,
        stats=stats,
        input_filename=filename,
    )

    flash("Image processed successfully.", "success")
    return redirect(url_for("results"))


@app.route("/upload_video", methods=["POST"])
@login_required
def upload_video():
    file = request.files.get("video")
    if not file or file.filename == "":
        flash("Please select a video file.", "danger")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        flash("Unsupported video format.", "danger")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    input_path = UPLOAD_DIR / unique_name
    file.save(input_path)

    output_filename = f"{uuid.uuid4().hex}_processed.mp4"
    output_path = OUTPUT_DIR / output_filename

    try:
        stats = pipeline.process_video(str(input_path), str(output_path))
    except Exception as exc:  # pragma: no cover
        flash(f"Processing failed: {exc}", "danger")
        return redirect(url_for("dashboard"))

    _save_detection(
        input_type="video",
        output_filename=output_filename,
        stats=stats,
        input_filename=filename,
    )

    flash("Video processed successfully.", "success")
    return redirect(url_for("results"))


@app.route("/start_live_capture", methods=["POST"])
@login_required
def start_live_capture():
    output_filename = f"{uuid.uuid4().hex}_live.mp4"
    output_path = OUTPUT_DIR / output_filename

    try:
        stats = pipeline.process_video(0, str(output_path), limit_seconds=15)
    except Exception as exc:  # pragma: no cover
        flash(f"Live capture failed: {exc}", "danger")
        return redirect(url_for("live_camera"))

    _save_detection(
        input_type="live",
        output_filename=output_filename,
        stats=stats,
        input_filename="webcam",
    )

    flash("Live capture completed and saved.", "success")
    return redirect(url_for("results"))


@app.route("/live_camera")
@login_required
def live_camera():
    return render_template("live.html")


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(pipeline.live_frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/results")
@login_required
def results():
    user_results = (
        DetectionResult.query.filter_by(user_id=current_user.id).order_by(DetectionResult.created_at.desc()).all()
    )
    return render_template("results.html", results=user_results)


@app.route("/delete_result/<int:result_id>", methods=["POST"])
@login_required
def delete_result(result_id: int):
    result = DetectionResult.query.filter_by(id=result_id, user_id=current_user.id).first()
    if not result:
        flash("Result not found.", "warning")
        return redirect(url_for("results"))

    output_full_path = STATIC_DIR / result.output_path
    if output_full_path.exists():
        output_full_path.unlink()

    db.session.delete(result)
    db.session.commit()
    flash("Result deleted.", "info")
    return redirect(url_for("results"))


@app.route("/download/<path:filename>")
@login_required
def download_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)


