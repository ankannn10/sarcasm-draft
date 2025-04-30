from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session
)
import os
import json
import pandas as pd
import traceback
from driver import main as run_pipeline
from auth import init_db, register_user, validate_user

app = Flask(__name__)
app.secret_key = "YOUR-SECRET-KEY"  # Replace with a secure key in production

# Initialize the user database
init_db()

def assign_sentiment(top_emotion, confidence):
    """Assign sentiment based on emotion and confidence."""
    if confidence < 0.5:
        return "Neutral"
    return "Negative" if top_emotion in ["Anger", "Sadness", "Fear"] else "Positive"

def ratio_as_percentage(part, whole):
    """Calculate part/whole as a percentage."""
    return round((part / whole) * 100, 2) if whole else 0.0

def login_required(f):
    """Decorator to enforce login for protected routes."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Handle user signup."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            return render_template("signup.html", error="Please fill in all fields.")

        if register_user(username, password):
            return redirect(url_for("login"))
        return render_template("signup.html", error="Username is already taken.")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if validate_user(username, password):
            session["username"] = username
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")

@app.route("/logout")
def logout():
    """Log out the user and clear the session."""
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/", methods=["GET"])
@login_required
def index():
    """Render the main index page."""
    return render_template("index.html", username=session.get("username"))

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    """
    Process and analyze YouTube video data.
    Now the pipeline produces a final CSV (final_predictions.csv)
    that contains both emotion predictions and irony (sarcasm) outputs.
    """
    try:
        youtube_link = request.form.get("yt_link", "").strip()
        if not youtube_link:
            return jsonify({"error": "No YouTube link provided"}), 400

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        run_pipeline(youtube_link, output_dir, cleanup=False)

        # Instead of emotion_predictions.csv, we now load final_predictions.csv.
        final_csv_path = os.path.join(output_dir, "final_predictions.csv")
        if not os.path.exists(final_csv_path):
            return jsonify({"error": "final_predictions.csv not found"}), 500

        df_final = pd.read_csv(final_csv_path)
        required_cols = ["comment", "comment_emotion1", "comment_conf1", "Irony", "Irony Probability"]
        if not all(col in df_final.columns for col in required_cols):
            return jsonify(
                {"error": f"Missing columns: {required_cols} in final_predictions.csv"}
            ), 500

        # Assign sentiment for each comment based on emotion predictions
        df_final["sentiment"] = df_final.apply(
            lambda row: assign_sentiment(row["comment_emotion1"], row["comment_conf1"]),
            axis=1
        )
        report_data = df_final.to_dict(orient="records")

        # Determine overall sentiment and emotion for all comments
        overall_sentiment = "N/A"
        overall_emotion = "N/A"
        if not df_final.empty:
            sentiment_counts = df_final["sentiment"].value_counts()
            if not sentiment_counts.empty:
                overall_sentiment = sentiment_counts.idxmax()

            emotion_counts = df_final["comment_emotion1"].value_counts()
            if not emotion_counts.empty:
                overall_emotion = emotion_counts.idxmax()

        # Video metadata & engagement metrics
        video_json_path = os.path.join(output_dir, "video_data.json")
        title, channel_name, upload_date, tags = "", "", "", []
        likes_to_views_perc = comments_to_views_perc = views_to_subs_perc = 0.0

        if os.path.exists(video_json_path):
            with open(video_json_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)

            title = video_data.get("title", "No Title")
            channel_name = video_data.get("channel_name", "Unknown Channel")
            tags = video_data.get("tags", [])
            views = video_data.get("views", 0)
            likes = video_data.get("likes", 0)
            upload_date = video_data.get("upload_date", "Unknown Date")
            comment_count = video_data.get("comment_count", 0)
            subscriber_count = video_data.get("subscriber_count", 0)

            likes_to_views_perc = ratio_as_percentage(likes, views)
            comments_to_views_perc = ratio_as_percentage(comment_count, views)
            views_to_subs_perc = ratio_as_percentage(views, subscriber_count)
        else:
            print("video_data.json not found. Skipping engagement metrics.")

        sentiment_counts = df_final["sentiment"].value_counts().to_dict()
        emotion_counts = df_final["comment_emotion1"].value_counts().to_dict()

        # Pass along the Irony columns for visualization
        return render_template(
            "results.html",
            video_link=youtube_link,
            report_data=report_data,
            overall_sentiment=overall_sentiment,
            overall_emotion=overall_emotion,
            video_title=title,
            channel_name=channel_name,
            tags=tags,
            upload_date=upload_date,
            total_views=views,
            total_likes=likes,
            total_comments=comment_count,
            total_subscribers=subscriber_count,
            likes_to_views_perc=likes_to_views_perc,
            comments_to_views_perc=comments_to_views_perc,
            views_to_subs_perc=views_to_subs_perc,
            sentiment_distribution=sentiment_counts,
            emotion_distribution=emotion_counts,
            # Additional fields for irony visualization:
            # Each record in report_data will include "Irony" and "Irony Probability"
            username=session.get("username")
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
