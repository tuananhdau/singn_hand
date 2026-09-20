from flask import Flask, render_template, Response

from camera import generate_frames


app = Flask(__name__)


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/camera")
def camera():

    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )