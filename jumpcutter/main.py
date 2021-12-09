import os
import pathlib
import threading
import time
import uuid

from flask import Flask, redirect, render_template, request

from jumpcutter import DiscordHook, JumpcutterDriver, JumpcutterParams

app = Flask(__name__)
INPUT_DIR = pathlib.Path("./input")
DRIVERS = []


@app.get("/")
def landing_page():
    return render_template(
        "home.html",
        files=get_input_files(),
        jobs=[get_job(*x) for x in DRIVERS]
    )


@app.get("/enqueue")
def enqueue_page():
    return render_template(
        "enqueue.html",
        info=get_file_info(INPUT_DIR / request.args.get("path")),
        courses=[x.strip() for x in os.environ.get("COURSE_NAMES", "").split(",")]
    )


@app.post("/enqueue")
def enqueue_job():
    file = INPUT_DIR / request.form.get("path")
    if not file.is_file():
        return display_error(f"the selected file '{request.form.get('path')}' doesn't exist anymore")
    params = JumpcutterParams(
        threshold=float(request.form.get("param-threshold")),
        silent_speed=float(request.form.get("param-speed-silent")),
        sounded_speed=float(request.form.get("param-speed-sounded")),
        frame_rate=float(request.form.get("param-fps")),
        sample_rate=int(request.form.get("param-sps"))
    )
    if not request.form.get("course"):
        return display_error("no course was selected :(")
    driver = JumpcutterDriver(str(file), params)
    driver.progress_hooks.append(DiscordHook(
        webhook_url=os.environ.get("DISCORD_WEBHOOK"),
        title=file.name,
        course=request.form.get("course")
    ))
    driver.module_dir = request.form.get("course")
    DRIVERS.append((
        {
            "id": str(uuid.uuid4()),
            "course": request.form.get("course"),
            "work_until": -1
        }, driver)
    )
    return redirect("/")


@app.post("/set_until")
def approve_until():
    for info, driver in DRIVERS:
        if info["id"] == request.form.get("id"):
            break
    else:
        return display_error(f"job with id '{request.form.get('id')}' not found")
    info["work_until"] = int(request.form.get("number"))
    return redirect("/")


def get_input_files():
    candiates = []
    for ext in "mkv", "mp4":
        candiates.extend(INPUT_DIR.glob(f"**/*.{ext}"))
    return [get_file_info(x) for x in candiates]


def get_file_info(path):
    return {
        "name": path.name,
        "path": str(path).removeprefix(str(INPUT_DIR)).replace("\\", "/").strip("/")
    }


# noinspection PyProtectedMember
def get_job(info, driver: JumpcutterDriver):
    return {
        "id": info["id"],
        "course": info["course"],
        "current_step": driver.current_job,
        "file": get_file_info(pathlib.Path(driver._input_file)),
        "work_until": info["work_until"]
    }


def display_error(message):
    return f"""
    <script type="application/javascript">
        alert("{message}");
        location.href = "/";
    </script>
    """


def worker_func():
    while True:
        for info, driver in DRIVERS:
            driver: JumpcutterDriver
            if driver.done:
                DRIVERS.remove((info, driver))
            if info["work_until"] > driver.current_job:
                driver.do_work()
        time.sleep(1)


INPUT_DIR.mkdir(exist_ok=True, parents=True)
threading.Thread(target=worker_func, daemon=True).start()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
