import flask

from . import main


@main.route("/")
def index():
    return flask.render_template("main/index.html")


@main.route("/about")
def about():
    return flask.render_template("main/about.html")


@main.route("/faq")
def faq():
    return flask.render_template("main/faq.html")


@main.route("/features")
def features():
    return flask.render_template("main/features.html")


@main.route("/terms-and-conditions")
def terms_and_conditions():
    return flask.render_template("main/terms_and_conditions.html")


@main.route("/privacy-policy")
def privacy_policy():
    return flask.render_template("main/privacy_policy.html")
