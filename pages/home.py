from dataclasses import asdict, dataclass
from enum import Enum, auto
from dash import html, dcc, callback, ctx, Input, Output, State
import dash_bootstrap_components as dbc
import dash
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import numpy as np

dash.register_page(__name__, path="/")


class ids(Enum):
    x_value = auto()
    measurement_value = auto()
    submit_measurement = auto()
    clear_measurement = auto()
    progress_plot = auto()
    goal_value = auto()
    goal_time_value = auto()
    submit_goal = auto()

    def id(self, index=None):
        if index is None:
            return dict(type=self.name, index=self.value)
        return dict(type=self.name, index=index)


measurement_form = dbc.Form(
    dbc.Row(
        [
            dbc.Label("X value", width="auto"),
            dbc.Col(
                dbc.Input(
                    id=ids.x_value.id(),
                    type="number",
                    placeholder="Enter x value",
                    value=1,
                ),
                className="me-3",
            ),
            dbc.Label("Measured value", width="auto"),
            dbc.Col(
                dbc.Input(
                    id=ids.measurement_value.id(),
                    type="number",
                    placeholder="Enter measurement",
                    value=0,
                ),
                className="me-3",
            ),
            dbc.Col(
                dbc.Button("Submit", id=ids.submit_measurement.id(), color="primary"),
                width="auto",
            ),
            dbc.Col(
                dbc.Button("Clear", id=ids.clear_measurement.id(), color="secondary"),
                width="auto",
            ),
        ],
        className="g-2",
    ),
    class_name="m-2",
)
goal_form = dbc.Form(
    dbc.Row(
        [
            dbc.Label("Goal", width="auto"),
            dbc.Col(
                dbc.Input(
                    id=ids.goal_value.id(),
                    type="number",
                    placeholder="Enter goalvalue",
                    value=10,
                ),
                className="me-3",
            ),
            dbc.Label("Time to goal", width="auto"),
            dbc.Col(
                dbc.Input(
                    id=ids.goal_time_value.id(),
                    type="number",
                    readonly=True,
                ),
                className="me-3",
            ),
            dbc.Col(
                dbc.Button("Update Goal", id=ids.submit_goal.id(), color="danger"),
                width="auto",
                className="g-2",
            ),
        ],
        class_name="m-2",
    ),
)
layout = (
    html.Div(
        [
            dbc.Row(dbc.Col(measurement_form, width=6), justify="center"),
            dbc.Row(dbc.Col(goal_form, width=6), justify="center"),
            dcc.Graph(
                id=ids.progress_plot.id(),
                figure=go.Figure(
                    go.Scatter(
                        x=[0, 1, 2], y=[0, 3, 2], mode="markers", name="Measurements"
                    )
                ),
            ),
        ],
        className="background",
    ),
)


@callback(
    dict(
        plot=Output(ids.progress_plot.id(), "figure"),
        x_value=Output(ids.x_value.id(), "value"),
        goal_time_value=Output(ids.goal_time_value.id(), "value"),
    ),
    dict(
        submit=Input(ids.submit_measurement.id(), "n_clicks"),
        submit_goal=Input(ids.submit_goal.id(), "n_clicks"),
        clear=Input(ids.clear_measurement.id(), "n_clicks"),
        x_value=State(ids.x_value.id(), "value"),
        measurement_value=State(ids.measurement_value.id(), "value"),
        goal_value=State(ids.goal_value.id(), "value"),
        plot=State(ids.progress_plot.id(), "figure"),
    ),
)
def update_progress_plot(**kwargs):
    @dataclass
    class update:
        plot: ... = dash.no_update
        x_value: ... = dash.no_update
        goal_time_value: ... = dash.no_update

    # Update plot with measurement
    if (
        ctx.triggered_id == ids.submit_measurement.id()
        and kwargs["x_value"] is not None
        and kwargs["measurement_value"] is not None
    ):
        # Auto increment x and append measurements
        while kwargs["plot"]["data"][0]["x"][-1] >= kwargs["x_value"]:
            kwargs["x_value"] += 1
        kwargs["plot"]["data"][0]["x"].append(kwargs["x_value"])
        kwargs["plot"]["data"][0]["y"].append(kwargs["measurement_value"])

    if ctx.triggered_id in [ids.submit_measurement.id(), ids.submit_goal.id()]:
        # Fit curve to measurements
        measurement_trace = go.Figure(kwargs["plot"]["data"][0])
        xdata = np.array(kwargs["plot"]["data"][0]["x"])
        ydata = kwargs["plot"]["data"][0]["y"]

        def fit_func(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(fit_func, xdata, ydata)

        # Determine time to goal
        goal = kwargs["goal_value"]
        goal_time = (goal - popt[1]) / popt[0]

        # Add trace for goal
        measurement_trace.add_hline(
            y=goal, annotation_text="Goal", annotation_position="bottom right"
        )

        # Add trace with curve fit
        measurement_trace.add_scatter(x=xdata, y=fit_func(xdata, *popt), name="Fit")
        measurement_trace.add_scatter(
            x=np.linspace(0, goal_time),
            y=fit_func(np.linspace(0, goal_time), *popt),
            name="Prediction",
        )

        return asdict(
            update(
                plot=measurement_trace,
                x_value=kwargs["x_value"],
                goal_time_value=round(goal_time),
            )
        )

    # Clear plot measurements
    if ctx.triggered_id == ids.clear_measurement.id():
        kwargs["plot"]["data"][0]["x"] = [0]
        kwargs["plot"]["data"][0]["y"] = [0]
        return asdict(update(plot=kwargs["plot"]))

    return asdict(update())
