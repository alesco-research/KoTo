"Live browser plotting of TCLab temperatures and heater inputs via a Bokeh server."

from __future__ import annotations

import threading
import time
from collections import deque
from functools import partial
from itertools import islice
from typing import Any

from bokeh.document import Document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Title
from bokeh.plotting import figure
from bokeh.server.server import Server

COLUMNS = ("time", "T1", "T2", "U1", "U2")


class TCLabPlotter:
    """Live plotter for the TCLab.

    Parameters
    ----------
    port : int, default=5006
        Port on which the Bokeh server listens.
    title : str, default="TCLab Live"
        Title of the browser tab.
    buffer_length : int, default=1000
        Maximum number of records kept in the history and per plot; older
        records roll over.
    flush_rate : int, default=500
        Interval in milliseconds at which new records are pushed to each open
        browser session.
    """

    def __init__(
        self,
        port: int = 5006,
        title: str = "TCLab Live",
        buffer_length: int = 1000,
        flush_rate: int = 500,
    ):
        self.port = port
        self.title = title
        self.buffer_length = buffer_length
        self.flush_rate = flush_rate

        self._t0 = 0
        self._lock = threading.Lock()
        self._history: deque[tuple[float, ...]] = deque(maxlen=buffer_length)
        self._n_records = 0

        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> TCLabPlotter:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def start(self) -> None:
        "Start Bokeh server and open the plot in the browser."
        if hasattr(self, "_server"):
            return

        self._t0 = time.perf_counter()
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self.stop()
            raise RuntimeError(f"Bokeh server failed to start on port {self.port}")
        print(f"Live plot available at http://localhost:{self.port}")

    def update(self, temp1: float, temp2: float, inp1: float, inp2: float) -> None:
        """Add one record to the plot history.

        Parameters
        ----------
        temp1 : float
            Temperature of sensor 1 in °C.
        temp2 : float
            Temperature of sensor 2 in °C.
        inp1 : float
            Power of heater 1 in percent.
        inp2 : float
            Power of heater 2 in percent.
        """
        record = (time.perf_counter() - self._t0, temp1, temp2, inp1, inp2)
        with self._lock:
            self._history.append(record)
            self._n_records += 1

    def stop(self) -> None:
        "Stop plotting and shut down the Bokeh server."
        if hasattr(self, "_server"):
            self._server.io_loop.add_callback(self._stop_server)
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _create_figures(self, doc: Document) -> None:
        "Build the plot layout for a new browser session."
        doc.title = self.title

        data, cursor = self._get_data_since(0)
        source = ColumnDataSource(data)

        layout: dict[str, Any] = dict(
            height=300,
            align="center",
            sizing_mode="stretch_width",
            x_axis_label="Time [s]",
            toolbar_location="right",
        )

        styles: dict[str, Any] = dict(
            width="85%",
            position="absolute",
            top="50%",
            left="50%",
            transform="translate(-50%, -50%)",
        )

        panels = (
            ("Heater temperatures", "Temperature [°C]", "°C", ("T1", "T2"), False),
            ("Heater inputs", "Power [%]", "%", ("U1", "U2"), True),
        )

        plots = []
        for title, y_label, unit, columns, step in panels:
            p = figure(title=Title(text=title, text_font_size="15pt"), **layout)
            glyph = partial(p.step, mode="after") if step else p.line
            for col, color in zip(columns, ("#4878CF", "#9B59B6"), strict=True):
                glyph(
                    "time",
                    col,
                    source=source,
                    color=color,
                    legend_label=f"{col} [{unit}]",
                    line_width=2,
                )
            p.yaxis.axis_label = y_label
            p.legend.location = "top_left"
            p.axis.axis_label_text_font_size = "14pt"
            p.axis.major_label_text_font_size = "12pt"
            p.legend.label_text_font_size = "12pt"
            plots.append(p)

        plots[1].x_range = plots[0].x_range

        def flush() -> None:
            "Stream the records this session has not seen yet."
            nonlocal cursor
            new, cursor = self._get_data_since(cursor)
            if new["time"]:
                source.stream(new, rollover=self.buffer_length)

        doc.add_root(column(*plots, sizing_mode="stretch_width", styles=styles))
        doc.add_periodic_callback(flush, period_milliseconds=self.flush_rate)

    def _get_data_since(self, cursor: int) -> tuple[dict[str, Any], int]:
        """Return all records from ``cursor`` on and update ``cursor``.

        ``cursor`` is the number of records a session has already seen.
        """
        with self._lock:
            n_new = min(self._n_records - cursor, self.buffer_length)
            records = list(islice(self._history, len(self._history) - n_new, None))
            cursor = self._n_records

        return {c: [r[i] for r in records] for i, c in enumerate(COLUMNS)}, cursor

    def _run(self) -> None:
        "Start and run the Bokeh server in a background thread."
        self._server = Server({"/": self._create_figures}, port=self.port)
        self._server.start()
        self._server.show("/")
        self._ready.set()
        self._server.io_loop.start()

    def _stop_server(self) -> None:
        "Stop the Tornado server and its event loop."
        self._server.stop()
        self._server.io_loop.stop()
