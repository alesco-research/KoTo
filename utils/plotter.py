"Live browser plotting of TCLab temperatures and heater inputs via a Bokeh server."

from __future__ import annotations

import threading
import time
from collections import deque
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
        Maximum number of samples kept in the history and per plot; older
        samples roll over.
    flush_rate : int, default=500
        Interval in milliseconds at which new samples are pushed to each open
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
        self._n_samples = 0
        self._stopped = False

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
        """Add one sample to the plot history.

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
        sample = (time.perf_counter() - self._t0, temp1, temp2, inp1, inp2)
        with self._lock:
            self._history.append(sample)
            self._n_samples += 1

    def stop(self) -> None:
        "Stop plotting and shut down the Bokeh server."
        self._stopped = True
        if hasattr(self, "_server"):
            self._server.io_loop.add_callback(self._stop_server)
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _create_figure(self, doc: Document) -> None:
        "Build the plot layout for a new browser session."
        doc.title = self.title

        data, cursor = self._samples_since(0)
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

        p_temp = figure(
            title=Title(text="Heater temperatures", text_font_size="15pt"),
            name="temperatures",
            **layout,
        )
        p_temp.line(
            "time",
            "T1",
            source=source,
            color="#4878CF",
            legend_label="T1 [°C]",
            line_width=2,
        )
        p_temp.line(
            "time",
            "T2",
            source=source,
            color="#9B59B6",
            legend_label="T2 [°C]",
            line_width=2,
        )
        p_temp.yaxis.axis_label = "Temperature [°C]"
        p_temp.legend.location = "top_left"

        p_inp = figure(
            title=Title(text="Heater inputs", text_font_size="15pt"),
            x_range=p_temp.x_range,
            name="inputs",
            **layout,
        )
        p_inp.step(
            "time",
            "U1",
            source=source,
            color="#4878CF",
            legend_label="U1 [%]",
            line_width=2,
            mode="after",
        )
        p_inp.step(
            "time",
            "U2",
            source=source,
            color="#9B59B6",
            legend_label="U2 [%]",
            line_width=2,
            mode="after",
        )
        p_inp.yaxis.axis_label = "Power [%]"
        p_inp.legend.location = "top_left"

        for p in (p_temp, p_inp):
            p.axis.axis_label_text_font_size = "14pt"
            p.axis.major_label_text_font_size = "12pt"
            p.legend.label_text_font_size = "12pt"

        def flush() -> None:
            "Stream the samples this session has not seen yet."
            nonlocal cursor
            if self._stopped:
                return

            new, cursor = self._samples_since(cursor)
            if new["time"]:
                source.stream(new, rollover=self.buffer_length)

        doc.add_root(column(p_temp, p_inp, sizing_mode="stretch_width", styles=styles))
        doc.add_periodic_callback(flush, period_milliseconds=self.flush_rate)

    def _samples_since(self, cursor: int) -> tuple[dict[str, Any], int]:
        """Return the samples recorded after ``cursor`` and update the cursor.

        ``cursor`` is the number of samples a session has already seen.
        """
        with self._lock:
            n_new = min(self._n_samples - cursor, self.buffer_length)
            samples = list(islice(self._history, len(self._history) - n_new, None))
            cursor = self._n_samples

        return {c: [s[i] for s in samples] for i, c in enumerate(COLUMNS)}, cursor

    def _run(self) -> None:
        "Start and run the Bokeh server in a background thread"
        self._server = Server({"/": self._create_figure}, port=self.port, num_procs=1)
        self._server.start()
        self._server.io_loop.add_callback(self._open_browser)
        self._server.io_loop.start()

    def _open_browser(self) -> None:
        "Open the plot in the browser and signal that the page is ready."
        self._server.show("/")
        self._ready.set()

    def _stop_server(self) -> None:
        "Stop the Tornado server and its event loop."
        self._server.stop()
        self._server.io_loop.stop()
