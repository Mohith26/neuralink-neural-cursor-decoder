"""FastAPI streaming decode service: feed a bin of spike counts -> get velocity.

Stateful: the Kalman filter carries state across /decode calls, exactly like a
real-time BCI loop would. This is OFFLINE decode of recorded/synthetic bins, not
a live closed loop with a subject.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .kalman import KalmanVelocityDecoder


class DecodeState:
    """Holds the fitted decoder and integrated cursor position for a session."""

    def __init__(self) -> None:
        self.decoder: KalmanVelocityDecoder | None = None
        self.n_units: int = 0
        self.bin_s: float = 0.0
        self.kind: str = "UNINITIALIZED"
        self.position = np.zeros(2)

    def init(self, decoder: KalmanVelocityDecoder, n_units: int, bin_s: float,
             kind: str, init_position: np.ndarray) -> None:
        self.decoder = decoder
        self.n_units = n_units
        self.bin_s = bin_s
        self.kind = kind
        self.position = np.asarray(init_position, dtype=np.float64).copy()
        decoder.reset(self.position)


STATE = DecodeState()


class BinRequest(BaseModel):
    counts: list[float] = Field(..., description="per-unit spike counts for one bin")


class DecodeResponse(BaseModel):
    velocity: list[float]
    position: list[float]
    data_kind: str


def create_app(state: DecodeState = STATE) -> FastAPI:
    app = FastAPI(title="NeuroCursor decode service", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if state.decoder is not None else "uninitialized",
            "n_units": state.n_units,
            "bin_s": state.bin_s,
            "data_kind": state.kind,
        }

    @app.post("/reset")
    def reset(position: list[float] | None = None) -> dict:
        if state.decoder is None:
            raise HTTPException(503, "decoder not initialized")
        pos = np.asarray(position, dtype=np.float64) if position else state.position.copy()
        state.position = pos
        state.decoder.reset(pos)
        return {"status": "reset", "position": pos.tolist()}

    @app.post("/decode", response_model=DecodeResponse)
    def decode(req: BinRequest) -> DecodeResponse:
        if state.decoder is None:
            raise HTTPException(503, "decoder not initialized")
        counts = np.asarray(req.counts, dtype=np.float64)
        if counts.shape[0] != state.n_units:
            raise HTTPException(
                422, f"expected {state.n_units} unit counts, got {counts.shape[0]}"
            )
        vel = state.decoder.step(counts)
        state.position = state.position + vel * state.bin_s
        return DecodeResponse(
            velocity=vel.tolist(), position=state.position.tolist(), data_kind=state.kind
        )

    return app


app = create_app()
