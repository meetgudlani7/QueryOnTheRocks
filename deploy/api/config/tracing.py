"""
Tracing Configuration (roadmap Phase 24)

Optional per-stage OpenTelemetry tracing, replacing "grep the local log
correlated by request_id" as the only way to diagnose where a request's
latency actually went. Fully graceful when the opentelemetry packages
aren't installed or TRACING_ENABLED isn't set: span() becomes a no-op
context manager in that case, so nothing about the core pipeline's
behavior, latency, or dependency footprint changes unless a deployment
explicitly opts in — mirrors this repo's existing pattern of keeping
heavier integrations (datasets, embeddings) commented-optional in
requirements.txt until actually needed.

Enable with TRACING_ENABLED=true. Exports to the console by default (zero
external infrastructure required to see it working); set
OTEL_EXPORTER_OTLP_ENDPOINT to export to a real backend instead (Jaeger,
Honeycomb, Grafana Tempo, or anything else that speaks OTLP).
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from config import settings

logger = logging.getLogger(__name__)

_tracer = None
_configured = False


def _configure() -> None:
    """Lazy, one-time setup — deferred to first use rather than import time, so importing this module never has side effects."""
    global _tracer, _configured
    if _configured:
        return
    _configured = True

    if not settings.TRACING_ENABLED:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.warning(
            "TRACING_ENABLED=true but opentelemetry packages aren't installed — run "
            "`pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http` "
            "(see requirements.txt). Tracing stays disabled; nothing else is affected."
        )
        return

    resource = Resource.create({"service.name": "voice-rag-api"})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
            logger.info(f"Tracing enabled, exporting spans to OTLP endpoint {otlp_endpoint}")
        except ImportError:
            logger.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp-proto-http "
                "isn't installed — falling back to console export."
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("Tracing enabled, exporting spans to console (set OTEL_EXPORTER_OTLP_ENDPOINT for a real backend)")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("voice-rag-pipeline")


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """
    Wraps a block in a tracing span when tracing is enabled and
    configured; a complete no-op otherwise. Safe to sprinkle through the
    pipeline unconditionally — the disabled-path cost is one dict lookup
    and an early return, not a branch around every call site.
    """
    _configure()
    if _tracer is None:
        yield
        return

    with _tracer.start_as_current_span(name) as current_span:
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
        yield
